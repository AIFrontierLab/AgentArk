import argparse
from datetime import datetime
import os
import math
import torch
from transformers import get_scheduler

from openrlhf.utils import get_strategy, blending_datasets, get_tokenizer
from openrlhf.models import get_llm_for_sequence_regression, Actor
from openrlhf.trainer import PURETrainer  # Use base PURETrainer, not Ray-specific version
from openrlhf.datasets import PromptDataset, SFTDataset


def _validate_args(args):
    # For non-Ray version, we use DeepSpeed's distributed training
    # No need to validate actor_world_size for Ray

    assert args.gamma == 1, "Only gamma=1 is supported for PURE training"
    assert not args.packing_samples, "Packing samples not supported in non-Ray version"
    assert args.advantage_estimator == "rloo", "Only RLOO advantage estimator is supported"

    # Disable vLLM in non-Ray version for simplicity
    if args.vllm_num_engines and args.vllm_num_engines > 0:
        print("[Warning] vLLM is not supported in non-Ray version, disabling it")
        args.vllm_num_engines = 0


def train(args):
    _validate_args(args)

    # configure strategy (DeepSpeed)
    strategy = get_strategy(args)

    # Initialize torch distributed if not already initialized
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')

    # Initialize actor model (the model being trained)
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )
    strategy.print(actor)

    # Get tokenizer
    tokenizer = get_tokenizer(
        args.pretrain,
        actor.model,
        "left",
        strategy,
        use_fast=not args.disable_fast_tokenizer
    )

    # EMA model (optional)
    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
            packing_samples=args.packing_samples,
        )
    else:
        ema_model = None

    # Create optimizer
    actor_optim = strategy.create_optimizer(
        actor,
        lr=args.actor_learning_rate,
        betas=args.adam_betas,
        weight_decay=args.l2
    )

    # Prepare datasets (needed for scheduler calculation)
    if args.prompt_data:
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            stopping_strategy="first_exhausted",
            split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        prompts_dataset = PromptDataset(
            prompts_data, tokenizer, strategy, input_template=args.input_template
        )
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.rollout_batch_size // strategy.world_size,
            True,
            True
        )
    else:
        raise ValueError("prompt_data is required for PURE training")

    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            stopping_strategy="first_exhausted",
            split=args.pretrain_split,
        )
        pretrain_dataset = SFTDataset(
            pretrain_data, tokenizer, args.prompt_max_len, strategy, pretrain_mode=True
        )
        pretrain_dataloader = strategy.setup_dataloader(
            pretrain_dataset,
            args.micro_train_batch_size,
            True,
            False,
        )
    else:
        pretrain_dataloader = None

    # Calculate scheduler steps
    num_update_steps_per_episodes = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
    )
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    # Create scheduler
    actor_scheduler = get_scheduler(
        "cosine",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # Prepare actor model, optimizer, and scheduler with DeepSpeed
    actor, actor_optim, actor_scheduler = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        is_rlhf=True,
    )

    # Prepare EMA model if enabled
    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

    # Initialize reference model (for KL divergence)
    reference_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_reward_offload),
        packing_samples=args.packing_samples,
    )
    reference_model = strategy.prepare(reference_model, is_rlhf=True)
    reference_model.eval()

    # Initialize reward model (process reward model)
    reward_model = get_llm_for_sequence_regression(
        args.reward_pretrain,
        "process_reward",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_reward_offload),
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
    )
    reward_model = strategy.prepare(reward_model, is_rlhf=True)
    reward_model.eval()

    # Create trainer (using base PURETrainer, not Ray-specific ActorPURETrainer)
    trainer = PURETrainer(
        strategy,
        actor,
        reward_model,
        reference_model,  # initial_actor (reference model for KL divergence)
        None,  # last_actor (not used in current PURE implementation)
        None,  # initial_reward_model (not used in current PURE implementation)
        ema_model,
        actor_optim,
        actor_scheduler,
        None,  # reward_model_optim (reward model is frozen)
        None,  # reward_model_scheduler (reward model is frozen)
        ema_beta=0.992,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ptx_coef=args.ptx_coef,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        max_epochs=args.max_epochs,
        max_norm=args.max_norm,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        eps_clip=args.eps_clip,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        # Generation kwargs
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Training loop
    trainer.fit(
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=num_update_steps_per_episodes,
    )

    # Save final model (already handled by trainer, but can save explicitly if needed)
    strategy.print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note: Ray-specific arguments removed. This version uses DeepSpeed for multi-GPU training.
    # Launch with: deepspeed --num_gpus N train_pure_no_ray.py [args]

    # vLLM support removed for simplicity in non-Ray version
    parser.add_argument(
        "--vllm_num_engines", type=int, default=0, help="(Disabled in non-Ray version)"
    )

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["reinforce", "rloo"],
        default="rloo",
        help="Choose advantage estimation method: reinforce, rloo",
    )
    parser.add_argument(
        "--reward_baseline",
        type=str,
        choices=["token", "step"],
        default="step",
        help="Use per token or per step reward as baseline",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["PRM", "PRMVR", "VR"],
        default="PRMVR",
        help="PRM is process reward, VR is verifiable reward",
    )
    parser.add_argument("--verifiable_reward_coef", type=float, default=1.0)
    parser.add_argument("--disable_advantage_normalization", action="store_true", default=False)
    parser.add_argument("--disable_weighted_reward", action="store_true", default=False)
    parser.add_argument("--nomask_separator_adv", action="store_true", default=False)

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default='{}\n\nPlease reason step by step with steps separated by "\n\n", and put your final answer within \\boxed{{}}.')
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    args = parser.parse_args()

    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    # vLLM disabled in non-Ray version
    args.vllm_num_engines = 0

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples:
        print("[Warning] Packing samples is not supported in non-Ray version, disabling it")
        args.packing_samples = False

    if 'deepseek' in args.pretrain.lower() and args.input_template:
        args.input_template += ' <think>\n'

    train(args)

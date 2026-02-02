import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig

from openrlhf.models.model import get_llm_for_sequence_regression
from openrlhf.utils.math_score import compute_score


class PRMRewardFunction:
    """Custom reward function wrapping ProcessRewardModel for TRL's GRPOTrainer."""

    def __init__(self, prm_model_path, reward_mode, tokenizer, verifiable_reward_coef=1.0,
                 temperature=0.1, disable_weighted_reward=False, device="cuda"):
        self.reward_mode = reward_mode
        self.tokenizer = tokenizer
        self.verifiable_reward_coef = verifiable_reward_coef
        self.temperature = temperature
        self.disable_weighted_reward = disable_weighted_reward
        self.device = device

        # Load PRM model if needed
        self.prm_model = None
        if "PRM" in reward_mode:
            self.prm_model = get_llm_for_sequence_regression(
                prm_model_path, "process_reward", normalize_reward=False,
                use_flash_attention_2=True, bf16=True, ds_config=None,
                value_head_prefix="score", packing_samples=False
            )
            self.prm_model.to(device).eval()

        # Step separators
        self.step_sep = '\n\n'
        self.prm_sep = '\n'
        self.prm_sep_token = tokenizer.encode(self.prm_sep, add_special_tokens=False)[0]

    def __call__(self, prompts, completions, ground_truths=None, **kwargs):
        """TRL reward function interface."""
        batch_size = len(completions)
        rewards = torch.zeros(batch_size, device=self.device)

        # Compute PRM rewards
        if "PRM" in self.reward_mode and self.prm_model:
            rewards += self._compute_prm_rewards(prompts, completions)

        # Add verifiable rewards
        if "VR" in self.reward_mode and ground_truths:
            vr_rewards = torch.tensor(
                [float(compute_score(comp, gt)) * 2 - 1 for comp, gt in zip(completions, ground_truths)],
                device=self.device
            )
            rewards += vr_rewards * self.verifiable_reward_coef

        return rewards.cpu().tolist()

    def _compute_prm_rewards(self, prompts, completions):
        """Compute process rewards using PRM model."""
        batch_size = len(completions)

        # Parse steps and reconstruct inputs for PRM
        prm_inputs = []
        score_positions = []

        for prompt, completion in zip(prompts, completions):
            # Tokenize
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            comp_ids = self.tokenizer.encode(completion, add_special_tokens=False)

            # Find step separators (\\n\\n)
            step_boundaries = []
            i = 0
            while i < len(comp_ids) - 1:
                # Check if this is a step separator pattern
                if self.tokenizer.decode(comp_ids[i:i+2]) == self.step_sep:
                    step_boundaries.append(i + 2)  # After \\n\\n
                    i += 2
                else:
                    i += 1

            # Build PRM input: prompt + step1 + \\n + step2 + \\n + ...
            input_ids = prompt_ids.copy()
            positions = []
            prev = 0

            for boundary in step_boundaries:
                # Extract step tokens (excluding \\n\\n)
                step_tokens = comp_ids[prev:boundary-2] if boundary > 2 else comp_ids[prev:boundary]
                input_ids.extend(step_tokens + [self.prm_sep_token])
                positions.append(len(input_ids) - 1)
                prev = boundary

            # Add remaining tokens if any
            if prev < len(comp_ids):
                input_ids.extend(comp_ids[prev:])

            prm_inputs.append(input_ids)
            score_positions.append(positions)

        if not prm_inputs or not any(score_positions):
            return torch.zeros(batch_size, device=self.device)

        # Pad sequences
        max_len = max(len(ids) for ids in prm_inputs)
        padded_inputs = []
        attn_masks = []

        for ids in prm_inputs:
            pad_len = max_len - len(ids)
            padded = ids + [self.tokenizer.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_inputs.append(padded)
            attn_masks.append(mask)

        input_ids = torch.tensor(padded_inputs, device=self.device)
        attention_mask = torch.tensor(attn_masks, device=self.device)

        # Forward through PRM
        with torch.no_grad():
            logits = self.prm_model(input_ids, attention_mask)  # [B, L, 2]

        # Extract logits at score positions
        max_steps = max(len(pos) for pos in score_positions)
        if max_steps == 0:
            return torch.zeros(batch_size, device=self.device)

        score_logits = torch.zeros(batch_size, max_steps, 2, device=self.device)
        score_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=self.device)

        for i, positions in enumerate(score_positions):
            for j, pos in enumerate(positions):
                if pos < logits.size(1):
                    score_logits[i, j] = logits[i, pos]
                    score_mask[i, j] = True

        # Convert to rewards using weighted aggregation
        step_rewards = self._turn_logits_to_rewards(score_logits, score_mask)
        outcome_rewards = step_rewards.sum(dim=1)

        return outcome_rewards

    def _turn_logits_to_rewards(self, logits, mask):
        """Convert logits to rewards with weighted aggregation."""
        # Softmax to get P(correct) - P(incorrect)
        softmax_logits = F.softmax(logits, dim=-1)
        rewards = softmax_logits[..., 1] - softmax_logits[..., 0]
        rewards = rewards.masked_fill(~mask, 0)

        # Weighted aggregation (emphasize low rewards)
        if not self.disable_weighted_reward:
            weights = F.softmax(
                -rewards.masked_fill(~mask, float("inf")) / self.temperature,
                dim=-1
            )
        else:
            weights = F.softmax(
                torch.ones_like(rewards).masked_fill(~mask, -float("inf")),
                dim=-1
            )

        return weights * rewards


def preprocess_dataset(dataset, args):
    """Preprocess dataset to TRL format."""
    def format_example(example):
        prompt = example.get('prompt') or example.get('question', '')
        if args.input_template:
            prompt = args.input_template.format(prompt)

        gt = example.get('answer') or example.get('gt', 'null')
        return {
            'prompt': prompt,
            'ground_truth': gt if gt != 'null' else None
        }

    return dataset.map(format_example, remove_columns=[c for c in dataset.column_names if c not in ['prompt', 'ground_truth']])


def create_grpo_config(args):
    """Create GRPOConfig from args."""
    return GRPOConfig(
        output_dir=args.save_path,
        num_train_epochs=args.num_episodes,
        per_device_train_batch_size=args.micro_rollout_batch_size,
        gradient_accumulation_steps=max(1, args.train_batch_size // (args.micro_rollout_batch_size * (torch.cuda.device_count() or 1))),
        learning_rate=args.actor_learning_rate,
        warmup_ratio=args.lr_warmup_ratio,
        max_grad_norm=args.max_norm,

        # Generation params
        max_prompt_length=args.prompt_max_len,
        max_length=args.prompt_max_len + args.generate_max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        num_generations=args.n_samples_per_prompt,

        # GRPO-specific
        beta=args.init_kl_coef if args.use_kl_loss else 0.0,
        scale_rewards="group",

        # Optimization
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,

        # Logging and saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.max_ckpt_num,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name if args.use_wandb else None,

        # Misc
        seed=args.seed,
    )


def train(args):
    """Main training function using TRL GRPOTrainer."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain,
        trust_remote_code=True,
        use_fast=not args.disable_fast_tokenizer
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )

    # Load dataset
    dataset = load_dataset(args.prompt_data, split=args.prompt_split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Preprocess
    dataset = preprocess_dataset(dataset, args)

    # Create reward function
    reward_fn = PRMRewardFunction(
        prm_model_path=args.reward_pretrain,
        reward_mode=args.reward_mode,
        tokenizer=tokenizer,
        verifiable_reward_coef=args.verifiable_reward_coef,
        temperature=0.1,
        disable_weighted_reward=args.disable_weighted_reward,
    )

    # Create config
    training_args = create_grpo_config(args)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--pretrain", type=str, required=True)
    parser.add_argument("--reward_pretrain", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./outputs/grpo")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)

    # Batch sizes
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)

    # Generation
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n_samples_per_prompt", type=int, default=4)

    # Optimization
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument("--use_kl_loss", action="store_true", default=True)
    parser.add_argument("--no_use_kl_loss", action="store_true", default=False)

    # Reward
    parser.add_argument("--reward_mode", type=str, choices=["PRM", "VR", "PRMVR"], default="PRMVR")
    parser.add_argument("--reward_baseline", type=str, choices=["token", "step"], default="step")
    parser.add_argument("--verifiable_reward_coef", type=float, default=1.0)
    parser.add_argument("--disable_weighted_reward", action="store_true", default=False)
    parser.add_argument("--advantage_estimator", type=str, default="rloo")

    # Dataset
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--input_template", type=str, default=None)

    # Optimization flags
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=f"grpo_{datetime.now().strftime('%m%dT%H:%M')}")
    parser.add_argument("--seed", type=int, default=42)

    # Deprecated Ray args (ignored)
    parser.add_argument("--actor_num_nodes", type=int, default=1, help="[Deprecated] Use accelerate config")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=1, help="[Deprecated] Use accelerate config")
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="[Deprecated] Not needed with TRL")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=1, help="[Deprecated] Not needed with TRL")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="[Deprecated] Not needed with TRL")
    parser.add_argument("--reward_num_gpus_per_node", type=int, default=1, help="[Deprecated] Not needed with TRL")
    parser.add_argument("--vllm_num_engines", type=int, default=None, help="[Deprecated] Use --use_vllm in config")

    args = parser.parse_args()

    # Handle use_kl_loss flag
    if args.no_use_kl_loss:
        args.use_kl_loss = False

    # Dataset-specific handling
    if args.prompt_data and ("QMSum" in args.prompt_data or "QASPER" in args.prompt_data or "HotpotQA" in args.prompt_data):
        if args.input_template and "boxed" in args.input_template:
            print("[Info] QMSum/QASPER/HotpotQA detected. Overriding input_template.")
        args.input_template = "Keep both your answer and reasoning concise and to the point. do not generate irrelevant information."

    # RLOO validation
    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    train(args)

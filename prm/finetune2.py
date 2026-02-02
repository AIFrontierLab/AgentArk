import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers
from datasets import load_dataset
from torch import distributed as dist
import torch.nn as nn

from trl import PRMConfig, PRMTrainer
import torch
import torch.nn.functional as F
import logging
import sys
from transformers import TrainerCallback
from datasets import Features, Value, Sequence, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from misc_utils import print_colored
from utils.utils import read_jsonl



# Configure logging for NaN debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nan_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_nan_inf(tensor, name, step=None):
    """Check for NaN/Inf in tensors and log detailed diagnostics."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        prefix = f"[Step {step}] " if step is not None else ""
        logger.error(f"{prefix}{'NaN' if has_nan else 'Inf'} detected in {name}!")
        logger.error(f"  Shape: {tensor.shape}")
        logger.error(f"  NaN count: {torch.isnan(tensor).sum().item()}")
        logger.error(f"  Inf count: {torch.isinf(tensor).sum().item()}")

        # Get min/max of valid values
        valid_mask = ~torch.isnan(tensor) & ~torch.isinf(tensor)
        if valid_mask.any():
            logger.error(f"  Min (valid): {tensor[valid_mask].min().item()}")
            logger.error(f"  Max (valid): {tensor[valid_mask].max().item()}")
        else:
            logger.error(f"  All values are NaN or Inf!")
        return True
    return False


def log_tensor_stats(tensor, name, step=None):
    """Log detailed statistics of a tensor."""
    prefix = f"[Step {step}] " if step is not None else ""

    # Filter out NaN/Inf for statistics
    valid_tensor = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]

    stats = {
        'shape': tensor.shape,
        'mean': valid_tensor.mean().item() if valid_tensor.numel() > 0 else float('nan'),
        'std': valid_tensor.std().item() if valid_tensor.numel() > 0 else float('nan'),
        'min': valid_tensor.min().item() if valid_tensor.numel() > 0 else float('nan'),
        'max': valid_tensor.max().item() if valid_tensor.numel() > 0 else float('nan'),
        'nan_count': torch.isnan(tensor).sum().item(),
        'inf_count': torch.isinf(tensor).sum().item(),
    }

    logger.info(f"{prefix}{name} stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
                f"nan={stats['nan_count']}, inf={stats['inf_count']}")

    return stats


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B")


@dataclass
class DataArguments:
    train_data_path: str = field(default="trl-lib/math_shepherd")
    eval_data_path: str = field(default=None)
    lazy_preprocess: bool = False
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(PRMConfig):
    cache_dir: Optional[str] = field(default=None)
    max_length: int = field(default=128000)
    max_completion_length: int = field(default=8000)
    fix_llm: bool = field(default=False)

    # NaN monitoring flags
    enable_nan_monitoring: bool = field(default=True, metadata={"help": "Enable comprehensive NaN monitoring"})
    nan_check_interval: int = field(default=50, metadata={"help": "Check parameters for NaN every N steps"})


def safe_save_model_for_hf_trainer(
        trainer: transformers.Trainer,
        output_dir: str,
    ):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()

    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert data_args.train_data_path is not None
    
    
    if "QASPER" in data_args.train_data_path or "QMSum" in data_args.train_data_path or "HotpotQA" in data_args.train_data_path:
        
        train_dataset = read_jsonl(data_args.train_data_path)
        # Filter and transform data format to match PRMTrainer expectations
        original_length = len(train_dataset)
        train_dataset = [
            {
                'prompt': data['query'],
                'completions': [sol['text'] for sol in data['solutions']],
                'labels': data['labels']
            }
            for data in train_dataset
            if (True in data["labels"] and False in data["labels"])
        ]
        train_dataset = Dataset.from_list(train_dataset)
        print(f"Retain {len(train_dataset)}/{original_length} samples.")
        eval_dataset = None
    
    elif data_args.train_data_path.endswith(".jsonl"):
        train_dataset = load_dataset("json", data_files=data_args.train_data_path)
        eval_dataset = None
    
    else:

        train_dataset = load_dataset(data_args.train_data_path, split="train")
        
        try:
            eval_dataset = load_dataset(data_args.train_data_path, split="test")
            
        except Exception as e:
            print_colored(f"Error loading eval dataset. Sampling train dataset for validation. {e}", "red")
            eval_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_eval_samples)))

        # Limit the number of samples if specified
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )



def contrastive_loss_fn_original(logits, labels, temperature=0.1, **kwargs):
    """
    Contrastive loss for PRM token-level labels.
    
    Args:
        outputs: model outputs (logits or embeddings)
        labels: tensor of shape [batch, seq] with 1 for positive, 0 for negative, -100 for padding
        temperature: scaling factor for similarity
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    
    # Normalize the high-dimensional embeddings
    embeddings = F.normalize(logits, dim=-1)

    # similarity matrix
    sim_matrix = embeddings @ embeddings.T   
    sim_matrix = sim_matrix / temperature

    # create positive mask: 1 if labels match
    positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0) 

    # InfoNCE-style loss
    exp_sim = torch.exp(sim_matrix)       
    
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.diag(sim_matrix)).unsqueeze(1)
    
    # compute log-probabilities for positives
    log_prob = torch.log((exp_sim * positive_mask.float()).sum(dim=1) / exp_sim_sum.squeeze(1))
    
    # final loss
    loss = -log_prob.mean()
    return loss


def contrastive_loss_fn(embeddings, labels, temperature=0.5, step=None, enable_monitoring=True, **kwargs):
    """
    Numerically stable contrastive loss for PRM token-level labels.

    Uses log-sum-exp trick (via F.log_softmax) for numerical stability in BF16.
    Clamping prevents exp() overflow (BF16 threshold: ~88.7) while staying in native precision.

    Args:
        outputs: model outputs (logits or embeddings)
        labels: tensor of shape [batch, seq] with 1 for positive, 0 for negative, -100 for padding
        temperature: scaling factor for similarity
        step: current training step for logging
        enable_monitoring: whether to enable NaN checks and logging

    Returns:
        loss: scalar tensor
    """

    # Check for empty batch
    if embeddings.numel() == 0:
        if enable_monitoring:
            logger.warning(f"[Step {step}] Empty embeddings passed to loss!")
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # 1. Normalize high-dim embeddings for cosine similarity
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.T  # [N, N]

    # Temperature scaling in native BF16 with safety clamping
    # BF16 can handle exp(x) up to x≈88.7. With temp=0.1, normalized embeddings
    # produce sim/temp ∈ [-10, 10], well within safe range. 
    # Clamp to ±85 as safety margin.
    sim_matrix_scaled = torch.clamp(sim_matrix / temperature, min=-85.0, max=85.0)

    # Create masks
    positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]
    batch_size = sim_matrix_scaled.size(0)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix_scaled.device)

    # CRITICAL FIX: Exclude self-pairs from positive mask to avoid -inf in loss
    # Without this, positive_mask[i,i]=True combined with log_probs[i,i]=-inf causes loss=inf
    positive_mask = positive_mask & ~self_mask

    # Exclude self-similarity from denominator (mask with -inf)
    sim_for_denominator = sim_matrix_scaled.masked_fill(self_mask, float('-inf'))

    # Compute stable log probabilities using PyTorch built-in
    log_probs = F.log_softmax(sim_for_denominator, dim=1)  # [N, N]

    # Compute log probability of positive pairs
    # Use where() instead of multiplication to avoid -inf * 0 = nan
    log_pos_probs = torch.where(
        positive_mask,
        log_probs,
        torch.zeros_like(log_probs)
    )

    # Handle samples with no positive pairs
    num_positives = positive_mask.sum(dim=1).float()  # [N]
    has_positives = num_positives > 0

    # Average log probability over positive pairs
    avg_log_pos_prob = torch.where(
        has_positives,
        log_pos_probs.sum(dim=1) / num_positives.clamp(min=1),
        torch.zeros_like(num_positives)
    )

    # Final loss: negative mean of log probabilities
    if has_positives.any():
        loss = -avg_log_pos_prob[has_positives].mean()
    else:
        if enable_monitoring:
            logger.warning(f"[Step {step}] No positive pairs in batch!")
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"[Step {step}] NaN/Inf detected in final loss!")
        logger.error(f"  Loss value: {loss.item()}")
        logger.error(f"  Batch size: {batch_size}")
        logger.error(f"  Samples with positives: {has_positives.sum().item()}")
        logger.error(f"  Temperature: {temperature}")

        # Save problematic batch
        if step is not None:
            import os
            debug_dir = "nan_debug"
            os.makedirs(debug_dir, exist_ok=True)
            torch.save({
                'step': step,
                'embeddings': embeddings.cpu(),
                'labels': labels.cpu(),
                'sim_matrix': sim_matrix.cpu(),
                'positive_mask': positive_mask.cpu(),
                'loss': loss.cpu(),
            }, f'{debug_dir}/nan_batch_step_{step}.pt')
            logger.error(f"Saved batch to {debug_dir}/nan_batch_step_{step}.pt")

        raise ValueError(f"NaN/Inf detected in final loss at step {step}")

    # Periodic logging (reduced to every 100 steps)
    if enable_monitoring and step is not None and step % 100 == 0:
        logger.info(f"[Step {step}] Loss: {loss.item():.6f}, "
                f"Batch size: {batch_size}, "
                f"Samples with positives: {has_positives.sum().item()}")

    return loss


class PRMTrainerWithContrastive(PRMTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure projection head is on the same device and dtype as the model
     
        self.nan_detected = False
        self.gradient_nan_detected = False
        
        # Add a projection head for contrastive learning 
        # This maps the hidden dim to a dedicated space for contrastive separation
        hidden_dim = self.model.config.hidden_size
        self.contrastive_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.model.device).to(self.model.dtype)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)

        logits = outputs.logits.squeeze(-1)
        mask = labels != -100
        
        masked_logits = logits[mask] 
        masked_labels = labels[mask]
        
        bce_loss = F.cross_entropy(
            masked_logits, 
            masked_labels
        )

        # 2. Contrastive Loss using Hidden States
        # last_hidden_state: [batch, seq, hidden_dim]
        hidden_states = outputs.hidden_states[-1] 
        masked_hidden = hidden_states[mask]
        
        # Project hidden states to the contrastive space
        projected_embeddings = self.contrastive_projector(masked_hidden)
        
        step = self.state.global_step if hasattr(self, 'state') else None
        
        # Now embeddings have high dimensionality (e.g., 896 or 4096)
        # instead of just 1.
        con_loss = contrastive_loss_fn(
            projected_embeddings, 
            masked_labels,
            temperature=0.1,
            step=step,
            enable_monitoring=self.args.logging_steps > 0
        )

        # 3. Combined Loss
        loss = bce_loss + 0.5 * con_loss 

        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_detected = True
            logger.error(f"\n{'!'*80}")
            logger.error(f"NaN/Inf LOSS DETECTED at step {step}!")
            logger.error(f"{'!'*80}\n")

            # Save the problematic batch for analysis
            debug_dir = os.path.join(self.args.output_dir, "nan_debug")
            os.makedirs(debug_dir, exist_ok=True)

            torch.save({
                'step': step,
                'labels': labels,
                'inputs': inputs,
                'outputs': outputs.logits if hasattr(outputs, 'logits') else outputs,
                'loss': loss,
            }, os.path.join(debug_dir, f'nan_batch_step_{step}.pt'))

            logger.error(f"Saved problematic batch to {debug_dir}/nan_batch_step_{step}.pt")

            # Stop training
            raise ValueError(f"NaN/Inf detected in loss at step {step}. Training stopped.")

        return (loss, outputs) if return_outputs else loss
        
    def training_step(self, model, inputs, num_items_in_batch, **kwargs):
        """Override to add gradient monitoring."""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)
        # print(loss)

        # Check gradients after backward pass
        if self.state.global_step % self.args.logging_steps == 0:
            self._check_gradients(model)

        return loss

    def _check_gradients(self, model):
        """Check for NaN/Inf in gradients."""
        step = self.state.global_step
        grad_norms = {}
        nan_grads = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm

                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_grads.append(name)
                    logger.error(f"NaN/Inf gradient in {name} at step {step}")

        if nan_grads:
            self.gradient_nan_detected = True
            logger.error(f"\n{'!'*80}")
            logger.error(f"NaN/Inf GRADIENTS DETECTED at step {step}!")
            logger.error(f"Affected parameters: {nan_grads}")
            logger.error(f"{'!'*80}\n")
            raise ValueError(f"NaN/Inf detected in gradients at step {step}")

        # Log gradient norms periodically
        if step % (self.args.logging_steps * 10) == 0:
            max_grad_norm = max(grad_norms.values()) if grad_norms else 0
            min_grad_norm = min(grad_norms.values()) if grad_norms else 0
            avg_grad_norm = sum(grad_norms.values()) / len(grad_norms) if grad_norms else 0

            logger.info(f"\n[Step {step}] Gradient Statistics:")
            logger.info(f"  Max gradient norm: {max_grad_norm:.6e}")
            logger.info(f"  Min gradient norm: {min_grad_norm:.6e}")
            logger.info(f"  Avg gradient norm: {avg_grad_norm:.6e}")


class NaNMonitorCallback(TrainerCallback):
    """Callback to monitor for NaN in parameters and log additional metrics."""

    def __init__(self, check_interval=50):
        self.last_check_step = 0
        self.check_interval = check_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after logging."""
        if logs is not None and 'loss' in logs:
            if logs['loss'] != logs['loss']:  # NaN check (NaN != NaN)
                logger.error(f"\n{'!'*80}")
                logger.error(f"NaN loss detected in logs at step {state.global_step}!")
                logger.error(f"{'!'*80}\n")
                control.should_training_stop = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if model is not None and state.global_step - self.last_check_step >= self.check_interval:
            self.last_check_step = state.global_step

            # Check parameters for NaN
            nan_params = []
            inf_params = []
            param_stats = {}

            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
                if torch.isinf(param).any():
                    inf_params.append(name)

                # Collect statistics for important layers
                if 'classifier' in name or 'score' in name or 'output' in name:
                    param_stats[name] = {
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'min': param.data.min().item(),
                        'max': param.data.max().item(),
                    }

            if nan_params or inf_params:
                logger.error(f"\n{'!'*80}")
                logger.error(f"NaN/Inf PARAMETERS DETECTED at step {state.global_step}!")
                if nan_params:
                    logger.error(f"NaN parameters: {nan_params}")
                if inf_params:
                    logger.error(f"Inf parameters: {inf_params}")
                logger.error(f"{'!'*80}\n")
                control.should_training_stop = True
            else:
                logger.info(f"[Step {state.global_step}] Parameter check: All parameters valid (no NaN/Inf)")

            # Log parameter statistics periodically
            if state.global_step % (self.check_interval * 4) == 0 and param_stats:
                for name, stats in param_stats.items():
                    logger.info(f"[Step {state.global_step}] {name}: "
                              f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                              f"min={stats['min']:.6f}, max={stats['max']:.6f}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info("Training completed successfully without NaN issues!")


def train():
    # Disable wandb by default to avoid initialization issues
    # Can be overridden with --report_to wandb command line argument
    if "WANDB_DISABLED" not in os.environ and "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("wandb disabled by default. Use --report_to wandb to enable.")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        _
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    assert model_args.model_name_or_path.split('/')[-1] in training_args.output_dir
     
    # Enable wandb if explicitly requested
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        if os.environ.get("WANDB_DISABLED") == "true":
            os.environ.pop("WANDB_DISABLED")
        os.environ["WANDB_PROJECT"] = "PRM_Math_Shepherd"
        logger.info("wandb enabled via --report_to wandb")

    # Load model and tokenizer
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_cache = False,
        low_cpu_mem_usage=True,
    )

    # freeze llm except last layer if needed
    if training_args.fix_llm:
        model.model.requires_grad_(False)
        
    # Force the LLM to keep the gradient graph alive
    model.enable_input_require_grads() 
    # If you are using gradient checkpointing
    model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=False,
        trust_remote_code=True,
    )

    # Load data
    data_module = make_supervised_data_module(data_args=data_args)


    # Start trainer with NaN monitoring callback
    callbacks = []
    if training_args.enable_nan_monitoring:
        callbacks.append(NaNMonitorCallback(check_interval=training_args.nan_check_interval))
        logger.info(f"NaN monitoring enabled with check interval: {training_args.nan_check_interval}")

    trainer = PRMTrainerWithContrastive(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        # compute_loss_func=contrastive_loss_fn,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )

    try:
        dist.destroy_process_group()
    
    except Exception as e:
        logger.error(f"Error destroying process group: {e}")


if __name__ == "__main__":
    train()

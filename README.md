# AgentArk


## Table of Contents

- [Academic Abstract](#academic-abstract)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Inference](#inference)
  - [Solution Labeling](#solution-labeling)
  - [Process Reward Model Training](#process-reward-model-training)
  - [RL Finetuning with GRPO](#rl-finetuning-with-grpo)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)

## Academic Abstract

While large language model (LLM) multi-agent systems achieve superior reasoning performance through iterative debate, practical deployment is limited by their high computational cost and error propagation. In this paper, we propose AgentArk, a framework to distill these collective dynamics into the weights of a single model, effectively transforming explicit test-time interactions into implicit model capabilities. This equips a single agent with the intelligence of multi-agent systems while remaining computationally efficient. Specifically, we investigate three hierarchical distillation strategies across various models, tasks, scaling, and scenarios: reasoning-enhanced fine-tuning; trajectory-based augmentation; and process-aware distillation. By shifting the burden of computation from inference to training, the distilled models preserve the efficiency of one agent while exhibiting strong reasoning and self-correction performance of multiple agents. They further demonstrate enhanced robustness and generalizability across diverse reasoning tasks. We hope this work can shed light on future research on efficient and robust multi-agent development.

## Installation

### Requirements

- Python 3.10+
- CUDA 12.5
- 40GB+ GPU memory recommended for inference

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgentArk

# Create virtual environment
conda create -n agentark python=3.12
conda activate agentark

# Install dependencies
pip install -r requirements.txt

```

### Key Dependencies

| Category | Packages |
|----------|----------|
| LLM Inference | `transformers`, `vllm`, `flash-attn` |
| RL Training | `deepspeed`, `trl`, `torch` |
| Evaluation | `rouge_score`, `bert_score`, `sympy` |
| Utilities | `datasets`, `accelerate`, `peft`, `wandb` |

## Quick Start

```bash
# Run inference with LLM Debate on QMSum dataset
python inference.py \
    --method_name llm_debate \
    --test_dataset_name QMSum \
    --model_name Qwen/Qwen3-8B \
    --use_vllm \
    --tensor_parallel_size 2

# Evaluate results
python -m eval.short_answer_eval \
    --input_file results/QMSum/Qwen/Qwen3-8B/llm_debate_infer.jsonl \
    --dataset_name QMSum
```



## Usage

### Inference

Run multi-agent inference on a dataset:

```bash
python inference.py \
    --method_name <method> \
    --test_dataset_name <dataset> \
    --model_name <model_path_or_name> \
    --use_vllm \
    --tensor_parallel_size <num_gpus>
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--method_name` | Multi-agent method to use | Required |
| `--test_dataset_name` | Dataset for evaluation | Required |
| `--model_name` | HuggingFace model or local path | Required |
| `--model_temperature` | Sampling temperature | 0.5 |
| `--model_max_tokens` | Maximum tokens per generation | 4096 |
| `--use_vllm` | Enable vLLM for efficient batching | False |
| `--tensor_parallel_size` | Number of GPUs for tensor parallelism | 1 |
| `--use_modal_batch` | Use Modal for cloud deployment | False |

**Example - Running DyLAN on MATH:**

```bash
python inference.py \
    --method_name dylan \
    --test_dataset_name MATH \
    --model_name Qwen/Qwen3-32B \
    --use_vllm \
    --tensor_parallel_size 4 \
    --model_temperature 0.7
```

**Example - Running with Modal Cloud:**

```bash
# First deploy the Modal model
modal deploy modal/launch_modal.py

# Then run inference
python inference.py \
    --method_name llm_debate \
    --test_dataset_name QMSum \
    --use_modal_batch \
    --model_name Qwen/Qwen3-8B
```

### Solution Labeling

Label generated solutions for correctness (required for PRM training):

```bash
python label.py \
    --input results/QMSum/Qwen/Qwen3-32B/llm_debate_infer.jsonl \
    --dataset_name QMSum \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor_parallel_size 4
```

This produces labeled data with the format:
```json
{
    "query": "...",
    "gt": "ground truth answer",
    "solutions": [
        {"id": 1, "text": "solution text", "is_correct": true},
        {"id": 2, "text": "solution text", "is_correct": false}
    ],
    "labels": [true, false]
}
```

### Process Reward Model Training

Train a PRM to score intermediate reasoning steps:

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python prm/finetune2.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --train_data_path results/QMSum/labeled.jsonl \
    --output_dir outputs/prm_qmsum \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 True \
    --gradient_checkpointing True \
    --fix_llm True \
    --enable_nan_monitoring True
```

### RL Finetuning with GRPO

Finetune the policy model using Group Relative Policy Optimization:

```bash
python -m openrlhf.cli.train_grpo \
    --pretrain Qwen/Qwen3-0.6B \
    --reward_pretrain outputs/prm_qmsum \
    --save_path outputs/grpo_qmsum \
    --temperature 0.5 \
    --n_samples_per_prompt 8 \
    --advantage_estimator rloo \
    --reward_baseline token \
    --reward_mode PRMVR \
    --verifiable_reward_coef 1.0 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 64 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.001 \
    --max_epochs 1 \
    --num_episodes 1 \
    --prompt_max_len 40960 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 20 \
    --logging_steps 1
```

**GRPO Key Arguments:**

| Argument | Description |
|----------|-------------|
| `--pretrain` | Base model to finetune |
| `--reward_pretrain` | Trained PRM checkpoint |
| `--n_samples_per_prompt` | Group size for RLOO baseline (keep >= 4) |
| `--advantage_estimator` | `rloo` or `gae` |
| `--reward_mode` | Reward computation mode (`PRMVR`, `ORM`, etc.) |
| `--micro_rollout_batch_size` | Prompts per GPU during rollout |
| `--micro_train_batch_size` | Samples per GPU during training |

**Memory Optimization Tips:**
- Lower `micro_rollout_batch_size` and `micro_train_batch_size` to save GPU memory
- Keep `n_samples_per_prompt >= 4` for stable GRPO performance
- Total samples = `rollout_batch_size` x `n_samples_per_prompt`

### Evaluation

#### Short-Answer Evaluation (ROUGE, BERTScore, F1)

```bash
python -m eval.short_answer_eval \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_name QMSum \
    --split validation \
    --output_dir outputs \
    --temperature 0.7 \
    --use_vllm \
    --apply_chat_template
```

#### Batch Evaluation Across Models

```bash
for MODEL in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B; do
    for DATASET in QMSum QASPER HotpotQA; do
        python -m eval.short_answer_eval \
            --model_name_or_path "$MODEL" \
            --dataset_name "$DATASET" \
            --split validation \
            --output_dir outputs \
            --use_vllm
    done
done
```

#### Math Evaluation (Exact Match)

```bash
python -m eval.math_eval \
    --input_file results/MATH/Qwen/Qwen3-8B/mav_infer.jsonl \
    --dataset_name MATH
```

## Configuration

Each method has YAML configuration files in `methods/<method_name>/configs/`.

### Example: DyLAN Configuration

```yaml
# methods/dylan/configs/config_main.yaml
random_seed: 0
num_agents: 4           # Number of agents in the network
num_rounds: 3           # Communication rounds
activation: "listwise"  # Agent ranking strategy
roles:
    - "Assistant"
    - "Assistant"
    - "Assistant"
    - "Assistant"
```

### Example: AgentVerse Configuration

```yaml
# methods/agentverse/configs/config_main.yaml
cnt_agents: 2               # Number of collaborative agents
max_turn: 3                 # Maximum conversation turns
max_criticizing_rounds: 3   # Critic feedback iterations
```

### Example: Self-Consistency Configuration

```yaml
# methods/self_consistency/configs/config_main.yaml
parallel_num: 5  # Number of parallel solution paths
```

### Example: LLM Debate Configuration

```yaml
# methods/llm_debate/configs/config_main.yaml
num_agents: 3           # Number of debating agents
num_rounds: 2           # Debate rounds
```

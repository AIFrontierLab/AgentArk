import argparse
import os
import json
import re
import time
import torch
from datasets import load_dataset
from model_utils import generate_completions, load_hf_lm_and_tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from math_eval import parse_args as parse_args_math
from eval_utils import set_seed, save_jsonl, load_jsonl
from misc_utils import print_colored

def check_nan_parameters(model):
    """Check for NaN values in model parameters and print them in red."""
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
            print_colored(f"NaN detected in parameter: {name}", "red")

    if not nan_params:
        print_colored("No NaN values detected in model parameters.", "green")
    else:
        print_colored(f"\nTotal parameters with NaN: {len(nan_params)}", "red")

    return nan_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--subfolder", default=None, type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true")
    args = parser.parse_args()
    return args


def extract_answer(text):
    """Extract A/B/C/D from model output."""
    # Remove content between <think> and </think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)

    patterns = [
        # LaTeX boxed format: \boxed{A}
        r'\\boxed\{([ABCD])\}',
        # Markdown bold answer: **Answer**: C or **Answer** C or **Answer: C**
        r'\*\*\s*[Aa]nswer\s*\*\*\s*:?\s*([ABCD])\b',
        # Answer with colon: Answer: C or answer: C
        r'[Aa]nswer\s*:?\s*([ABCD])\b',
        # Bold letter: **C**
        r'\*\*([ABCD])\*\*',
        # Italic letter: *C*
        r'(?<!\*)\*([ABCD])\*(?!\*)',
        # Parentheses: (C)
        r'\(([ABCD])\)',
        # Standalone letter with word boundary
        r'\b([ABCD])\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: check if first character is A/B/C/D
    return text.strip()[:1].upper() if text and text[0] in "ABCD" else None


def main(args):
    print(f"Loading {args.dataset_name} split={args.split}")
    dataset = load_dataset(args.dataset_name, split=args.split)
    examples = list(dataset)[:args.num_test_sample if args.num_test_sample > 0 else len(dataset)]

    # Setup output
    os.makedirs(f"{args.output_dir}/medmcqa", exist_ok=True)
    out_file = f"{args.output_dir}/medmcqa/test_{args.model_name_or_path.split('/')[-1]}_n{args.num_test_sample}_t{args.temperature}.jsonl"

    # Load model
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if args.use_vllm:
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(gpus), trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) if args.apply_chat_template else None
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(args.model_name_or_path, subfolder=args.subfolder, load_in_half=True, use_safetensors=args.use_safetensors)
        # Check for NaN values in model parameters
        print("\nChecking for NaN values in model parameters...")
        check_nan_parameters(llm)
        
    prompts = []
    for ex in examples:
        prompt = f"""{ex["question"]}
A. {ex["opa"]}
B. {ex["opb"]}
C. {ex["opc"]}
D. {ex["opd"]}
Answer with a single capital letter in the last line of your final answer, such as A, B, C, or D."""
        prompts.append(prompt)


    if args.apply_chat_template:
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                   tokenize=False, add_generation_prompt=True) for p in prompts]

    # Generate
    print(f"Generating {len(prompts)} responses...")
    start_time = time.time()
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.use_vllm:
        outputs = llm.generate(prompts, SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens_per_call,
            stop=stop_words, stop_token_ids=[151645, 151643] if "qwen2" in args.model_name_or_path.lower() else None
        ))
        outputs = [o.outputs[0].text for o in sorted(outputs, key=lambda x: int(x.request_id))]
    else:
        outputs = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_tokens_per_call,
            batch_size=args.batch_size,
            stop_id_sequences=stop_words,
        )
        

    time_use = time.time() - start_time

    # Evaluate
    predictions = [extract_answer(o) for o in outputs]
    answers = ["ABCD"[ex["cop"]] for ex in examples]

    results = [{
        "idx": i, 
        "prompt": ex["question"], 
        "output": outputs[i],
        "pred": predictions[i], 
        "label": answers[i], 
        "correct": predictions[i] == answers[i],
    } for i, ex in enumerate(examples)]
    
    correct = sum(1 for i, result in enumerate(results) if result["correct"])

    metrics = {
        "num_samples": len(examples), 
        "num_correct": correct,
        "acc": round(correct / len(examples) * 100, 2),
        "time_seconds": time_use
    }

    save_jsonl(results, out_file)
    with open(out_file.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults: {json.dumps(metrics, indent=2)}")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)

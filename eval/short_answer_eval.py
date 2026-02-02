"""Evaluation script for short-answer datasets (QMSum, etc.) with ROUGE, BERTScore, and F1."""
import argparse
import os
import json
import re
import time
import logging
import torch
import string
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rouge_score import rouge_scorer
import bert_score
import pandas as pd
from collections import Counter
from tqdm.contrib import tzip

from .eval_utils import set_seed, save_jsonl, load_jsonl
from .model_utils import generate_completions, load_hf_lm_and_tokenizer

from utils.model_utils import build_dataset

# Suppress rouge_score INFO messages
logging.getLogger("rouge_scorer").setLevel(logging.ERROR)





def read_jsonl(f):
    return [json.loads(x) for x in open(f).readlines()]

def write_jsonl(data: list, f):
    with open(f, 'a') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--subfolder", default=None, type=str)
    parser.add_argument("--setting", default=None, type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max_tokens_per_call", default=512, type=int)
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--max_model_len", default=None, type=int, help="Maximum model context length for vLLM")
    parser.add_argument("--max_prompt_tokens", default=None, type=int,
                       help="Maximum tokens for prompt transcript")

    args = parser.parse_args()
    if args.max_prompt_tokens and args.max_model_len is None:
        assert args.max_model_len > args.max_prompt_tokens
    return args


def normalize_answer(s):
    """Normalize answer for F1 calculation (based on calculate_metrics.py)."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)

    def lower(text):
        return text.lower()

    def remove_stop_words(text):
        # Common English stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will',
            'with'
        }
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    return white_space_fix(remove_stop_words(remove_punc(lower(remove_articles(s)))))


def calculate_f1(prediction, ground_truth):
    """Calculate token-based F1 score (based on evaluate_f1_em_qa_automatic from calculate_metrics.py)."""
    if not prediction or not ground_truth:
        return 0.0

    # Handle lists of ground truths
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    f1_scores = []
    for gt in ground_truth:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(gt).split()

        if not prediction_tokens or not ground_truth_tokens:
            f1_scores.append(0.0)
            continue

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1_score = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1_score)

    return max(f1_scores) if f1_scores else 0.0


def calculate_rouge(prediction, ground_truth):
    """Calculate ROUGE scores using rouge-score package."""
    if not prediction or not ground_truth:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Handle lists of ground truths
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE for each reference and take max
    all_scores = []
    for gt in ground_truth:
        scores = scorer.score(gt, prediction)
        all_scores.append({
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        })

    # Return max F1 scores across all references
    if not all_scores:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    return {
        "rouge1": max(s["rouge1"] for s in all_scores),
        "rouge2": max(s["rouge2"] for s in all_scores),
        "rougeL": max(s["rougeL"] for s in all_scores)
    }


def calculate_bertscore_batch(predictions, ground_truths, device=None):
    """Calculate BERTScore in batch for efficiency."""
    if not predictions or not ground_truths:
        return []

    # Prepare references (handle lists of ground truths)
    refs = []
    for gt in ground_truths:
        if isinstance(gt, list):
            refs.append(gt[0] if gt else "")  # Use first reference for BERTScore
        else:
            refs.append(str(gt))

    # Calculate BERTScore
    P, R, F1 = bert_score.score(predictions, refs, lang="en", rescale_with_baseline=False, verbose=False, batch_size=1024, device=device)

    return F1.tolist()


def get_dataset_fields(dataset_name):
    """Get dataset-specific field names for query and answer."""
    # Map dataset names to their field names
    if "QMSum" in dataset_name:
        return "query", "gt"
    else:
        # Default fields (can be extended for other datasets)
        return "query", "answer"


def main(args):
    # Get dataset-specific field names

    

    # Extract dataset shortname for output path
    dataset_shortname = args.dataset_name.split("/")[-1] if "/" in args.dataset_name else args.dataset_name

    # Setup output
    os.makedirs(f"{args.output_dir}/{dataset_shortname}", exist_ok=True)
    
    if args.setting == "generalization":
        output_file = os.path.join(args.output_dir, f"max_length_{args.max_prompt_tokens}", args.setting, dataset_shortname, f"{args.model_name_or_path.split('/')[-1]}-{args.split}_t{args.temperature}.jsonl")
        metric_file = os.path.join(args.output_dir, f"max_length_{args.max_prompt_tokens}", args.setting, dataset_shortname, f"metrics_{args.model_name_or_path.split('/')[-1]}-{args.split}_t{args.temperature}.csv")
    
    else:
        output_file = os.path.join(args.output_dir, f"max_length_{args.max_prompt_tokens}", dataset_shortname, f"{args.model_name_or_path.split('/')[-1]}-{args.split}_t{args.temperature}.jsonl")
        metric_file = os.path.join(args.output_dir, f"max_length_{args.max_prompt_tokens}", dataset_shortname, f"metrics_{args.model_name_or_path.split('/')[-1]}-{args.split}_t{args.temperature}.csv")
    
    # if os.path.exists(metric_file) and not args.save_outputs:
    #     print(f"Metrics file {metric_file} already exists. Skipping evaluation.")
    #     return
    

    # Load tokenizer early for prompt truncation
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    print(f"Loading {args.dataset_name} split={args.split}")

    # Prepare prompts
    prompts = []

    dataset = build_dataset(args,
                            # tokenizer=tokenizer # uncomment to truncate the prompts
                            )
    for entry in tqdm(dataset, desc="Building prompts"):
        prompt = entry["query"]
        prompts.append(prompt)
        
    if "outputs" in args.model_name_or_path:
        assert args.dataset_name in args.model_name_or_path, "Dataset name must be in model name or path."
    
    if False: # os.path.exists(output_file):
        outputs = read_jsonl(output_file)
        
    else:

        # Load model
        if args.use_vllm:
            llm_kwargs = {
                "model": args.model_name_or_path,
                "tensor_parallel_size": args.tensor_parallel_size,
                "trust_remote_code": True
            }
            if args.max_model_len is not None:
                llm_kwargs["max_model_len"] = args.max_model_len
            llm = LLM(**llm_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            llm, tokenizer = load_hf_lm_and_tokenizer(
                args.model_name_or_path,
                subfolder=args.subfolder,
                load_in_half=True,
                use_safetensors=args.use_safetensors
            )

        max_length = 39936 if args.max_prompt_tokens is None else min(args.max_prompt_tokens, 39936)
        if "llama" in args.model_name_or_path.lower():
            max_length = 7168 if args.max_prompt_tokens is None else min(args.max_prompt_tokens, 7168)
            
        truncated_prompts = tokenizer.batch_encode_plus(prompts, truncation=True, max_length=max_length)
        prompts = tokenizer.batch_decode(truncated_prompts.input_ids)
        
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                tokenize=False, 
                add_generation_prompt=True, 
                truncation=True)for p in prompts]
        

        # Generate
        print(f"Generating {len(prompts)} responses...")
        start_time = time.time()
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

        if args.use_vllm:
            outputs = llm.generate(prompts, SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens_per_call,
                stop=stop_words,
                stop_token_ids=[151645, 151643] if "qwen2" in args.model_name_or_path.lower() else None
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
        
        
        save_jsonl(outputs, output_file)
        
    
    ground_truths = [ex["gt"] for ex in dataset]
    assert len(outputs) == len(ground_truths), "Number of outputs and ground truths must match."
    # Calculate BERTScore in batch
    print("Calculating BERTScore...")
    outputs = [out.split("</think>")[-1].strip() for out in outputs]
    bertscore_f1s = calculate_bertscore_batch(outputs, ground_truths, device="cpu")
    
    F1 = np.mean(bertscore_f1s)

    # Calculate metrics for each sample
    print("Calculating ROUGE and F1 scores...")
    results = []
    rouge1_scores, rouge2_scores, rougeL_scores, f1_scores = [], [], [], []

    for i, ex in enumerate(tzip(outputs, ground_truths)):
        output = outputs[i]
        gt = str(ground_truths[i])

        # Calculate ROUGE
        rouge_scores = calculate_rouge(output, gt)
        rouge1_scores.append(rouge_scores["rouge1"])
        rouge2_scores.append(rouge_scores["rouge2"])
        rougeL_scores.append(rouge_scores["rougeL"])

        # Calculate F1
        f1 = calculate_f1(output, gt)
        f1_scores.append(f1)

        result = {
            "idx": i,
            "output": output,
            "gt": gt,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bertscore_f1": bertscore_f1s[i],
            "f1_score": f1,
        }
        results.append(result)


    assert len(results) == len(outputs)
    # Aggregate metrics
    metrics = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_name_or_path,
        "num_samples": len(outputs),
        "rouge1_f": round(sum(rouge1_scores) / len(rouge1_scores) * 100, 2),
        "rouge2_f": round(sum(rouge2_scores) / len(rouge2_scores) * 100, 2),
        "rougeL_f": round(sum(rougeL_scores) / len(rougeL_scores) * 100, 2),
        "bertscore_f1_avg": round(sum(bertscore_f1s) / len(bertscore_f1s) * 100, 2),
        "f1_avg": round(sum(f1_scores) / len(f1_scores) * 100, 2),
    }

    metrics = pd.Series(metrics)
    print(metrics)
    metrics.to_csv(metric_file, index=True)
    print(f"Metrics saved to: {metric_file}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)

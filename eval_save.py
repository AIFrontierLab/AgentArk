# ### GSM8K

# import os
# import json
# import argparse
# import threading
# import concurrent.futures
# from tqdm import tqdm
# import re

# from methods import get_method_class
# from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl, read_valid_jsonl


# # Moved your numeric extractor + custom eval_func here
# def extract_final_answer(text: str):
#     """Extract final numeric answer from reasoning text."""
#     if text is None:
#         return None
#     if "####" in text:
#         match = re.search(r"####\s*([-+]?[0-9]*\.?[0-9]+)", text)
#         if match:
#             return match.group(1)
#     if "\\boxed" in text:
#         match = re.search(r"\\boxed\{([-+]?[0-9]*\.?[0-9]+)\}", text)
#         if match:
#             return match.group(1)
#     matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
#     if matches:
#         return matches[-1]
#     return None


# def custom_eval_func(item, llm=None):
#     gt_ans = extract_final_answer(item.get("gt", ""))
#     pred_ans = extract_final_answer(item.get("response", ""))
#     if gt_ans is None or pred_ans is None:
#         return "Missing answer", None
#     correct = gt_ans == pred_ans
#     return f"GT: {gt_ans}, Pred: {pred_ans}", int(correct)


# def evaluate_sample(args, item, save_eval_path, lock=None, llm=None):
#     # Use our custom evaluator instead of get_eval_func(...)
#     eval_content, eval_score = custom_eval_func(item, llm) if 'response' in item else ("Infer Error", None)

#     save_data = item.copy()
#     save_data["eval_content"] = eval_content
#     save_data["eval_score"] = eval_score

#     # Ensure directory exists before writing
#     dir_path = os.path.dirname(save_eval_path)
#     if dir_path:
#         os.makedirs(dir_path, exist_ok=True)

#     # Save all samples (or only correct ones if you prefer)
#     write_to_jsonl(lock, save_eval_path, save_data)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_protocol", type=str, default="xverify")
#     parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
#     parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
#     parser.add_argument("--model_temperature", type=float, default=0.5)
#     parser.add_argument("--model_max_tokens", type=int, default=2048)
#     parser.add_argument("--model_timeout", type=int, default=600)
#     parser.add_argument("--tested_dataset_name", type=str, default="MATH")
#     parser.add_argument("--tested_method_name", type=str, default="MAS-GPT")
#     parser.add_argument("--tested_method_config_name", type=str, default=None)
#     parser.add_argument("--tested_mas_model_name", type=str, default="Qwen/Qwen3-8B")
#     parser.add_argument("--tested_infer_path", type=str, default=None)
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--sequential", action="store_true")
#     args = parser.parse_args()

#     print("="*50)
#     print(f"Evaluating {args.tested_method_name} on {args.tested_dataset_name}")

#     # Load model
#     model_api_config = load_model_api_config(args.model_api_config, args.model_name)
#     LLM_METHOD = get_method_class("vanilla")
#     llm = LLM_METHOD(vars(args))

#     # Paths
#     tested_infer_path = args.tested_infer_path or f"./results/{args.tested_dataset_name}/{args.tested_mas_model_name}/{args.tested_method_name}_infer.jsonl"
#     if args.tested_method_config_name and args.tested_infer_path is None:
#         tested_infer_path = tested_infer_path.replace("_infer.jsonl", f"_{args.tested_method_config_name}_infer.jsonl")
#     save_eval_path = tested_infer_path.replace("infer", "xverify_eval")

#     if args.debug:
#         sample = {"query": "1+3=?", "gt": "#### 4", "response": "\\boxed{4}"}
#         evaluate_sample(args, sample, save_eval_path, lock=None, llm=llm)
#         exit()

#     # Load eval data
#     eval_data = read_valid_jsonl(tested_infer_path)
#     print(f">> {len(eval_data)} samples before filtering")

#     if args.overwrite and os.path.exists(save_eval_path):
#         os.remove(save_eval_path)
#         print(f">> Overwriting {save_eval_path}")

#     eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
#     print(f">> {len(eval_data)} samples after filtering")

#     lock = threading.Lock()
#     max_workers = model_api_config[args.model_name]["max_workers"]
#     if args.sequential:
#         for sample in eval_data:
#             evaluate_sample(args, sample, save_eval_path, lock, llm)
#     else:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             list(tqdm(executor.map(lambda s: evaluate_sample(args, s, save_eval_path, lock, llm), eval_data), total=len(eval_data)))

#     # Ensure file exists even if no sample passes
#     if not os.path.exists(save_eval_path):
#         os.makedirs(os.path.dirname(save_eval_path), exist_ok=True)
#         open(save_eval_path, "w").close()

#     # Compute stats
#     with open(save_eval_path, "r") as f:
#         saved_data = [json.loads(line) for line in f if line.strip()]

#     sample_num = len(saved_data)
#     valid_scores = [s["eval_score"] for s in saved_data if s["eval_score"] is not None]
#     correct = sum(1 for s in valid_scores if s == 1)
#     valid = len(valid_scores)
#     non_err = len([s for s in saved_data if not str(s.get("eval_content", "")).startswith("Eval Error")])
#     if valid > 0:
#         print(f">> Evaluation Finished:\n{sample_num} samples total\n{valid} valid | {correct} correct | acc: {correct/valid*100:.2f}%\n{non_err} excl. eval error | acc: {correct/non_err*100:.2f}%")
#     else:
#         print(">> No valid samples evaluated.")


# ### MATH

# import os
# import json
# import argparse
# import threading
# import concurrent.futures
# from tqdm import tqdm
# import re

# from methods import get_method_class
# from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl, read_valid_jsonl



# def extract_final_answer(text: str):
#     """Extract the final numeric answer (int or float) from reasoning text."""
#     if not text:
#         return None

#     # Try to extract LaTeX boxed answers
#     match = re.search(r'\\boxed\{([-+]?\d*\.?\d+)\}', text)
#     if match:
#         return match.group(1)

#     # Try to extract answers following 'Answer:', 'Final Answer:', etc.
#     match = re.search(r'(?:Final\s+Answer|Answer)\s*[:\-]?\s*\$?([-+]?\d*\.?\d+)\$?', text, re.IGNORECASE)
#     if match:
#         return match.group(1)

#     # Try to extract answers after '####', '###', etc.
#     match = re.search(r'#+\s*\$?([-+]?\d*\.?\d+)\$?', text)
#     if match:
#         return match.group(1)

#     # Try to extract the last standalone number in the text
#     matches = re.findall(r'[-+]?\d*\.?\d+', text)
#     if matches:
#         return matches[-1]

#     return None



# # def custom_eval_func(item, llm=None):
# #     gt_ans = extract_final_answer(item.get("gt", ""))
# #     pred_ans = extract_final_answer(item.get("response", ""))
# #     if gt_ans is None or pred_ans is None:
# #         return "Missing answer", None
# #     correct = gt_ans == pred_ans
# #     return f"GT: {gt_ans}, Pred: {pred_ans}", int(correct)

# import sympy as sp

# def custom_eval_func(item, llm=None):
#     gt_ans_str = extract_final_answer(item.get("gt", ""))
#     pred_ans_str = extract_final_answer(item.get("response", ""))

#     if gt_ans_str is None or pred_ans_str is None:
#         return "Missing answer", None

#     # Try to parse both as SymPy numbers (support fractions, decimals, negative numbers)
#     try:
#         gt_val = sp.sympify(gt_ans_str)
#         pred_val = sp.sympify(pred_ans_str)
#         correct = sp.simplify(gt_val - pred_val) == 0
#     except Exception:
#         # Fallback to strict string match if parsing fails
#         correct = gt_ans_str.strip() == pred_ans_str.strip()

#     return f"GT: {gt_ans_str}, Pred: {pred_ans_str}", int(correct)




# def evaluate_sample(args, item, save_eval_path, lock=None, llm=None):
#     # Use our custom evaluator instead of get_eval_func(...)
#     eval_content, eval_score = custom_eval_func(item, llm) if 'response' in item else ("Infer Error", None)

#     save_data = item.copy()
#     save_data["eval_content"] = eval_content
#     save_data["eval_score"] = eval_score

#     # Ensure directory exists before writing
#     dir_path = os.path.dirname(save_eval_path)
#     if dir_path:
#         os.makedirs(dir_path, exist_ok=True)

#     # Save all samples (or only correct ones if you prefer)
#     write_to_jsonl(lock, save_eval_path, save_data)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_protocol", type=str, default="xverify")
#     parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
#     parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
#     parser.add_argument("--model_temperature", type=float, default=0.5)
#     parser.add_argument("--model_max_tokens", type=int, default=2048)
#     parser.add_argument("--model_timeout", type=int, default=600)
#     parser.add_argument("--tested_dataset_name", type=str, default="MATH")
#     parser.add_argument("--tested_method_name", type=str, default="MAS-GPT")
#     parser.add_argument("--tested_method_config_name", type=str, default=None)
#     parser.add_argument("--tested_mas_model_name", type=str, default="Qwen/Qwen3-8B")
#     parser.add_argument("--tested_infer_path", type=str, default=None)
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--sequential", action="store_true")
#     args = parser.parse_args()

#     print("="*50)
#     print(f"Evaluating {args.tested_method_name} on {args.tested_dataset_name}")

#     # Load model
#     model_api_config = load_model_api_config(args.model_api_config, args.model_name)
#     LLM_METHOD = get_method_class("vanilla")
#     llm = LLM_METHOD(vars(args))

#     # Paths
#     tested_infer_path = args.tested_infer_path or f"./results/{args.tested_dataset_name}/{args.tested_mas_model_name}/{args.tested_method_name}_infer.jsonl"
#     if args.tested_method_config_name and args.tested_infer_path is None:
#         tested_infer_path = tested_infer_path.replace("_infer.jsonl", f"_{args.tested_method_config_name}_infer.jsonl")
#     save_eval_path = tested_infer_path.replace("infer", "xverify_eval")

#     if args.debug:
#         sample = {"query": "1+3=?", "gt": "#### 4", "response": "\\boxed{4}"}
#         evaluate_sample(args, sample, save_eval_path, lock=None, llm=llm)
#         exit()

#     # Load eval data
#     eval_data = read_valid_jsonl(tested_infer_path)
#     print(f">> {len(eval_data)} samples before filtering")

#     if args.overwrite and os.path.exists(save_eval_path):
#         os.remove(save_eval_path)
#         print(f">> Overwriting {save_eval_path}")

#     eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
#     print(f">> {len(eval_data)} samples after filtering")

#     lock = threading.Lock()
#     max_workers = model_api_config[args.model_name]["max_workers"]
#     if args.sequential:
#         for sample in eval_data:
#             evaluate_sample(args, sample, save_eval_path, lock, llm)
#     else:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             list(tqdm(executor.map(lambda s: evaluate_sample(args, s, save_eval_path, lock, llm), eval_data), total=len(eval_data)))

#     # Ensure file exists even if no sample passes
#     if not os.path.exists(save_eval_path):
#         os.makedirs(os.path.dirname(save_eval_path), exist_ok=True)
#         open(save_eval_path, "w").close()

#     # Compute stats
#     with open(save_eval_path, "r") as f:
#         saved_data = [json.loads(line) for line in f if line.strip()]

#     sample_num = len(saved_data)
#     valid_scores = [s["eval_score"] for s in saved_data if s["eval_score"] is not None]
#     correct = sum(1 for s in valid_scores if s == 1)
#     valid = len(valid_scores)
#     non_err = len([s for s in saved_data if not str(s.get("eval_content", "")).startswith("Eval Error")])
#     if valid > 0:
#         print(f">> Evaluation Finished:\n{sample_num} samples total\n{valid} valid | {correct} correct | acc: {correct/valid*100:.2f}%\n{non_err} excl. eval error | acc: {correct/non_err*100:.2f}%")
#     else:
#         print(">> No valid samples evaluated.")



### MedMCQA

### MULTIPLE-CHOICE

import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import re

from methods import get_method_class
from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl, read_valid_jsonl


def extract_final_choice(text: str):
    """Extract the final choice letter (A, B, C, D, etc.) from reasoning text."""
    if not text:
        return None

    # Extract LaTeX boxed letters like \boxed{A}
    match = re.search(r'\\boxed\{([A-Da-d])\}', text)
    if match:
        return match.group(1).upper()

    # Extract 'Answer: A' or 'Final Answer: (B)' style
    match = re.search(r'(?:Final\s+Answer|Answer)\s*[:\-]?\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Extract the last standalone capital letter choice
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()

    return None


def extract_gt_choice(text: str):
    if not text:
        return None

    match = re.search(r'\(([A-Da-d])\)', text)
    if match:
        return match.group(1).upper()

    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1).upper()

    return None


def custom_eval_func(item, llm=None):
    gt_ans = extract_gt_choice(item.get("gt", ""))
    pred_ans = extract_final_choice(item.get("response", ""))
    if gt_ans is None or pred_ans is None:
        return f"Missing answer (GT={gt_ans}, Pred={pred_ans})", None
    correct = gt_ans == pred_ans
    return f"GT: {gt_ans}, Pred: {pred_ans}", int(correct)


def evaluate_sample(args, item, save_eval_path, lock=None, llm=None):
    eval_content, eval_score = custom_eval_func(item, llm) if 'response' in item else ("Infer Error", None)

    save_data = item.copy()
    save_data["eval_content"] = eval_content
    save_data["eval_score"] = eval_score

    dir_path = os.path.dirname(save_eval_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    write_to_jsonl(lock, save_eval_path, save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_protocol", type=str, default="xverify")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5)
    parser.add_argument("--model_max_tokens", type=int, default=2048)
    parser.add_argument("--model_timeout", type=int, default=600)
    parser.add_argument("--tested_dataset_name", type=str, default="MedMCQA")
    parser.add_argument("--tested_method_name", type=str, default="MAS-GPT")
    parser.add_argument("--tested_method_config_name", type=str, default=None)
    parser.add_argument("--tested_mas_model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--tested_infer_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print(f"Evaluating {args.tested_method_name} on {args.tested_dataset_name}")

    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    LLM_METHOD = get_method_class("vanilla")
    llm = LLM_METHOD(vars(args))

    tested_infer_path = args.tested_infer_path or f"./results/{args.tested_dataset_name}/{args.tested_mas_model_name}/{args.tested_method_name}_infer.jsonl"
    if args.tested_method_config_name and args.tested_infer_path is None:
        tested_infer_path = tested_infer_path.replace("_infer.jsonl", f"_{args.tested_method_config_name}_infer.jsonl")
    save_eval_path = tested_infer_path.replace("infer", "xverify_eval")

    if args.debug:
        sample = {
            "query": "1st dental visit of the child should be at the age of:\n(A) 6 months\n(B) 1 year\n(C) 2 years\n(D) 3 months",
            "gt": "The correct answer is: (B) 1 year",
            "response": "The answer is \\boxed{A}"
        }
        evaluate_sample(args, sample, save_eval_path, lock=None, llm=llm)
        exit()

    eval_data = read_valid_jsonl(tested_infer_path)
    print(f">> {len(eval_data)} samples before filtering")

    if args.overwrite and os.path.exists(save_eval_path):
        os.remove(save_eval_path)
        print(f">> Overwriting {save_eval_path}")

    eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
    print(f">> {len(eval_data)} samples after filtering")

    lock = threading.Lock()
    max_workers = model_api_config[args.model_name]["max_workers"]
    if args.sequential:
        for sample in eval_data:
            evaluate_sample(args, sample, save_eval_path, lock, llm)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(lambda s: evaluate_sample(args, s, save_eval_path, lock, llm), eval_data), total=len(eval_data)))

    if not os.path.exists(save_eval_path):
        os.makedirs(os.path.dirname(save_eval_path), exist_ok=True)
        open(save_eval_path, "w").close()

    with open(save_eval_path, "r") as f:
        saved_data = [json.loads(line) for line in f if line.strip()]

    sample_num = len(saved_data)
    valid_scores = [s["eval_score"] for s in saved_data if s["eval_score"] is not None]
    correct = sum(1 for s in valid_scores if s == 1)
    valid = len(valid_scores)
    non_err = len([s for s in saved_data if not str(s.get("eval_content", "")).startswith("Eval Error")])

    if valid > 0:
        print(f">> Evaluation Finished:\n{sample_num} samples total\n{valid} valid | {correct} correct | acc: {correct/valid*100:.2f}%\n{non_err} excl. eval error | acc: {correct/non_err*100:.2f}%")
    else:
        print(">> No valid samples evaluated.")














# import os
# import json
# import argparse
# import threading
# import concurrent.futures
# from tqdm import tqdm
# import re

# from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl, read_valid_jsonl
# from methods import get_method_class


# import sympy as sp
# import re

# def extract_final_answer(text: str):
#     if not text:
#         return None

#     # Keep original regex rules
#     match = re.search(r'\\boxed\{([-+]?\d*\.?\d+)\}', text)
#     if match:
#         return match.group(1)

#     match = re.search(r'(?:Final\s+Answer|Answer)\s*[:\-]?\s*\$?([-+]?\d*\.?\d+)\$?', text, re.IGNORECASE)
#     if match:
#         return match.group(1)

#     match = re.search(r'#+\s*\$?([-+]?\d*\.?\d+)\$?', text)
#     if match:
#         return match.group(1)

#     matches = re.findall(r'[-+]?\d*\.?\d+', text)
#     if matches:
#         return matches[-1]

#     return None

# def custom_eval_func(item):
#     gt_ans_str = extract_final_answer(item.get("gt", ""))
#     pred_ans_str = extract_final_answer(item.get("response", ""))

#     if gt_ans_str is None or pred_ans_str is None:
#         return "Missing answer", None

#     try:
#         # Evaluate fractions or numeric strings safely
#         gt_val = sp.Rational(gt_ans_str)
#         pred_val = sp.Rational(pred_ans_str)
#     except:
#         # fallback: compare as strings
#         correct = gt_ans_str.strip() == pred_ans_str.strip()
#         return f"GT: {gt_ans_str}, Pred: {pred_ans_str}", int(correct)

#     correct = gt_val == pred_val
#     return f"GT: {gt_val}, Pred: {pred_val}", int(correct)


#     return None

# def safe_eval(expr: str):
#     """Evaluate a math expression safely using SymPy."""
#     try:
#         return sp.sympify(expr).evalf()
#     except Exception:
#         return None

# # import sympy as sp

# # def custom_eval_func(item, llm=None):
# #     gt_ans_str = extract_final_answer(item.get("gt", ""))
# #     pred_ans_str = extract_final_answer(item.get("response", ""))

# #     if gt_ans_str is None or pred_ans_str is None:
# #         return "Missing answer", None

# #     # Use sympy to evaluate numerically
# #     try:
# #         gt_val = sp.sympify(gt_ans_str).evalf()
# #         pred_val = sp.sympify(pred_ans_str).evalf()
# #     except:
# #         return f"GT: {gt_ans_str}, Pred: {pred_ans_str} (could not eval)", 0

# #     # Compare with tolerance
# #     correct = sp.Abs(gt_val - pred_val) < 1e-9
# #     return f"GT: {gt_val}, Pred: {pred_val}", int(correct)


# def evaluate_sample(args, item, save_eval_path, lock=None, llm=None):
#     eval_content, eval_score = custom_eval_func(item, llm) if 'response' in item else ("Infer Error", None)

#     save_data = item.copy()
#     save_data["eval_content"] = eval_content
#     save_data["eval_score"] = eval_score

#     dir_path = os.path.dirname(save_eval_path)
#     if dir_path:
#         os.makedirs(dir_path, exist_ok=True)

#     write_to_jsonl(lock, save_eval_path, save_data)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_protocol", type=str, default="xverify")
#     parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
#     parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
#     parser.add_argument("--model_temperature", type=float, default=0.5)
#     parser.add_argument("--model_max_tokens", type=int, default=2048)
#     parser.add_argument("--model_timeout", type=int, default=600)
#     parser.add_argument("--tested_dataset_name", type=str, default="MATH")
#     parser.add_argument("--tested_method_name", type=str, default="MAS-GPT")
#     parser.add_argument("--tested_method_config_name", type=str, default=None)
#     parser.add_argument("--tested_mas_model_name", type=str, default="Qwen/Qwen3-8B")
#     parser.add_argument("--tested_infer_path", type=str, default=None)
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--sequential", action="store_true")
#     args = parser.parse_args()

#     print("="*50)
#     print(f"Evaluating {args.tested_method_name} on {args.tested_dataset_name}")

#     # Load model
#     model_api_config = load_model_api_config(args.model_api_config, args.model_name)
#     LLM_METHOD = get_method_class("vanilla")
#     llm = LLM_METHOD(vars(args))

#     # Paths
#     tested_infer_path = args.tested_infer_path or f"./results/{args.tested_dataset_name}/{args.tested_mas_model_name}/{args.tested_method_name}_infer.jsonl"
#     if args.tested_method_config_name and args.tested_infer_path is None:
#         tested_infer_path = tested_infer_path.replace("_infer.jsonl", f"_{args.tested_method_config_name}_infer.jsonl")
#     save_eval_path = tested_infer_path.replace("infer", "xverify_eval")

#     if args.debug:
#         sample = {"query": "1+3=?", "gt": "#### 4", "response": "\\boxed{4}"}
#         evaluate_sample(args, sample, save_eval_path, lock=None, llm=llm)
#         exit()

#     # Load eval data
#     eval_data = read_valid_jsonl(tested_infer_path)
#     print(f">> {len(eval_data)} samples before filtering")

#     if args.overwrite and os.path.exists(save_eval_path):
#         os.remove(save_eval_path)
#         print(f">> Overwriting {save_eval_path}")

#     eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
#     print(f">> {len(eval_data)} samples after filtering")

#     lock = threading.Lock()
#     max_workers = model_api_config[args.model_name]["max_workers"]
#     if args.sequential:
#         for sample in eval_data:
#             evaluate_sample(args, sample, save_eval_path, lock, llm)
#     else:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             list(tqdm(executor.map(lambda s: evaluate_sample(args, s, save_eval_path, lock, llm), eval_data), total=len(eval_data)))

#     # Ensure file exists even if no sample passes
#     if not os.path.exists(save_eval_path):
#         os.makedirs(os.path.dirname(save_eval_path), exist_ok=True)
#         open(save_eval_path, "w").close()

#     # Compute stats
#     with open(save_eval_path, "r") as f:
#         saved_data = [json.loads(line) for line in f if line.strip()]

#     sample_num = len(saved_data)
#     valid_scores = [s["eval_score"] for s in saved_data if s["eval_score"] is not None]
#     correct = sum(1 for s in valid_scores if s == 1)
#     valid = len(valid_scores)
#     non_err = len([s for s in saved_data if not str(s.get("eval_content", "")).startswith("Eval Error")])
#     if valid > 0:
#         print(f">> Evaluation Finished:\n{sample_num} samples total\n{valid} valid | {correct} correct | acc: {correct/valid*100:.2f}%\n{non_err} excl. eval error | acc: {correct/non_err*100:.2f}%")
#     else:
#         print(">> No valid samples evaluated.")

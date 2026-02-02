import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback

from methods import get_method_class
from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl, read_valid_jsonl
from evaluations import get_eval_func

def evaluate_sample(args, item, save_eval_path, lock=None, llm=None):
    eval_func = get_eval_func(args.eval_protocol, args.tested_dataset_name)

    if 'response' in item:
        eval_content, eval_score = eval_func(item, llm)
    else:
        eval_content, eval_score = "Infer Error", None
        
    save_data = item.copy()
    save_data["eval_content"] = eval_content
    save_data["eval_score"] = eval_score
    if args.debug:
        print(json.dumps(save_data, indent=4))
    else:
        write_to_jsonl(lock, save_eval_path, save_data)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the evaluation
    parser.add_argument("--eval_protocol", type=str, default="xverify", help="The evaluation protocol to be used.")

    # args related to the model
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to evaluated objects
    parser.add_argument("--tested_dataset_name", type=str, default="MATH", help="The dataset to be used for testing.")
    parser.add_argument("--tested_method_name", type=str, default="MAS-GPT", help="MAS name.")
    parser.add_argument("--tested_method_config_name", type=str, default=None, help="The config name for the method.")
    parser.add_argument("--tested_mas_model_name", type=str, default="Qwen/Qwen3-8B", help="The agent backend to be used for inference.")
    parser.add_argument("--tested_infer_path", type=str, default=None, help="Path to the output file.")
    

    parser.add_argument("--debug", action="store_true", help="Turn this on to run one defined sample for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Turn this on to overwrite the existing output file.")
    parser.add_argument("--sequential", action="store_true", help="Turn this on to run the evaluation sequentially.")
    args = parser.parse_args()
    
    general_config = vars(args)

    print("="*50, f"\nEvaluating {args.tested_method_name} on {args.tested_dataset_name} with {args.tested_mas_model_name} as MAS model using {args.model_name} as LLM")
    print(json.dumps(general_config, indent=4))
    
    # Load model config
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})
    print("-"*50, f"\n>> Model API config: {model_api_config[args.model_name]}")
    
    LLM_METHOD = get_method_class("vanilla")
    llm = LLM_METHOD(general_config)
    
    # Load evaluation data
    tested_infer_path = args.tested_infer_path if args.tested_infer_path is not None else f"./results/{args.tested_dataset_name}/{args.tested_mas_model_name}/{args.tested_method_name}_infer.jsonl"
    tested_infer_path = tested_infer_path.replace("_infer.jsonl", f"_{args.tested_method_config_name}_infer.jsonl") if (args.tested_method_config_name is not None and args.tested_infer_path is None) else tested_infer_path
    save_eval_path = tested_infer_path.replace("infer", "xverify_eval")
    
    if args.debug:
        sample = {"query": "1+3=?", "gt": "4", "response": "\\boxed{4}"}
        evaluate_sample(args, sample, save_eval_path, lock=None, llm=llm)
    else:
        eval_data = read_valid_jsonl(tested_infer_path)
        print(f">> Before filtering: {len(eval_data)} samples")

        if args.overwrite and os.path.exists(save_eval_path):
            os.remove(save_eval_path)
            print(f">> {save_eval_path} exists, remove it.")
        else:
            eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
        print(f">> After filtering: {len(eval_data)} samples")

        max_workers = model_api_config[args.model_name]["max_workers"]
        lock = threading.Lock()
        if args.sequential:
            for sample in eval_data:
                evaluate_sample(args, sample, save_eval_path, lock, llm)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(lambda sample: evaluate_sample(args, sample, save_eval_path, lock, llm), eval_data), total=len(eval_data), desc="Evaluating MAS"):
                    pass
        
        # Load evaluation results and print the statistics
        with open(save_eval_path, "r") as f:
            saved_data = [json.loads(line) for line in f.readlines()]
    
        sample_num = len(saved_data)
        valid_eval_score_list = [sample["eval_score"] for sample in saved_data if sample["eval_score"] is not None]
        valid_correct_num = sum([1 for score in valid_eval_score_list if score==1])
        num_valid = len(valid_eval_score_list)
        num_exclude_eval_error = len([sample for sample in saved_data if not sample["eval_content"].startswith("Eval Error")])
        print(f">> Evaluation Finished:\n{sample_num} samples in total\n{num_valid} valid samples | {valid_correct_num} correct samples | accuracy: {valid_correct_num/num_valid*100:.2f}%\n{num_exclude_eval_error} samples excluding eval error | accuracy: {valid_correct_num/num_exclude_eval_error*100:.2f}%")
        

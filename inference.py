import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback
from pathlib import Path
import asyncio
from methods import get_method_class
from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl
import time



def process_sample(args, general_config, sample, output_path, lock, dataset_name=None):
    MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
    mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
    save_data = sample.copy()
    try:
        mas_output = mas.inference(sample)
        if "response" not in mas_output:    # ensure that there is a key named "response"
            raise ValueError(f"The key 'response' is not found in the MAS output: {mas_output}")
        save_data.update(mas_output)
    except Exception as e:
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"
    save_data.update({"token_stats": mas.get_token_stats()})
    write_to_jsonl(lock, output_path, save_data)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the method
    parser.add_argument("--method_name", type=str, default="vanilla", help="MAS name.")
    parser.add_argument("--method_config_name", type=str, default=None, help="The config file name. If None, the default config file will be used.")

    # args related to the model
    # parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="The agent backend to be used for inference.")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=4096, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to dataset
    parser.add_argument("--test_dataset_name", type=str, default="MATH", help="The dataset to be used for testing.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--result_dir", type=str, default="results", help="Path to the results directory.")
    parser.add_argument("--require_val", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process from the dataset.")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for vLLM.")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for batch inference. Processes all samples at once.")
    parser.add_argument("--use_modal_batch", action="store_true", help="Use Modal's .map() for parallel batch inference.")
    parser.add_argument("--start", type=int, default=0, help="Start index for processing samples.")
    args = parser.parse_args()
    
    general_config = vars(args)
    
    # Load model config
    
    if args.use_vllm:
        pass
    else:
        model_api_config = load_model_api_config(args.model_api_config, args.model_name)
        general_config.update({"model_api_config": model_api_config})
        print("-"*50, f"\n>> Model API config: {model_api_config[args.model_name]}")

        model_cfg = model_api_config[args.model_name]["model_list"][0]

    
    if args.debug:
        # MAS inference
        sample = {"query": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."}
        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)

        response = mas.inference(sample)
        
        print(json.dumps(response, indent=4))
        print(f"\n>> Token stats: {json.dumps(mas.get_token_stats(), indent=4)}")
    
    else:
        print(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")

        # load dataset
        with open(f"./datasets/data/{args.test_dataset_name}.json", "r") as f:
            test_dataset = json.load(f)
        
        if args.require_val:
            val_dataset_path = f"./datasets/data/{args.test_dataset_name}_val.json"
            if not os.path.exists(val_dataset_path):
                raise FileNotFoundError(f"Validation dataset not found at {val_dataset_path}. Please provide a valid path.")
            with open(val_dataset_path, "r") as f:
                val_dataset = json.load(f)
        
        # get output path
        output_path = args.output_path if args.output_path is not None else Path(args.result_dir) / f"{args.test_dataset_name}/{args.model_name}/{args.method_name}_infer.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # reserve unprocessed samplesPl
        test_dataset = reserve_unprocessed_queries(output_path, test_dataset)
        if args.max_samples is not None:
            test_dataset = test_dataset[:args.max_samples]
        print(f">> After filtering: {len(test_dataset)} samples")
        
        # optimize mas if required (e.g., GPTSwarm, ADAS, and AFlow)
        if args.require_val:
            # get MAS instance
            MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
            mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
            mas.optimizing(val_dataset)
        
        # inference the mas
        lock = threading.Lock()

        if args.use_vllm:
            # vLLM batch processing - process all samples at once
            MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
            mas = MAS_METHOD(general_config, method_config_name=args.method_config_name, tensor_parallel_size=args.tensor_parallel_size)


            print(f"Processing {len(test_dataset)} samples with vLLM batch inference...")
            mas_outputs = mas.inference_batch(test_dataset, args.test_dataset_name)

            # Save results
            for i, sample in enumerate(test_dataset):
                save_data = sample.copy()
                save_data.update(mas_outputs[i])
                save_data.update({"token_stats": mas.get_token_stats()})
                write_to_jsonl(lock, output_path, save_data)


        else:
            if args.use_modal_batch:

                MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
                mas = MAS_METHOD(general_config, method_config_name=args.method_config_name, tensor_parallel_size=args.tensor_parallel_size)

                print(f"Using Modal batch processing for {len(test_dataset)} samples...")
                BATCH_SIZE = 10
                for start_idx in range(args.start, len(test_dataset), BATCH_SIZE):
                    batch = test_dataset[start_idx:start_idx+BATCH_SIZE]
                    print(f"Processing batch {start_idx//BATCH_SIZE + 1} ({start_idx+1}-{min(start_idx+BATCH_SIZE, len(test_dataset))}/{len(test_dataset)})...")
                    num_attempts = 0
                    while num_attempts < 20:
                        try:
                            mas_outputs = asyncio.run(mas.inference_vllm_batch(batch, args.test_dataset_name))
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                            num_attempts += 1
                            time.sleep(15)
                    
                    if num_attempts >= 3:
                        raise Exception(f"Failed to process batch {start_idx//BATCH_SIZE + 1} after 3 attempts.")
                    
                    # Save results for this batch
                    for i, sample in enumerate(batch):
                        save_data = sample.copy()
                        save_data.update(mas_outputs[i])
                        save_data.update({"token_stats": mas.get_token_stats()})
                        write_to_jsonl(lock, output_path, save_data)


            else:
                # Sequential processing (existing behavior)
                for sample in tqdm(test_dataset, desc="Processing samples"):
                    process_sample(args, general_config, sample, output_path, lock, dataset_name=args.test_dataset_name)
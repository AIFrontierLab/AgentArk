import os
import json

def read_jsonl(f):
    return [json.loads(x) for x in open(f).readlines()]

def write_jsonl(data: list, f):
    with open(f, 'a') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')




def load_model_api_config(model_api_config, model_name):
    with open(model_api_config, "r") as f:
        model_api_config = json.load(f)
    for model_name in model_api_config:
        actural_max_workers = model_api_config[model_name]["max_workers_per_model"] * len(model_api_config[model_name]["model_list"])
        model_api_config[model_name]["max_workers"] = actural_max_workers
    return model_api_config

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')

def read_valid_jsonl(file_name):
    all_data = []
    with open(file_name, "r") as f:
        tmp = f.readlines()
    for line in tmp:
        try:
            all_data.append(json.loads(line))
        except Exception as e:
            print(line)
            print(f"{e}")
    return all_data

def reserve_unprocessed_queries(output_path, test_dataset):
    processed_queries = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset
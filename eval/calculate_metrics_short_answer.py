"""Aggregate metrics CSV files from short_answer_eval.py into a single DataFrame."""
import argparse
import glob
import os
import pandas as pd


# Column names matching the order in short_answer_eval.py
METRIC_COLUMNS = [
    "dataset_name",
    "model_name",
    "num_samples",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bertscore_f1_avg",
    "f1_avg"
]


model_map = {
    "gemma_7b": "google/gemma-7b-it",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics CSV files into a single DataFrame")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory containing dataset subdirectories with metrics files")
    parser.add_argument("--output_file", type=str, default="combined_metrics.csv",
                        help="Output filename for the combined metrics CSV")
    args = parser.parse_args()

    # Find all metrics*.csv files
    pattern = os.path.join(args.output_dir, "**", "metrics*.csv")
    metrics_files = glob.glob(pattern, recursive=True)

    if not metrics_files:
        print(f"No metrics files found in {args.output_dir}")
        return

    print(f"Found {len(metrics_files)} metrics files:")
    for f in metrics_files:
        print(f"  - {f}")

    # Load all metrics
    all_metrics = []
    for filepath in metrics_files:
        data = pd.read_csv(filepath, index_col=0)
        max_length = eval(filepath.split("max_length_")[1].split('/')[0])
        data.loc['max_length', '0'] = max_length
        
        if "generalization" in filepath:
            data.loc['base_model', '0'] = model_map[os.path.basename(data.loc['model_name', '0']).lower()] 
            data.loc['setting', '0'] = "generalization"
            
        else:
            data.loc['base_model', '0'] = data.loc['model_name', '0'] 

        
        if data is not None:
            all_metrics.append(data)
            
        else:
            print(f"No data found in {filepath}")
            

    if not all_metrics:
        print("No valid metrics files loaded")
        return

    # Create DataFrame and save
    df = pd.concat(all_metrics, ignore_index=True, axis=1).T

    df.sort_values(by=["dataset_name", "base_model", "setting", "max_length"], inplace=True)
    
    
    df.to_csv(args.output_file, index=False)

    print(f"\nCombined {len(df)} metrics entries into {args.output_file}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

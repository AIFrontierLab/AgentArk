"""
Calculate metrics for GSM8K dataset predictions.
This script extracts answers from \boxed{} if needed, and calculates accuracy.
"""
import json
import os
import re
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from misc_utils import print_colored
from eval_utils import load_jsonl

from eval.calculate_metrics_numeric import extract_boxed_answer, normalize_answer




def compare_answers(pred: str, ground_truth: str) -> bool:
    """
    Compare prediction with ground truth.
    Returns True if they match (after normalization).
    """
    pred_normalized = normalize_answer(pred)
    ground_truth_normalized = normalize_answer(ground_truth)

    # Direct string match
    if pred_normalized == ground_truth_normalized:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_normalized)
        ground_truth_num = float(ground_truth_normalized)
        # Allow small tolerance for floating point comparison
        return abs(pred_num - ground_truth_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def process_predictions(input_file: str, output_file: str = None):
    """
    Process predictions from GSM8K file.
    - Extracts answers from \boxed{} if answer_extracted field doesn't exist
    - Calculates accuracy metrics
    - Saves results
    """
    print(f"Loading predictions from: {input_file}")

    predictions = load_jsonl(input_file)

    print(f"Loaded {len(predictions)} predictions")

    # Process each prediction
    results = []
    correct_count = 0
    total_count = 0

    for item in tqdm(predictions, desc="Processing predictions"):
        idx = item.get('idx', None)
        question = item['question']
        ground_truth = item['ground_truth']

        # Extract answer from \boxed{} in code field
        code = item.get('code', [])
        assert isinstance(code, list) and len(code) > 0, f"code must be a list and not empty: {code}"
        
        
        pred_answer = extract_boxed_answer(code[0])

        # Compare answers
        is_correct = compare_answers(pred_answer, ground_truth)

        if is_correct:
            correct_count += 1
        total_count += 1

        # Store result
        results.append({
            'idx': idx,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': pred_answer,
            'correct': is_correct
        })

    # Calculate metrics
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    
    incorrect_answers = [result for result in results if not result['correct']]

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Total samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # Save results to Excel
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_metrics.xlsx')

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

    # Also save summary
    summary_file = output_file.replace('.xlsx', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"GSM8K Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Total samples: {total_count}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Incorrect: {total_count - correct_count}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"{'='*60}\n")

    print(f"Summary saved to: {summary_file}")

    return {
        'total': total_count,
        'correct': correct_count,
        'incorrect': total_count - correct_count,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python calculate_metrics_numeric.py <input_jsonl_file> [output_xlsx_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    process_predictions(input_file, output_file)

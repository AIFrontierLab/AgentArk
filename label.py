import json
import re
import torch
from collections import namedtuple, defaultdict
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm

def extract_final_answer(gt_text):
    """Extract the final numeric answer from the ground truth."""
    numbers = re.findall(r'\d+', gt_text)
    return numbers[-1] if numbers else gt_text


def split_solutions(response):
    """Split the response into 'Solution X:' sections."""
    if "===== Solution 1 =====" in response:
        parts = [s.strip() for s in re.split(r'===== Solution \d+ =====', response) if s.strip()]
        
        solutions = [{"id": i + 1, "text": text} for i, text in enumerate(parts)]
    
    else:
        parts = re.split(r'(Solution\s+\d+:)', response)

        if len(parts) == 1:
            return [{"id": 1, "text": response.strip()}]

        solutions = []
        current_id = None
        current_text = []

        for part in parts:
            header_match = re.match(r'Solution\s+(\d+):', part)
            if header_match:
                if current_id is not None:
                    solutions.append({"id": current_id, "text": "".join(current_text).strip()})
                current_id = int(header_match.group(1))
                current_text = []
            else:
                current_text.append(part)

        if current_id is not None:
            solutions.append({"id": current_id, "text": "".join(current_text).strip()})

    return solutions


def build_label_prompt(query, solution_text, gt_answer, dataset_name):
    """Make the labeling prompt."""
    
    SAMPLE_ANSWER_START_STRING = "## Sample Answers (Use them to guide the style of your answer)"
    SAMPLE_ANSWER_END_STRING = "--- End of Sample Answers ---"        
    
    
    prompt = re.sub(
        rf"{re.escape(SAMPLE_ANSWER_START_STRING)}.*?\n{SAMPLE_ANSWER_END_STRING}",
        "",
        query,
        flags=re.DOTALL,
    )
    
    part2 = ""
    if dataset_name == "QMSum":
        assert "## Meeting Transcript" in prompt
        parts = prompt.split("## Meeting Transcript")
        assert len(parts) == 2
        query, part2 = parts
        
    elif dataset_name == "QASPER":
        assert "# Paper Content" in prompt
        parts = prompt.split("# Paper Content")
        assert len(parts) == 2
        query, part2 = parts
        part2 = part2.split("## Conclusion")[0].strip()
    
    prompt = f"""
You are labeling whether a solution correctly solves a question.
--------------------------------
# QUESTION
{query}
--------------------------------
# GROUND TRUTH FINAL ANSWER:
{gt_answer}
--------------------------------
# WHAT YOU SHOULD DO
Does the candidate solution given below contain the correct final answer?
Respond ONLY with: true or false

Rules
- If the ground truth is a number, the solution should match in numeric value.
- If the ground truth is a person's name, the solution might slightly differ, e.g. `Richard Bertrand Spencer`, `Richard B. Spencer`, `Spencer` are all the same person.
--------------------------------
# CANDIDATE SOLUTION:
{solution_text}
""".strip()

    if dataset_name == "QMSum":
        
        prompt += f"""

--------------------------------
## Meeting Transcript
{part2}
"""
        # transcript = parts[0].split("\n")[:1200]
        # transcript = "\n".join(transcript)
        # new_prompt = "--------------------------------".join([""] + parts[1:] + [""]) + "## Meeting Transcript" + transcript
        # prompt = new_prompt
        
    elif dataset_name == "QASPER":
        prompt += f"""
--------------------------------
# Paper Content
{part2}
"""

    return prompt

def truncate_for_vllm(prompt, tokenizer, max_model_len=2048, keep_tail_tokens=256, dataset_name=None):
    """
    Truncate a prompt so that AFTER applying the chat template,
    total tokens <= max_model_len.
    """
    
    chat = [{"role": "user", "content": prompt}]
    full_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    )

    raw_ids = tokenizer.encode(prompt)
    keep_head = max_model_len - keep_tail_tokens
    if keep_head < 0:
        keep_head = max_model_len

    truncated_ids = raw_ids[:keep_head] + raw_ids[-keep_tail_tokens:]
    truncated_prompt = tokenizer.decode(truncated_ids)

    chat = [{"role": "user", "content": truncated_prompt}]
    full_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    )
    if len(full_ids) > max_model_len:
        print(f"Truncated prompt is too long: {len(full_ids)} > {max_model_len}")
        # Hard fallback: keep only last tokens
        truncated_ids = raw_ids[-keep_tail_tokens:]
        truncated_prompt = tokenizer.decode(truncated_ids)

    return truncated_prompt + "\n\n...[TRUNCATED]..."

def label_file_with_vllm(
    input_path,
    output_path,
    model_name="Qwen/Qwen2.5-72B-Instruct",
    start=0,
    end=None,
    max_tokens=5,
    max_model_len=40960,
    dataset_name=None,
):
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=max_model_len,
        max_num_seqs=4
    )

    tokenizer = llm.get_tokenizer()

    guided_decoding = GuidedDecodingParams(choice=["true", "false"])
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        guided_decoding=guided_decoding,
    )

    # Load data
    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = [json.loads(l) for l in fin]

    lines = all_lines[start:end]
    print(f"Processing {len(lines)} items (indices {start} to {start + len(lines) - 1})")

    # ===== PHASE 1: COLLECT ALL PROMPTS =====
    PromptMetadata = namedtuple('PromptMetadata', ['prompt', 'line_idx', 'solution', 'item'])
    all_prompts_metadata = []

    for line_idx, item in enumerate(tqdm(lines, desc="Processing"), start=start):
        if dataset_name in {"Math", "GSM8K"}:
            gt_answer = extract_final_answer(item["gt"])
        else:
            gt_answer = item["gt"]
        solutions = split_solutions(item["response"])

        for solution in solutions:
            full_prompt = build_label_prompt(item["query"], solution["text"], gt_answer, dataset_name=dataset_name)
            truncated = truncate_for_vllm(
                full_prompt,
                tokenizer,
                max_model_len=max_model_len,
                keep_tail_tokens=256,
                dataset_name=dataset_name
            )

            all_prompts_metadata.append(PromptMetadata(
                prompt=truncated,
                line_idx=line_idx,
                solution=solution,
                item=item
            ))

    print(f"Collected {len(all_prompts_metadata)} prompts across {len(lines)} items")

    # ===== PHASE 2: BATCH INFERENCE =====
    prompts = [m.prompt for m in all_prompts_metadata]
    print(f"Running batch inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts), f"Output/prompt mismatch: {len(outputs)} vs {len(prompts)}"

    # ===== PHASE 3: RECONSTRUCT RESULTS =====
    results_by_line = defaultdict(list)

    for output, metadata in zip(outputs, all_prompts_metadata):
        pred = output.outputs[0].text.strip().lower()
        if pred not in ["true", "false"]:
            pred = None
        is_correct = (pred == "true")

        solution_result = {
            "id": metadata.solution["id"],
            "text": metadata.solution["text"],
            "is_correct": is_correct,
        }

        results_by_line[metadata.line_idx].append({
            'solution': solution_result,
            'item': metadata.item
        })

    # Build final output structures
    final_results = []
    for line_idx in sorted(results_by_line.keys()):
        line_data = results_by_line[line_idx]
        item = line_data[0]['item']
        solutions = [entry['solution'] for entry in line_data]

        solutions_sorted = sorted(solutions, key=lambda x: x["id"])
        labels = [s["is_correct"] for s in solutions_sorted]

        result = {
            "query": item["query"],
            "gt": item["gt"],
            "solutions": solutions_sorted,
            "labels": labels,
            "tag": item.get("tag"),
            "source": item.get("source"),
        }
        final_results.append(result)

    # ===== PHASE 4: WRITE OUTPUT =====
    print(f"Writing {len(final_results)} labeled items to {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for result in final_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done. Saved labeled file to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    
    # Max model length for Qwen2.5-72B-Instruct is 32768
    parser.add_argument("--max_model_len", type=int, default=32768) 
    
    args = parser.parse_args()
    assert args.input.endswith("infer.jsonl"), "Input file must end with infer.jsonl"
    assert args.dataset_name in args.input, f"Dataset name must be in input file, got {args.dataset_name} and  {args.input}"
    
    args.output = args.input.replace("infer.jsonl", f"infer_labeled_by_{args.model.split('/')[-1]}.jsonl")
    
    format_map = {
        "QMSum": "short_answer",
        "HotpotQA": "short_answer",
        "NarrativeQA": "short_answer",
        "QASPER": "short_answer",
        "MedMCQA": "multiple_choice",
        "MedQA": "multiple_choice",
        "MMLU": "multiple_choice",
        "GSM8K": "short_answer",
    }

    label_file_with_vllm(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        start=args.start,
        end=args.end,
        max_model_len=args.max_model_len,
        dataset_name=args.dataset_name,
    )

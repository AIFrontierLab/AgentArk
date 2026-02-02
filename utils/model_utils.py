from datasets import load_dataset
from tqdm import tqdm


SAMPLE_ANSWER_START_STRING = "## Sample Answers (Use them to guide the style of your answer)"
SAMPLE_ANSWER_END_STRING = "--- End of Sample Answers ---"

# Maximum prompt tokens by model family
MODEL_MAX_PROMPT_TOKENS = {
    "google/gemma-7b-it": 40960,
    "Qwen/Qwen3-0.6B": 40960,
    "Qwen/Qwen3-1.7B": 40960,
    "Qwen/Qwen3-8B": 40960,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
}


def get_max_prompt_tokens_for_model(model_name_or_path, default=40960):
    """Get max prompt tokens based on model name."""
    return MODEL_MAX_PROMPT_TOKENS.get(model_name_or_path, default)



def truncate_context(transcript_formatted, tokenizer, max_tokens):
    """
    Truncate context (evidence) to fit within token budget.

    Args:
        transcript_formatted: String of formatted transcript
        tokenizer: HuggingFace tokenizer for token counting
        max_tokens: Maximum tokens allowed for transcript

    Returns:
        (truncated_text, was_truncated)
    """
    # Count current tokens
    tokens = tokenizer.encode(transcript_formatted, add_special_tokens=False)
    current_count = len(tokens)

    # Check if truncation needed
    if current_count <= max_tokens:
        return transcript_formatted, False

    # Keep first max_tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    return truncated_text, True


def build_dataset(args, tokenizer=None):
    if args.dataset_name == "QMSum":
        data_list = []
        samples_general_queries = [
            {
                "query": "Summarize the meeting",
                "answer": "The team began the meeting by discussing the logistics of setting up the interface for data collection. Some members ran a trial of it earlier and found someone who would make a suitable wizard. The team shared concern about how they would recruit non-university student participants. Grad D introduced the team to the second iteration of the bayes-net model and its schemas. Then, the discussion moved onto controlling the size of the bayes-net as it would otherwise be based on too much information. The team ended the meeting by delving into how the method of creating a Bayes-net in different scenarios could itself be abstracted, i.e. narrowing the input and output factors and the intermediate representation."
            },
            {
                "query": "What was the advantage of using Noisy-ORs?",
                "answer": "The actual number of the inputs in the Bayes-net can create a combinatorial explosion when setting the probabilities. Noisy-OR's can help avoid this by simplifying the probability tables and applying a deterministic function to produce their complete version.",
            },
        ]
        samples_specific_queries = [
            {
                "query": "Summarize the discussion about the current XML format to link up different components in data.",
                "answer": "C developed an XML format that links together utterances based on time tags, essentially creating a lattice. The XML format would be divided into many sections, each with its own ID and timeline tag. The XML format could be modified to deal with smaller linguistic units since that would only entail changing the timestamps. Despite being easy to use, the format was not efficient for smaller linguistic units, like phones. It would work for word units, at best.",
            },
            {
                "query": "What did F think about the current XML format to link up different components in data?",
                "answer": "F was concerned about how the time labels would adjust to smaller phonetic units. F inquired if the time boundaries could be changed by propagating new information throughout the XML. F thought that they could configure different XML files to deal with different units, but it would lead to large file sizes.",
            },
            {
                "query": "What did the Professor think about controlling the size of the combinatorial input?",
                "answer": "The professor was the one to raise the issue and suggested that a knowledge engineering trick could be used to narrow down inputs. He thought that perhaps adding deterministic rules to properties that have actions would be helpful and the property types could be retrieved from the ontology.",
            }
            
        ]
        
        formatted_samples_general_queries = "\n\n".join([f"Sample Question: {sample['query']}\nSample Answer: {sample['answer']}" for sample in samples_general_queries])
        formatted_samples_specific_queries = "\n\n".join([f"Sample Question: {sample['query']}\nSample Answer: {sample['answer']}" for sample in samples_specific_queries])
        dataset = load_dataset(args.dataset_name, split=args.split)
        
        for entry in tqdm(dataset, desc="Building prompts"):

            transcript_formatted = "\n".join([f"{t['speaker']}: {t['content']}"
                                for t in entry['meeting_transcripts']])

            # # # Truncate if needed
            if tokenizer is not None:
                # Reserve a few hundred tokens for prompt template (instructions, samples, headers)
                transcript_budget = args.max_prompt_tokens - 512
                transcript_formatted, _ = truncate_context(
                    transcript_formatted,
                    tokenizer,
                    transcript_budget
                )

            for query_item in entry['general_query_list']:
                prompt = f"""
Use 3-5 sentences to answer the following question based on the meeting transcript. 
You must keep both your reasoning and your final answer concise and to the point. Focus on the main topics, key decisions, and outcomes. Avoid irrelevant or unnecessary thinking.

Question: {query_item['query']}

{SAMPLE_ANSWER_START_STRING}
{formatted_samples_general_queries}
{SAMPLE_ANSWER_END_STRING}

## Meeting Transcript
{transcript_formatted}

"""
                data_list.append({
                    "query": prompt,
                    "gt": query_item["answer"],
                    "context": entry['meeting_transcripts'],
                    "solutions": [],
                    "labels": [],
                    "topic": entry['topic'],
                    "source": "QMSum"
                })
                
            for query_item in entry['specific_query_list']:
                prompt = f"""
Use 3-5 sentences to answer the following question based on the meeting transcript. 
You must keep both your reasoning and your final answer concise and to the point. Focus on the main topics, key decisions, and outcomes. Avoid irrelevant or unnecessary thinking.

Question: {query_item['query']}

{SAMPLE_ANSWER_START_STRING}
{formatted_samples_specific_queries}
{SAMPLE_ANSWER_END_STRING}

## Meeting Transcript
{transcript_formatted}
"""
                data_list.append({
                    "query": prompt,
                    "gt": query_item["answer"],
                    "context": entry['meeting_transcripts'],
                    "solutions": [],
                    "labels": [],
                    "topic": entry['topic'],
                    "source": "QMSum"
                })

    elif args.dataset_name == "QASPER":

        dataset = load_dataset("allenai/qasper", split=args.split, trust_remote_code=True)
        print(f"{'='*50}\n", dataset)

        few_shot_examples = [
            {
                "question": "What is the seed lexicon?",
                "answer": "A vocabulary of positive and negative predicates."
            },
            {
                "question": "What datasets are used in the experiments?",
                "answer": "The authors use the SQuAD and NewsQA datasets."
            },
            {
                "question": "What is the main contribution of this paper?",
                "answer": "A novel approach for minimally supervised learning of affective events using a seed lexicon and bootstrapping method."
            }
        ]

        data_list = []
        formatted_few_shot_examples = "\n\n".join([f"Sample Question: {sample['question']}\nSample Answer: {sample['answer']}" for sample in few_shot_examples])

        for paper in tqdm(dataset, desc="QASPER"):
            # Construct full paper text from sections
            paper_text_parts = []
            paper_text_parts.append(f"# {paper['title']}\n\n## Abstract\n{paper['abstract']}")

            # Add full text sections
            if paper['full_text'] and paper['full_text']['section_name']:
                for section_name, paragraphs in zip(paper['full_text']['section_name'], paper['full_text']['paragraphs']):
                    if section_name and paragraphs:
                        paper_text_parts.append(f"\n## {section_name}\n")
                        paper_text_parts.append("\n\n".join(paragraphs))

            full_paper_text = "\n".join(paper_text_parts)

            # Truncate paper text if tokenizer provided
            if tokenizer is not None:
                # Get max tokens for model, reserve space for prompt template
                max_tokens = get_max_prompt_tokens_for_model(
                    args.model_name_or_path,
                    default=args.max_prompt_tokens
                )
                paper_budget = max_tokens - 512  # Reserve for prompt template
                full_paper_text, _ = truncate_context(
                    full_paper_text,
                    tokenizer,
                    paper_budget
                )

            # Process each question in the paper
            if 'qas' in paper and paper['qas']:
                questions = paper['qas'].get('question', [])
                question_ids = paper['qas'].get('question_id', [])
                answers_list = paper['qas'].get('answers', [])

                for q_idx, (question, question_id, answer_group) in enumerate(zip(questions, question_ids, answers_list)):
                    # Extract the best answer from multiple annotators
                    # Use the first available answer that is not unanswerable
                    best_answer = None

                    if answer_group and 'answer' in answer_group:
                        for answer_obj in answer_group['answer']:
                            if answer_obj.get('unanswerable', False):
                                continue

                            # Prioritize extractive spans, then free form, then yes/no
                            if answer_obj.get('extractive_spans'):
                                best_answer = ", ".join(answer_obj['extractive_spans'])
                                break
                            elif answer_obj.get('free_form_answer'):
                                best_answer = answer_obj['free_form_answer']
                                break
                            elif answer_obj.get('yes_no') is not None:
                                best_answer = answer_obj['yes_no']
                                break
                            
                    # Skip if no valid answer found
                    if not best_answer:
                        continue

                    prompt = f"""
Answer the question using ONLY the provided scientific paper.

Note:
* Keep both your answer and reasoning concise and to the point. Do not generate irrelevant information.
* Keep the answer concise, such as a short phrase or 1-2 sentences.
* Only output the answer string, nothing else.

Question: {question}

{SAMPLE_ANSWER_START_STRING}
{formatted_few_shot_examples}
{SAMPLE_ANSWER_END_STRING}

# Paper Content
{full_paper_text}
"""

                    data_list.append({
                        "query": prompt,
                        "gt": best_answer,
                        "context": full_paper_text,
                        "solutions": [],
                        "labels": [],
                        "source": "QASPER",
                        "paper_id": paper['id'],
                        "question_id": question_id
                    })



        

    elif args.dataset_name == "HotpotQA":
        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=args.split, trust_remote_code=True)
        print(f"{'='*50}\n", dataset)

        data_list = []
        few_shot_examples = [
            {
                "question": "Which magazine was started first Arthur's Magazine or First for Women?",
                "answer": "Arthur's Magazine",
            },
            {
                "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
                "answer": "Delhi"
            },
            {
                "question": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
                "answer": "President Richard Nixon"
            }
        ]
        
        formatted_few_shot_examples = "\n\n".join([f"Sample Question: {sample['question']}\nSample Answer: {sample['answer']}" for sample in few_shot_examples])
        
        for example in dataset:
            # Extract supporting facts to filter context
            supporting_titles = set(example['supporting_facts']['title'])
            

            # Build context from supporting documents only
            context_parts = []
            for i, title in enumerate(example['context']['title']):
                if title in supporting_titles:
                    sentences = example['context']['sentences'][i]
                    text = " ".join(sentences)
                    context_parts.append(f"### {title}\n{text}")

            context = "\n\n".join(context_parts)
            prompt = f"""
Answer the question using ONLY the provided context. 

Note:
* Keep both your answer and reasoning concise and to the point. Do not generate irrelevant information.
* Keep the answer short, such as a short phrase or a sentence. 
* Only output the answer string, nothing else. 
* If the answer is not in the context, reply exactly: `NOT_FOUND`.

Question: {example["question"]}

## Context
{context}

{SAMPLE_ANSWER_START_STRING}
{formatted_few_shot_examples}
{SAMPLE_ANSWER_END_STRING}
"""

            data_list.append({
                "query": prompt,
                "gt": example["answer"],
                "context": context,
                "solutions": [],
                "labels": [],
                "tag": ["HotpotQA", example["type"], example["level"]],
                "source": "HotpotQA"
            })
            
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")

    return data_list

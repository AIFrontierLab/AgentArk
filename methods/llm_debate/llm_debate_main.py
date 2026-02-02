# # The official implementation of LLM Debate https://github.com/composable-models/llm_multiagent_debate offen encounters errors.
# # This is a modified version of the original implementation.

# import os
# from ..mas_base import MAS
# import random
# import json
# import requests
# from vllm import SamplingParams
# from vllm.sampling_params import GuidedDecodingParams
# from tqdm import tqdm

# class LLM_Debate_Main(MAS):
#     def __init__(self, general_config, method_config_name=None):
#         method_config_name = "config_main" if method_config_name is None else method_config_name
#         super().__init__(general_config, method_config_name)

#         self.agents_num = self.method_config["agents_num"]
#         self.rounds_num = self.method_config["rounds_num"]
    
#     def inference(self, sample):

#         query = sample["query"]

#         agent_contexts = [[{"role": "user", "content": f"""{query} Make sure to state your answer at the end of the response."""}] for agent in range(self.agents_num)]

#         for round in range(self.rounds_num):
#             for i, agent_context in enumerate(agent_contexts):
#                 if round != 0:
#                     agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
#                     message = self.construct_message(agent_contexts_other, query, 2*round - 1)
#                     agent_context.append(message)

#                 # response = self.call_llm(messages=agent_context)
#                 # Convert agent_context to a single string
#                 prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in agent_context])
#                 response = self.call_llm(prompt=prompt_str)

#                 agent_context.append({"role": "assistant", "content": response})
        
#         answers = [agent_context[-1]['content'] for agent_context in agent_contexts]
        
#         final_answer = self.aggregate(query, answers)
#         return {"response": final_answer}
    
#     def construct_message(self, agents, question, idx):

#         # Use introspection in the case in which there are no other agents.
#         if len(agents) == 0:
#             return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

#         prefix_string = "These are the recent/updated opinions from other agents: "

#         for agent in agents:
#             agent_response = agent[idx]["content"]
#             response = "\n\n One agent response: ```{}```".format(agent_response)

#             prefix_string = prefix_string + response

#         prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response. \n The original problem is {}.".format(question)
#         return {"role": "user", "content": prefix_string}

#     # def aggregate(self, query, answers):
#     #     aggregate_instruction = f"Task:\n{query}\n\n"
#     #     for i, answer in enumerate(answers):
#     #         aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
#     #     aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task. Please put your final answer in the format of Final Answer: "
#     #     response = self.call_llm(prompt=aggregate_instruction)
#     #     return response

#     # from vllm import SamplingParams



#     # def aggregate(self, query, answers):
#     #     aggregate_instruction = f"Task:\n{query}\n\n"
#     #     for i, answer in enumerate(answers):
#     #         aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
#     #     aggregate_instruction += (
#     #         "Given all the above solutions, reason over them carefully and provide "
#     #         "a final answer to the task. "
#     #         "Please put your final answer strictly as one letter among A, B, C, or D. "
#     #         "Final Answer: "
#     #     )

#     #     # Use guided decoding instead of unsupported allowed_token_texts
#     #     allowed_choices = ["A", "B", "C", "D"]
#     #     guided_decoding = GuidedDecodingParams(choice=allowed_choices)

#     #     sampling_params = SamplingParams(
#     #         temperature=0.0,
#     #         max_tokens=4,
#     #         guided_decoding=guided_decoding,
#     #     )

#     #     # Generate with vLLM — same interface as before
#     #     response = self.call_llm(prompt=aggregate_instruction,
#     #                             sampling_params=sampling_params)

#     #     return response


#     def aggregate(self, query, answers):
#         prompt = f"Task:\n{query}\n\n"
#         for i, answer in enumerate(answers):
#             prompt += f"Solution {i+1}:\n{answer}\n\n"
#         prompt += "\nFinal Answer:"

#         # Force exact output
#         extra_body = {"guided_choice": ["A", "B", "C", "D"]}

#         response = self.call_llm(prompt=prompt, extra_body=extra_body)
#         return response



# The official implementation of LLM Debate https://github.com/composable-models/llm_multiagent_debate offen encounters errors.
# This is a modified version of the original implementation.

import os
from ..mas_base import MAS
import random
import json
import requests
from vllm import SamplingParams, LLM
from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm
from transformers import AutoTokenizer
import time
from asyncio import gather
import asyncio

class LLM_Debate_Main(MAS):
    def __init__(self, general_config, method_config_name=None, tensor_parallel_size=4):
        self.tensor_parallel_size = tensor_parallel_size
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        # self.agents_num = self.method_config["agents_num"]
        self.agents_num = 5
        self.rounds_num = self.method_config["rounds_num"]
    
    def inference(self, sample, dataset_name=None):

        query = sample["query"]

        agent_contexts = [[{"role": "user", "content": f"""{query} Make sure to state your answer at the end of the response."""}] for agent in range(self.agents_num)]

        for round in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1, dataset_name=dataset_name)
                    agent_context.append(message)

                # response = self.call_llm(messages=agent_context)
                # Convert agent_context to a single string
                prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in agent_context])
                response = self.call_llm(prompt=prompt_str)

                agent_context.append({"role": "assistant", "content": response})
        
        answers = [agent_context[-1]['content'] for agent_context in agent_contexts]
        
        formatted_answers = ""
        for i, answer in enumerate(answers):
            formatted_answers += f"Solution {i+1}:\n{answer}\n"
        
        final_answer = self.aggregate(query, answers)
        
        # return {"response": final_answer}
        return {"response": formatted_answers + f"\nSolution {len(answers)+1}:\n" + final_answer}

    async def inference_vllm_batch(self, samples, dataset_name=None):
        """
        Batch inference for multiple samples using Modal's async API.
        Parallelizes agent generation within each round while keeping rounds sequential.

        Args:
            samples: List of sample dictionaries with 'query' keys
            dataset_name: Name of the dataset being processed

        Returns:
            List of dictionaries with 'response' keys
        """

        num_samples = len(samples)

        # Step 1: Initialize agent contexts for all samples
        all_agent_contexts = []
        for sample in samples:
            query = sample["query"]
            agent_contexts = [[{"role": "user", "content": f"{query} Make sure to state your answer at the end of the response."}]
                             for _ in range(self.agents_num)]
            all_agent_contexts.append(agent_contexts)

        # Step 2: Run debate rounds (SEQUENTIAL)
        for round in range(self.rounds_num):
            # Collect all prompts for this round (PARALLEL across all samples and agents)
            all_prompts = []
            prompt_metadata = []  # Track (sample_idx, agent_idx) for each prompt
            print(f"Round {round+1}/{self.rounds_num}")
            for sample_idx, agent_contexts in enumerate(all_agent_contexts):
                for agent_idx, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        # Add message from other agents
                        agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                        message = self.construct_message(agent_contexts_other, samples[sample_idx]["query"], 2*round - 1, dataset_name=dataset_name)
                        agent_context.append(message)

                    # Convert agent_context to a single string
                    prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in agent_context])
                    all_prompts.append(prompt_str)
                    prompt_metadata.append((sample_idx, agent_idx))

            # Batch generate all responses for this round using async HTTP requests
            t0 = time.time()
            tasks = [self.call_llm_async(p) for p in all_prompts]
            responses = await asyncio.gather(*tasks)

            t1 = time.time()
            print(f"Round {round+1}/{self.rounds_num} batch inference time: {t1 - t0:.2f} seconds")

            # Update agent contexts - zip ensures we match response to the right agent
            for response, (sample_idx, agent_idx) in zip(responses, prompt_metadata):
                all_agent_contexts[sample_idx][agent_idx].append({"role": "assistant", "content": response})

        # Step 3: Batch aggregate all samples
        aggregation_prompts = []
        for sample, agent_contexts in zip(samples, all_agent_contexts):
            answers = [agent_context[-1]['content'] for agent_context in agent_contexts]

            # Build aggregation prompt (same as inference_batch method)
            aggregate_instruction = f"Task:\n{sample['query']}\n\n"
            for i, answer in enumerate(answers):
                aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
            aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task. Please put your final answer in the format of Final Answer: "

            aggregation_prompts.append(aggregate_instruction)

        # Batch generate aggregations
        t0 = time.time()
        agg_tasks = [self.call_llm_async(p) for p in aggregation_prompts]
        agg_responses = await asyncio.gather(*agg_tasks)
        t1 = time.time()
        print(f"Aggregation batch inference time: {t1 - t0:.2f} seconds")

        # Step 4: Format results
        results = []
        for sample, agent_contexts, final_answer in zip(samples, all_agent_contexts, agg_responses):
            answers = [agent_context[-1]['content'] for agent_context in agent_contexts]

            formatted_answers = ""
            for i, answer in enumerate(answers):
                formatted_answers += f"===== Solution {i+1} =====\n{answer}\n"

            results.append({
                "response": formatted_answers + f"===== Solution {len(answers)+1} =====\n" + final_answer
            })

        return results


    def inference_batch(self, samples, dataset_name=None):
        """
        Batch inference for multiple samples using vLLM.
        Parallelizes agent generation within each round while keeping rounds sequential.

        Args:
            samples: List of sample dictionaries with 'query' keys

        Returns:
            List of dictionaries with 'response' keys
        """
        
        # Step 1: Initialize vLLM engine (lazy init)
        if not hasattr(self, '_vllm_engine'):
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
            self._vllm_engine = LLM(
                model=self.model_name,
                tensor_parallel_size=len(gpus) if self.tensor_parallel_size is None else self.tensor_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization
            )

            # Initialize tokenizer for chat template formatting
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=self.model_temperature,
            max_tokens=self.model_max_tokens,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]
        )

        # Step 2: Initialize agent contexts for all samples
        all_agent_contexts = []
        for sample in samples:
            query = sample["query"]
            agent_contexts = [[{"role": "user", "content": f"{query} Make sure to state your answer at the end of the response."}]
                             for _ in range(self.agents_num)]
            all_agent_contexts.append(agent_contexts)

        # Step 3: Run debate rounds (SEQUENTIAL)
        for round in range(self.rounds_num):
            # Collect all prompts for this round (PARALLEL across all samples and agents)
            all_prompts = []
            prompt_metadata = []  # Track (sample_idx, agent_idx) for each prompt

            for sample_idx, agent_contexts in enumerate(all_agent_contexts):
                for agent_idx, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        # Add message from other agents
                        agent_contexts_other = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                        message = self.construct_message(agent_contexts_other, samples[sample_idx]["query"], 2*round - 1, dataset_name=dataset_name)
                        agent_context.append(message)
                      
                    """  
                    prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in agent_context])

                    # Format prompt using chat template for proper vLLM formatting
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_str}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    """
                    prompt = self.tokenizer.apply_chat_template(
                        agent_context,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    all_prompts.append(prompt)
                    prompt_metadata.append((sample_idx, agent_idx))

            # Batch generate all responses for this round
            outputs = self._vllm_engine.generate(all_prompts, sampling_params)

            # Update agent contexts - zip ensures we match prompt to the right agent
            for output, (sample_idx, agent_idx) in zip(outputs, prompt_metadata):
                response = output.outputs[0].text
                all_agent_contexts[sample_idx][agent_idx].append({"role": "assistant", "content": response})
                
                # Efficient stats update
                m_stats = self.token_stats[self.model_name]
                m_stats["num_llm_calls"] += 1
                m_stats["prompt_tokens"] += len(output.prompt_token_ids)
                m_stats["completion_tokens"] += len(output.outputs[0].token_ids)
                
        # Step 4: Batch aggregate all samples
        aggregation_prompts = []
        for sample, agent_contexts in zip(samples, all_agent_contexts):
            answers = [agent_context[-1]['content'] for agent_context in agent_contexts]

            # Build aggregation prompt (same as single-sample aggregate() method)
            if dataset_name in {"QMSum", "QASPER", "HotpotQA"}:
                aggregate_instruction = ""
                
            else:
                aggregate_instruction = f"Task:\n{sample['query']}\n\n"
                for i, answer in enumerate(answers):
                    aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
            
            aggregate_instruction = f"Task:\n{sample['query']}\n\n"
            for i, answer in enumerate(answers):
                aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
            aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task. Please put your final answer in the format of Final Answer: "

            # Format using chat template for proper vLLM formatting
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": aggregate_instruction}],
                tokenize=False,
                add_generation_prompt=True
            )
            aggregation_prompts.append(prompt)

        # Batch generate aggregations
        agg_outputs = self._vllm_engine.generate(aggregation_prompts, sampling_params)
        
        # Step 5: Format results
        results = []
        for sample, agent_contexts, agg_output in zip(samples, all_agent_contexts, agg_outputs):
            answers = [agent_context[-1]['content'] for agent_context in agent_contexts]

            formatted_answers = ""
            for i, answer in enumerate(answers):
                formatted_answers += f"===== Solution {i+1} =====\n{answer}\n"

            final_answer = agg_output.outputs[0].text
            results.append({
                "response": formatted_answers + f"===== Solution {len(answers)+1} =====\n" + final_answer
            })

        return results

    def construct_message(self, agents, question, idx, dataset_name=None):

        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

        prefix_string = "# These are the recent/updated opinions from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = f"""
# ONE AGENT RESPONSE
```
{agent_response}
```
----- END OF THE AGENT RESPONSE -----
"""

            prefix_string = prefix_string + response

        prefix_string = prefix_string + f"""
Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.

"""
        if dataset_name not in {"QMSum", "QASPER", "HotpotQA"}:
            prefix_string = prefix_string + f"""
----- THE ORIGINAL PROBLEM -----
{question}
----- END OF THE ORIGINAL PROBLEM -----
"""
        
        return {"role": "user", "content": prefix_string}

    def aggregate(self, query, answers):
        aggregate_instruction = f"Task:\n{query}\n\n"
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        aggregate_instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task. Please put your final answer in the format of Final Answer: "
        response = self.call_llm(prompt=aggregate_instruction)
        return response

    # from vllm import SamplingParams



    # def aggregate(self, query, answers):
    #     aggregate_instruction = f"Task:\n{query}\n\n"
    #     for i, answer in enumerate(answers):
    #         aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
    #     aggregate_instruction += (
    #         "Given all the above solutions, reason over them carefully and provide "
    #         "a final answer to the task. "
    #         "Please put your final answer strictly as one letter among A, B, C, or D. "
    #         "Final Answer: "
    #     )

    #     # Use guided decoding instead of unsupported allowed_token_texts
    #     allowed_choices = ["A", "B", "C", "D"]
    #     guided_decoding = GuidedDecodingParams(choice=allowed_choices)

    #     sampling_params = SamplingParams(
    #         temperature=0.0,
    #         max_tokens=4,
    #         guided_decoding=guided_decoding,
    #     )

    #     # Generate with vLLM — same interface as before
    #     response = self.call_llm(prompt=aggregate_instruction,
    #                             sampling_params=sampling_params)

    #     return response


    # def aggregate(self, query, answers):
    #     prompt = f"Task:\n{query}\n\n"
    #     for i, answer in enumerate(answers):
    #         prompt += f"Solution {i+1}:\n{answer}\n\n"
    #     prompt += "\nFinal Answer:"

    #     # Force exact output
    #     extra_body = {"guided_choice": ["A", "B", "C", "D"]}

    #     response = self.call_llm(prompt=prompt, extra_body=extra_body)
    #     return response


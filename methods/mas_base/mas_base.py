import os
import random
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
import random
import json
import requests
from transformers import AutoTokenizer
from methods.utils import handle_retry_error, load_config
import modal

class MAS():

    def __init__(self, general_config, method_config_name=None, tensor_parallel_size=None):

        if method_config_name is not None:
            # Get the child class's module path
            child_module_path = os.path.dirname(os.path.abspath(self.__class__.__module__.replace('.', '/')))
            self.method_config = load_config(os.path.join(child_module_path, "configs", f"{method_config_name}.yaml"))
        
        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = 0.9
        # Tracking compute costs
        self.token_stats = {
            self.model_name: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }

        self.memory_bank = {}
        self.tools = {}
        
    
    def inference(self, sample):
        """
        sample: data sample (dictionary) to be passed to the MAS
        """
        query = sample["query"]
        # response = self.call_llm(prompt=query)
        extra_body = {"guided_choice": ["A", "B", "C", "D"]}

        response = self.call_llm(prompt=query, extra_body=extra_body)
        return {"response": response}

    def inference_batch(self, samples):
        """
        Batch inference for multiple samples using native vLLM
        vLLM handles batching automatically - just pass all prompts at once.

        Args:
            samples: List of sample dictionaries with 'query' keys

        Returns:
            List of dictionaries with 'response' keys
        """
        from vllm import LLM, SamplingParams

        # Initialize vLLM engine if not already done
        if not hasattr(self, '_vllm_engine'):
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
            self._vllm_engine = LLM(
                model=self.model_name,  # Use model_name directly, not from config
                tensor_parallel_size=len(gpus) if self.tensor_parallel_size is None else self.tensor_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 

        # Extract queries from all samples
        
        prompts = [self.tokenizer.apply_chat_template([{"role": "user", "content": sample["query"]}],
                   tokenize=False, add_generation_prompt=True) for sample in samples]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=self.model_temperature,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]
        )

        # vLLM handles batching automatically - just pass all queries
        outputs = self._vllm_engine.generate(prompts, sampling_params)

        # Sort by request_id to maintain order
        outputs = sorted(outputs, key=lambda x: int(x.request_id))

        # Update token statistics
        stats = self.token_stats.setdefault(self.model_name,
            {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

        for output in outputs:
            stats["num_llm_calls"] += 1
            stats["prompt_tokens"] += len(output.prompt_token_ids)
            stats["completion_tokens"] += len(output.outputs[0].token_ids)

        # Return responses in expected format
        return [{"response": output.outputs[0].text} for output in outputs]

    # @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    # def call_llm(self, prompt=None, system_prompt=None, messages=None, model_name=None, temperature=None):
        
    #     model_name = model_name if model_name is not None else self.model_name
    #     model_dict = random.choice(self.model_api_config[model_name]["model_list"])
    #     model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']
        
    #     if messages is None:
    #         assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
    #         if system_prompt is not None:
    #             messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    #         else:
    #             messages = [{"role": "user", "content": prompt}]
        
    #     model_temperature = temperature if temperature is not None else self.model_temperature

    #     request_dict = {
    #         "model": model_name,
    #         "messages": messages,
    #         "max_tokens": self.model_max_tokens,
    #         "timeout": self.model_timeout
    #     }
    #     if "o1" not in model_name:              # OpenAI's o1 models do not support temperature
    #         request_dict["temperature"] = model_temperature

    #     llm = openai.OpenAI(base_url=model_url, api_key=api_key)
    #     try:
    #         completion = llm.chat.completions.create(**request_dict)
    #         response, num_prompt_tokens, num_completion_tokens = completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens
    #     finally:
    #         llm.close()     # TODO: Check if this is necessary

    #     if isinstance(response, str):       # in cases where response is None or an error message
    #         if model_name not in self.token_stats:
    #             self.token_stats[model_name] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    #         else:
    #             self.token_stats[model_name]["num_llm_calls"] += 1
    #             self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
    #             self.token_stats[model_name]["completion_tokens"] += num_completion_tokens
    #     else:
    #         raise ValueError(f"Invalid response from LLM: {response}")
        
    #     return response
    

    async def call_llm_async(self, prompt: str, model_name=None, temperature=None, extra_body=None) -> str:
        """
        Async version of call_llm for concurrent API requests.
        """
        import aiohttp

        model_name = model_name or self.model_name
        model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        model_name_actual, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']

        assert prompt is not None, "'prompt' must be provided."

        if "/chat/completions" in model_url:
            messages = [{"role": "user", "content": prompt}]
            request_dict = {
                "model": model_name_actual,
                "messages": messages,
                "max_tokens": self.model_max_tokens,
                "temperature": temperature or self.model_temperature
            }
        else:
            request_dict = {
                "model": model_name_actual,
                "prompt": prompt,
                "max_tokens": self.model_max_tokens,
                "temperature": temperature or self.model_temperature
            }

        if extra_body:
            request_dict.update(extra_body)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                model_url,
                headers={"Content-Type": "application/json"},
                json=request_dict,
                timeout=aiohttp.ClientTimeout(total=self.model_timeout)
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

        if "/chat/completions" in model_url:
            response = result["choices"][0]["message"]["content"].strip()
        else:
            response = result["choices"][0]["text"].strip()

        stats = self.token_stats.setdefault(model_name, {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        stats["num_llm_calls"] += 1
        stats["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
        stats["completion_tokens"] += result.get("usage", {}).get("completion_tokens", 0)

        return response

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=lambda retry_state: print("Retry failed:", retry_state)
    )
    # def call_llm(self, prompt=None, model_name=None, temperature=None):
    #     try:
    #         print("Starting call_llm")

    #         model_name = model_name or self.model_name
    #         # print(f"Model requested: {model_name}")

    #         model_dict = random.choice(self.model_api_config[model_name]["model_list"])
    #         model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']
    #         # print(f"Using model_url: {model_url}, api_key: {api_key}")

    #         assert prompt is not None, "'prompt' must be provided."
    #         # print(f"Prompt length: {len(prompt)}")

    #         if "/chat/completions" in model_url:
    #             # Chat endpoint expects "messages"
    #             messages = [{"role": "user", "content": prompt}]
    #             request_dict = {
    #                 "model": model_name,
    #                 "messages": messages,
    #                 "max_tokens": self.model_max_tokens,
    #                 "temperature": temperature or self.model_temperature
    #             }
    #         else:
    #             # Regular completion endpoint expects "prompt"
    #             request_dict = {
    #                 "model": model_name,
    #                 "prompt": prompt,
    #                 "max_tokens": self.model_max_tokens,
    #                 "temperature": temperature or self.model_temperature
    #             }

    #         # print("Sending request...")
    #         resp = requests.post(model_url, headers={"Content-Type": "application/json"},
    #                             data=json.dumps(request_dict),
    #                             timeout=self.model_timeout)
    #         # print(f"HTTP status code: {resp.status_code}")
    #         resp.raise_for_status()  # will raise an HTTPError if status != 200

    #         result = resp.json()
    #         # print(f"Response JSON keys: {list(result.keys())}")

    #         if "/chat/completions" in model_url:
    #             response = result["choices"][0]["message"]["content"].strip()
    #         else:
    #             response = result["choices"][0]["text"].strip()
    #         # print(f"Response: {response[:100]}...")  

    #         stats = self.token_stats.setdefault(model_name, {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    #         stats["num_llm_calls"] += 1
    #         stats["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
    #         stats["completion_tokens"] += result.get("usage", {}).get("completion_tokens", 0)
    #         # print(f"Updated token stats: {stats}")

    #         return response

    #     except Exception as e:
    #         print("Exception in call_llm:", e)
    #         raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=lambda retry_state: print("Retry failed:", retry_state)
    )
    def call_llm(self, prompt=None, model_name=None, temperature=None, extra_body=None):
        try:
            # print("Starting call_llm")

            model_name = model_name or self.model_name
            model_dict = random.choice(self.model_api_config[model_name]["model_list"])
            model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']

            assert prompt is not None, "'prompt' must be provided."

            if "/chat/completions" in model_url:
                messages = [{"role": "user", "content": prompt}]
                request_dict = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": self.model_max_tokens,
                    "temperature": temperature or self.model_temperature
                }
            else:
                request_dict = {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": self.model_max_tokens,
                    "temperature": temperature or self.model_temperature
                }

            # <-- Apply extra_body BEFORE sending request
            if extra_body:
                request_dict.update(extra_body)

            resp = requests.post(model_url,
                                headers={"Content-Type": "application/json"},
                                data=json.dumps(request_dict),
                                timeout=self.model_timeout)
            resp.raise_for_status()
            result = resp.json()

            if "/chat/completions" in model_url:
                response = result["choices"][0]["message"]["content"].strip()
            else:
                response = result["choices"][0]["text"].strip()

            stats = self.token_stats.setdefault(model_name, {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            stats["num_llm_calls"] += 1
            stats["prompt_tokens"] += result.get("usage", {}).get("prompt_tokens", 0)
            stats["completion_tokens"] += result.get("usage", {}).get("completion_tokens", 0)

            return response

        except Exception as e:
            print("Exception in call_llm:", e)
            raise



    def get_token_stats(self):
        return self.token_stats
    
    def optimizing(self, val_data):
        """
        For methods that requires validation data such as GPTSwarm and ADAS
        """
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass
    
    def get_tool(self):
        pass


import os
from ..mas_base import MAS
from ..utils import handle_retry_error
import re

import json
import random
import requests
import openai
from tenacity import retry, wait_exponential, stop_after_attempt

from .prompt_main import SystemPromptGenerator 

class CAMEL_Main(MAS):
    def __init__(self, model_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(model_config, method_config_name)

        self.chat_turn_limit = self.method_config["chat_turn_limit"]
        self.assistant_role = self.method_config["assistant_role"]
        self.user_role = self.method_config["user_role"]
        self.system_prompt_generator = SystemPromptGenerator()
        self.with_critic = self.method_config["with_critic"]
        self.option_num = self.method_config["option_num"]
        self.critic_role = self.method_config["critic_role"]

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt=None, system_prompt=None, messages=None, model_name=None, temperature=None, option_num=None, is_multi_options=False):
        
        model_name = model_name if model_name is not None else self.model_name
        model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']
        
        if messages is None:
            assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
        
        model_temperature = temperature if temperature is not None else self.model_temperature

        request_dict = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self.model_max_tokens,
            "timeout": self.model_timeout
        }
        if "o1" not in model_name:              # OpenAI's o1 models do not support temperature
            request_dict["temperature"] = model_temperature
        
        llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        try:
            if is_multi_options and option_num is not None:
                request_dict["n"] = option_num
                completion = llm.chat.completions.create(**request_dict)
                response, num_prompt_tokens, num_completion_tokens = completion, completion.usage.prompt_tokens, completion.usage.completion_tokens
            else:
                completion = llm.chat.completions.create(**request_dict)
                response, num_prompt_tokens, num_completion_tokens = completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens
        except Exception as e:
            print("Error in calling LLM:", e)
            response = None
            num_prompt_tokens, num_completion_tokens = 0, 0
        finally:
            llm.close()
        
        if model_name not in self.token_stats:
            self.token_stats[model_name] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        else:
            self.token_stats[model_name]["num_llm_calls"] += 1
            self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
            self.token_stats[model_name]["completion_tokens"] += num_completion_tokens
        
        return response

    def inference(self, sample):
        query = sample["query"]

        # Get the prompt for the task specified agent 
        _, _, _, task_specify_sys_msg, task_specify_prompt, _ = self.system_prompt_generator.generate(self.assistant_role, self.user_role, query)

        task_specify_messages = [{"role": "system", "content": task_specify_sys_msg}] + [{"role": "user", "content": task_specify_prompt}]
        
        specified_task_msg = self.call_llm(messages=task_specify_messages, is_multi_options=self.with_critic)
        
        response = f"Original idea prompt: {query}\n\n"
        response += f"Specified task prompt: {specified_task_msg}\n\n"

        if self.with_critic:
            # Get the system prompt for the assistant, user and critic
            assistant_sys_msg, user_sys_msg, user_prompt, _, _, critic_sys_msg = self.system_prompt_generator.generate(self.assistant_role, self.user_role, specified_task_msg, critic_role=self.critic_role)       
            self.user_messages = [{"role": "system", "content": user_sys_msg}] + [{"role": "user", "content": user_prompt}]
            self.assistant_messages = [{"role": "system", "content": assistant_sys_msg}]   
            self.critic_messages = [{"role": "system", "content": critic_sys_msg}] 
            
            
            for n in range(self.chat_turn_limit):
                # Get the user multiple choice response
                user_completion = self.call_llm(messages=self.user_messages, option_num=self.option_num, is_multi_options=True) 

                user_response = self.form_user_response(user_completion, self.option_num)
                response += f"User Message: \n{user_response}\n\n"
                
                # Find the response from critic agent and get the selected option
                self.critic_messages.append({"role": "user", "content": user_response})
                critic_response = self.call_llm(messages=self.critic_messages, is_multi_options=True)
                response += f"Critic Message: \n{critic_response}\n\n"
                self.critic_messages.append({"role": "assistant", "content": critic_response})
                selected_option = self.find_option(critic_response)
                selected_user_response = user_completion.choices[selected_option-1].message.content

                
                # Get the assistant multiple choice response based on the selected option of user response
                self.user_messages.append({"role": "assistant", "content": selected_user_response})
                self.assistant_messages.append({"role": "user", "content": selected_user_response})
                assistant_completion = self.call_llm(messages=self.assistant_messages, option_num=self.option_num, is_multi_options=True)
                assistant_response = self.form_assistant_response(assistant_completion, self.option_num)
                response += f"Assistant Message: \n{assistant_response}\n\n"
                
                # Find the response from critic agent and get the selected option
                self.critic_messages.append({"role": "user", "content": assistant_response})
                selected_option = self.find_option(critic_response)
                selected_assistant_response = assistant_completion.choices[selected_option-1].message.content
                self.assistant_messages.append({"role": "assistant", "content": selected_assistant_response})
                self.user_messages.append({"role": "user", "content": selected_assistant_response})
                if "CAMEL_TASK_DONE" in selected_user_response:
                    break
        else: 
            # Get the system prompt for the assistant and user
            assistant_sys_msg, user_sys_msg, user_prompt, _, _, _ = self.system_prompt_generator.generate(self.assistant_role, self.user_role, specified_task_msg)
            self.user_messages = [{"role": "system", "content": user_sys_msg}] + [{"role": "user", "content": user_prompt}]
            self.assistant_messages = [{"role": "system", "content": assistant_sys_msg}] 
            
            for n in range(self.chat_turn_limit):
                user_response = self.call_llm(messages=self.user_messages)
                self.user_messages.append({"role": "assistant", "content": user_response})
                self.assistant_messages.append({"role": "user", "content": user_response})
                response += f"User Message: \n{user_response}\n\n"
                
                assistant_response = self.call_llm(messages=self.assistant_messages)
                self.assistant_messages.append({"role": "assistant", "content": assistant_response})
                self.user_messages.append({"role": "user", "content": assistant_response})
                response += f"Assistant Message: \n{assistant_response}\n\n"
                

                if user_response is None:
                    break
                if "CAMEL_TASK_DONE" in user_response:
                    response += "Assistant Message: \nGreat! Let me know if you have any other tasks or questions."
                    break
               
        return {"response": response}

    def form_user_response(self, completion, option_num):
        # Form the user response with the multiple choice options to a standard format
        response = f"""> Proposals from {self.user_role} (RoleType.USER). Please choose an option:\n"""
        for i in range(option_num):
            response += f"""Option {i+1}:\n{completion.choices[i].message.content}\n"""
        response += f"""Please first enter your choice ([1-{option_num}]) and then your explanation and comparison: """
        return response
    
    def form_assistant_response(self, completion, option_num):
        # Form the assistant response with the multiple choice options to a standard format
        response = f"""> Proposals from {self.assistant_role} (RoleType.ASSISTANT). Please choose an option:\n"""
        for i in range(option_num):
            response += f"""Option {i+1}:\n{completion.choices[i].message.content}\n"""
        response += f"""Please first enter your choice ([1-{option_num}]) and then your explanation and comparison: """
        return response
    
    def find_option(self, string):
        # Find the first integer number found in the given string. It means the choice of the critic agent
        match = re.search(r'\d+', string)
        if match:
            return int(match.group())
        else:
            return None
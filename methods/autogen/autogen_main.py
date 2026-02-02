import os
import re
import sys
import io
import subprocess
from ..mas_base import MAS
from .prompt import ASSISTANT_AGENT_SYSTEM_MESSAGE_CODER, ASSISTANT_AGENT_SYSTEM_MESSAGE, DEFAULT_USER_PROXY_AGENT_SYSTEM_MESSAGE

from typing import List, Dict

class AutoGen_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        self.max_turn = self.method_config['max_turn']
        self.is_termination_msg = self.method_config['is_termination_msg']
        self.code_execute = self.method_config['code_execute']
        self.history = []
    
    def inference(self, sample):
        query = sample["query"]
        user_proxy_system_message = DEFAULT_USER_PROXY_AGENT_SYSTEM_MESSAGE
        if self.code_execute:
            assistant_agent_system_message = ASSISTANT_AGENT_SYSTEM_MESSAGE_CODER
        else:
            assistant_agent_system_message = ASSISTANT_AGENT_SYSTEM_MESSAGE
        user_proxy_history = []
        assistant_agent_history = []
        
        # User Proxy to Assistant Agent
        assistant_agent_messages = self.construct_messages(assistant_agent_system_message, assistant_agent_history, query)
        # print(f"[to Assistant Agent]: {assistant_agent_messages}")        
        assistant_agent_response = self.call_llm(None, None, assistant_agent_messages)
        user_proxy_history.append({"content": query, "role": "assistant"})
        assistant_agent_history.append({"content": query, "role": "user"})
        self.history.append({"content": query, "role": "user_proxy"})
        
        if self.is_termination_msg in assistant_agent_response:
            return {"response": assistant_agent_response}
        
        for i in range(self.max_turn - 1):
            
            ## Assistant Agent to User Proxy
            # print(f"user_proxy_system_message: {user_proxy_system_message}")
            user_proxy_messages = self.construct_messages(user_proxy_system_message, user_proxy_history, assistant_agent_response)
            if self.code_execute:
                is_code, code_output = self.process_response(assistant_agent_response)
                if is_code:
                    user_proxy_response = "The output of the your code is:/n" + code_output  
                    # print(f"response of code:", user_proxy_response)
                else:          
                    # print(f"[to User Proxy]: {user_proxy_messages}")
                    user_proxy_response = self.call_llm(None, None, user_proxy_messages)
            else:
                user_proxy_response = self.call_llm(None, None, user_proxy_messages)
            user_proxy_history.append({"content": assistant_agent_response, "role": "user"})
            assistant_agent_history.append({"content": assistant_agent_response, "role": "assistant"})
            self.history.append({"content": assistant_agent_response, "role": "assistant_agent"})
            if self.is_termination_msg in user_proxy_response:
                return {"response": assistant_agent_response}
            
            ## User Proxy to Assistant Agent
            assistant_agent_messages = self.construct_messages(assistant_agent_system_message, assistant_agent_history, user_proxy_response)
            assistant_agent_response = self.call_llm(None, None, assistant_agent_messages)
            # print(f"[to Assistant Agent]: {assistant_agent_messages}")
            user_proxy_history.append({"content": user_proxy_response, "role": "assistant"})
            assistant_agent_history.append({"content": user_proxy_response, "role": "user"})
            self.history.append({"content": user_proxy_response, "role": "user_proxy"})
            if self.is_termination_msg in assistant_agent_response:
                return {"response": assistant_agent_response}
    
        return {"response": assistant_agent_response}
    
    def construct_messages(self, prepend_prompt: str, history: List[Dict], append_prompt: str):
        messages = []
        if prepend_prompt:
            messages.append({"content": prepend_prompt, "role": "system"})
        if prepend_prompt == None:
            messages.append({"content": "", "role": "system"})
        if len(history) > 0:
            messages += history
        if append_prompt:
            messages.append({"content": append_prompt, "role": "user"})
        if append_prompt == None:
            messages.append({"content": "", "role": "user"})
        return messages

    def process_response(self, assistant_agent_response):
        # Extract and run code directly
        has_code, code, code_type = self.extract_code(assistant_agent_response)
        if has_code:
            stdout, stderr = self.run_code(code, code_type)
            print(f"Running {code_type} code")
            print("Standard Output:")
            print(stdout)
            if stderr:
                print("Standard Error:")
                print(stderr)
            return True, stdout
        else:
            return False, ''

    def extract_code(self, response):
        # Extract code from response without saving to file
        # TODO: the extraction pattern could be improved
        python_pattern = r'```python\n(.*?)\n```'
        shell_pattern = r'```sh\n(.*?)\n```'
        
        python_code = re.findall(python_pattern, response, re.DOTALL)
        shell_code = re.findall(shell_pattern, response, re.DOTALL)
        
        if python_code:
            return True, python_code[0], 'python'
        elif shell_code:
            return True, shell_code[0], 'sh'
        else:
            return False, '', ''

    def run_code(self, code, code_type):
        if code_type == 'python':
            try:
                local_vars = {}
                stdout_buffer = io.StringIO()  # Create a string buffer for stdout
                sys.stdout = stdout_buffer  # Redirect stdout to the buffer
                
                exec(code, {}, local_vars)
                
                sys.stdout = sys.__stdout__  # Restore original stdout
                output = stdout_buffer.getvalue()  # Get the captured output
                return output, None
            except Exception as e:
                sys.stdout = sys.__stdout__  # Restore original stdout in case of error
                return '', str(e)
        elif code_type == 'sh':
            result = subprocess.run(code, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr
        else:
            return None, "Unsupported code type"

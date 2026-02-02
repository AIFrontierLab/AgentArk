import os
import re
import xml.etree.ElementTree as ET

from methods.mas_base import MAS
from methods.mapcoder.func_evaluate import evaluate_functional_correctness
from methods.mapcoder.prompt import INPUT_KB_EXEMPLARS, ALGORITHM_PROMPT, SAMPLE_IO_PROMPT, PLANNING, PLANNING_FOR_VERIFICATION, FINAL_CODE_GENARATION, IMPROVING_CODE

class MapCoder_HumanEval(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.k = self.method_config["k"] 
        self.t = self.method_config["t"]
        self.language = self.method_config["language"]

        self.mapping = {
            1: "one (01)",
            2: "two (02)",
            3: "three (03)",
            4: "four (04)",
            5: "five (05)",
            6: "six (06)",
            7: "seven (07)",
            8: "eight (08)",
            9: "nine (09)",
        }

    def inference(self, sample):
        query = sample["query"]
        
        sample_io=self.get_humaneval_sample_io(query)

        input_kb_exemplars = [
            {
                "role": "user",
                "content": INPUT_KB_EXEMPLARS.format(query=query, k=self.mapping[self.k], language=self.language),
            },
        ]

        # Retrieval Agent: find k (user-defined) similar problems
        response = self.call_llm(messages=input_kb_exemplars)

        # Post processing
        response = self.trim_text(
            response, "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response, "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
        response = self.trim_text(
            response, "# Planning to solve this problem:")
        response = self.trim_text(
            response, f"# Let's think step by step to solve this problem in {self.language} programming language.")
        response = self.replace_tag(response, 'algorithm')
        response = self.replace_tag(response, 'description')
        response = self.replace_tag(response, 'code')
        response = self.replace_tag(response, 'planning')

        response = self.parse_xml(response)

        algorithm_prompt = ALGORITHM_PROMPT.format(algorithm=response['algorithm'])
        sample_io_prompt = SAMPLE_IO_PROMPT.format(sample_io=self.get_sample_io_str(sample_io))

        # Planning Agent: aim to create a step-by-step plan for the original problem.
        plannings = []
        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]

            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": PLANNING.format(example_problem=example_problem, example_planning=example_planning, algorithm_prompt=algorithm_prompt, prompt=query, sample_io_prompt=sample_io_prompt)
                }
            ]
            
            # print("\n\n________________________")
            # print(
            #     f"Input for our problem planning using example: {example_no}: ")
            # print(input_for_problem_planning[0]['content'], flush=True)

            planning = self.call_llm(messages=input_for_problem_planning)
            
            # print("\n\n________________________")
            # print("Response from our problem planning: ")
            # print(planning, flush=True)

            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": PLANNING_FOR_VERIFICATION.format(language=self.language, query=query, planning=planning)
                }
            ]

            # print("Input for planning verification: ")
            # print(input_for_planning_verification[0]['content'], flush=True)
            
            verification_res = self.call_llm(messages=input_for_planning_verification)

            verification_res = self.replace_tag(
                verification_res, 'explanation')
            verification_res = self.replace_tag(verification_res, 'confidence')

            verification_res = self.parse_xml(verification_res)

            verification_res['confidence'] = int(
                str(verification_res['confidence']).strip())

            # print("Response from planning verification: ")
            # print(verification_res, flush=True)

            plannings.append((
                planning,
                verification_res['confidence'],
                example
            ))

        plannings.sort(key=lambda x: x[1], reverse=True)
        std_input_prompt = ""   # HumanEval
        
        for planning_with_ex in plannings:
            planning, confidence, example = planning_with_ex

            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": FINAL_CODE_GENARATION.format(language=self.language, algorithm_prompt=algorithm_prompt, prompt=query, planning=planning, sample_io_prompt=sample_io_prompt, std_input_prompt=std_input_prompt)
                }
            ]

            # print("\n\n________________________")
            # print("Input for final code generation: ")
            # print(input_for_final_code_generation[0]['content'], flush=True)

            # Coding Agent: translate the corresponding planning into code to solve the problem
            code = self.call_llm(messages=input_for_final_code_generation)

            code = self.parse_code(code)

            # print("\n\n________________________")
            # print("Response from final code generation: ")
            # print(code, flush=True)

            response = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
            passed = False

            for i in range(1, self.t + 1):
                passed, test_log = evaluate_functional_correctness(
                    sample_io,
                    code
                )

                if passed:
                    break

                # print(f"Input for improving code generation: {i}")
                # Debugging Agent: utilize sample I/O from the problem description to rectify bugs in the generated code
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": IMPROVING_CODE.format(language=self.language, algorithm_prompt=algorithm_prompt, prompt=query, response=response, test_log=test_log, std_input_prompt=std_input_prompt)
                    }
                ]

                # print("\n\n________________________")
                # print("Input for improving code generation: ")
                # print(input_for_improving_code[0]['content'], flush=True)

                response = self.call_llm(messages=input_for_improving_code)

                code = self.parse_code(response)

                # print("\n\n________________________")
                # print("Response from improving code generation: ")
                # print(response, flush=True)

                # got a code that passed all sample test cases
            if passed:
                break

        # print("________________________\n\n", flush=True)
        return {"response": code}


    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str


    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()
    
    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text 
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()
    
    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io
    
    @staticmethod
    def get_humaneval_sample_io(query: str):
        pattern = r'>>> (.*?)\n\s*([^\n>]*)'
        matches = re.findall(pattern, query)
        
        assertions = []
        
        for match in matches:
            function_call = match[0].strip()
            expected_output = match[1].strip()
            
            if expected_output and not expected_output.startswith('>>>'):
                assertions.append(f"assert {function_call} == {expected_output}")
            elif not expected_output:
                assertions.append(f"assert {function_call} is None")
        
        return assertions

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result


    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)
        return self.xml_to_dict(root)

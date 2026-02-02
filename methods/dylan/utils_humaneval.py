import re
import ast
import random
from typing import Dict, List, Tuple, Any

# Constants
CODE_THRESHOLD = 0.9
TOOL_LIST = ['Passer']

# Roles and system prompts
ROLE_MAP = {
    "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).",
    "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "Programmer": "You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment).",
    "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
    "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature)."
}

ROLE_MAP_INIT = {
    "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a series of previous implementations of the same function signature and docstring. You use the previous implementations as a hint and your goal is to write your full implementation again (restate the function signature). Here're some examples.",
    "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples.",
    "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples.",
    "Programmer": "You are an intelligent programmer. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. And you must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment). Here're some examples.",
    "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples.",
    "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a series of previous implementations of the same function signature. You use the previous implementations as a hint and your goal is to complete your full implementation with better accuracy and robustness. Remember to follow the format (restate the function signature). Here're some examples."
}

EXAMPLE_TESTER = """Example:
function signature:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"

unit tests:
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False"""

EXAMPLE_REFLECTOR = """Example:
[previous impl 1]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[previous impl 2]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[reflection 1]:
The implementation will fail when no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.

[reflection 2]:
The implementation has an issue stemming from the while loop `while current_sum + nums[right] <= target:`, which directly accesses `nums[right]` without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to `while right < len(nums) and current_sum + nums[right] <= target:`. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError."""

JUDGE_MAP = {
    "Passer": "",
    "Tester": f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring. Here's the example.

{EXAMPLE_TESTER}""",
    "Reflector": f"You are a Python writing assistant. You will be given a series of function implementations of the same function signature. Write a few sentences to explain whether and why the implementations are wrong. These comments will be used as a hint and your goal is to write your thoughts on the n-th previous implementation after [reflection n]. Here's the example.\n{EXAMPLE_REFLECTOR}",
    "Ranker": "You are a Python writing assistant. You will be given a series of function implementations of the same function signature. You need to choose the best 2 implementations in consideration of correctness, efficiency, and possible corner cases.",
    "Debugger": "You are a debugger, specialized in finding and fixing bugs in Python code. You will be given a function implementation with a bug in it. Your goal is to identify the bug and provide a corrected implementation. Include comments to explain what was wrong and how it was fixed.",
    "QualityManager": "You are a quality manager, ensuring that the code meets high standards in terms of readability, efficiency, and accuracy. You will be given a function implementation and you need to provide a code review. Comment on its correctness, efficiency, and readability, and suggest improvements if needed."
}

JUDGE_PREFIX = {
    "Passer": "[syntax check {}]:\n",
    "Tester": "[unit test results {}]:\n",
    "Reflector": "[reflection {}]:\n",
    "Debugger": "[bug fix {}]:\n",
    "QualityManager": "[code review {}]:\n",
}

JUDGE_NAMES = {
    "Passer": "Syntax Checker",
    "Tester": "Unit Tests",
    "Reflector": "Reflector",
    "Debugger": "Debugger",
    "QualityManager": "Quality Manager",
}

# Utility functions
def extract_last_python_code_block(text):
    # The regular expression pattern for Python code blocks
    pattern = r"```[pP]ython(.*?)```"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # If there are matches, return the last one
    if matches:
        return matches[-1].strip()
    else:
        return None

def parse_code_completion(agent_response, question):
    python_code = extract_last_python_code_block(agent_response)
    if python_code is None:
        if agent_response.count("impl]") == 0:
            python_code = agent_response
        else:
            python_code_lines = agent_response.split("\n")
            python_code = ""
            in_func = False
            for line in python_code_lines:
                if in_func:
                    python_code += line + "\n"
                if "impl]" in line:
                    in_func = True
    if python_code.count("def") == 0:
        python_code = question + python_code
    return python_code

def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def check_function_result(python_code: str, timeout: float = 5.0) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    """
    try:
        # Simple syntax check
        ast.parse(python_code)
        return {"passed": True, "result": "passed"}
    except Exception as e:
        return {"passed": False, "result": f"failed: {e}"}

def parse_ranks(completion, max_num=4):
    if not isinstance(completion, str):
        content = completion["choices"][0]["message"]["content"]
    else:
        content = completion
    pattern = r'\[([1234]),\s*([1234])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > max_num-1:
                return max_num-1
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = random.sample(list(range(max_num)), 2)

    return tops

def most_frequent(clist, cmp_func):
    if not clist:
        return None, 0
        
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter
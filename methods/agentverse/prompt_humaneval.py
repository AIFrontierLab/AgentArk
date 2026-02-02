ROLE_ASSIGNER_PREPEND_PROMPT = """# Role Description
You are the leader of a group of experts, now you need to recruit a small group of experts with diverse identity to correctly write the code to solve the given problems:
{query}

You can recruit {cnt_agents} expert in different fields. What experts will you recruit to better generate an accurate solution?

Here are some suggestion:
{advice}"""

ROLE_ASSIGNER_APPEND_PROMPT = """# Response Format Guidance
You should respond with a list of expert description. For example:
1. an electrical engineer specified in the filed of xxx.
2. an economist who is good at xxx.
3. a lawyer with a good knowledge of xxx.
...

Only respond with the description of each role. Do not include your reason."""

SOLVER_PREPEND_PROMPT = """Can you complete the following code?
```python
{query}
```"""

SOLVER_APPEND_PROMPT = """You are {role_description}. Using the these information, can you provide a correct completion of the code? Explain your reasoning. Your response should contain only Python code. Do not give any additional information. Use ```python to put the completed Python code in markdown quotes. When responding, please include the given code and the completion."""

CRITIC_PREPEND_PROMPT = """You are in a discussion group, aiming to complete the following code function:
```python
{query}
```"""

CRITIC_APPEND_PROMPT = """You are {role_description}. Based on your knowledge, can you check the correctness of the completion given above? You should give your correct solution to the problem step by step. When responding, you should follow the following rules:
1. Analyze the above latest solution and the problem. 
2. If the latest solution is correct, end your response with a special token "[Agree]". 
3. If the latest solution is wrong, write down your critics in the code block and give a corrected code with comment explanation on the modification.
3. Your response should contain only Python code. Do not give any additional information. Use ```python to wrap your Python code in markdown quotes. When responding, please include the given code and the completion.

Now give your response."""

MANAGER_PROMPT = """According to the Previous Solution and the Previous Sentences, select the most appropriate Critic from a specific Role and output the Role.
```python 
{query} 
```
# Previous Solution
The solution you gave in the last step is:
{former_solution}

# Critics
There are some critics on the above solution:
```
{critic_opinions}
```

# Previous Sentences
The previous sentences in the previous rounds is:
{previous_sentence}"""

EXECUTOR_PREPEND_PROMPT = """You are an experienced program tester. Now your team is trying to solve the problem: 
'''
Complete the Python function:
{query}
'''

Your team has given the following answer:
'''
{solution}
'''"""

EXECUTOR_APPEND_PROMPT = """The solution has been written to `tmp/main.py`. Your are going to write the unit testing code for the solution. You should respond in the following json format wrapped with markdown quotes:
```json
{
    "thought": your thought,
    "file_path": the path to write your testing code,
    "code": the testing code,
    "command": the command to change directory and execute your testing code
}
```

Respond only the json, and nothing else."""

EVALUATOR_PREPEND_PROMPT = """# Experts
The experts recruited in this turn includes:
{all_role_description}

# Problem and Writer's Solution
Problem: 
{query}

Writer's Solution: 
{solution}"""

EVALUATOR_APPEND_PROMPT = """You are an experienced code reviewer. As a good reviewer, you carefully check the functional correctness of the given code completion. When the completion is incorrect, you should patiently teach the writer how to correct the completion, but do not give the code directly.

# Response Format Guidance
You must respond in the following format:
Score: (0 or 1, 0 for incorrect and 1 for correct)
Response: (give your advice on how to correct the solution, and your suggestion on on what experts should recruit in the next round)"""
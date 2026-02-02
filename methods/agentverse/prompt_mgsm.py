ROLE_ASSIGNER_PREPEND_PROMPT = """# Role Description
You are the leader of a group of experts, now you are facing a grade school math problem:
${query}

You can recruit ${cnt_agents} expert in different fields.

Here are some suggestion:
${advice}"""

ROLE_ASSIGNER_APPEND_PROMPT = """You can recruit ${cnt_agents} expert in different fields. What experts will you recruit to better generate an accurate solution?

# Response Format Guidance
You should respond with a list of expert description. For example:
1. an electrical engineer specified in the filed of xxx.
2. an economist who is good at xxx.
...

Only respond with the description of each role. Do not include your reason."""

SOLVER_PREPEND_PROMPT = """Solve the following math problem: 
${query} 

This math problem can be answered without any extra information. You should not ask for any extra information."""

SOLVER_APPEND_PROMPT = """You are ${role_description}. Using the information in the chat history and your knowledge, you should provide the correct solution to the math problem. Explain your reasoning. Your final answer must be a single numerical number and nothing else, in the form \boxed{answer}, at the end of your response."""

CRITIC_PREPEND_PROMPT = """You are ${role_description}. You are in a discussion group, aiming to collaborative solve the following math problem:
${query}

Based on your knowledge, give your correct solution to the problem step by step."""

CRITIC_APPEND_PROMPT = """Now compare your solution with the solution given in the chat history and give your response. The final answer is highlighted in the form \boxed{answer}. When responding, you should follow the following rules:
1. This math problem can be answered without any extra information. You should not ask for any extra information. 
2. Compare your solution with the given solution, give your critics. You should only give your critics, don't give your answer.
3. If the final answer in your solution is the same as the final answer in the above provided solution, end your response with a special token "[Agree]"."""

EVALUATOR_PREPEND_PROMPT = """Experts: ${all_role_description}
Problem: ${query}
Solution: 
```
${solution}
```"""

EVALUATOR_APPEND_PROMPT = """You are an experienced mathematic teacher. As a good teacher, you carefully check the correctness of the given solution on a grade school math problem. When the solution is wrong, you should output a correctness of 0 and give your advice to the students on how to correct the solution. When it is correct, output a correctness of 1 and why it is correct. Also check that the final answer is in the form \boxed{answer} at the end of the solution. The answer must be a numerical number (not a equation, fraction, function or variable). You should also give some suggestion on on what experts should recruit in the next round.

You should respond in the following format:
Correctness: (0 or 1, 0 is wrong, and 1 is correct)
Response: (advice to correct the answer or why it is correct)"""
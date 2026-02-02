GEN_PROMPT_MATH = """
    You are a helpful assistant skilled in math problem-solving. 
    Always end your solution with the final numerical answer in latex, using '\\boxed{{<answer>}}'. 
    If there is no solution, reply with an empty boxed '\\boxed{{}}'.
    Please solve the following math problem step by step:
    QUESTION: {problem}
    Provide your detailed solution below:
"""  
SYSTEM_STR_MATH = """
    You are a critical verifier tasked with evaluating mathematical problem-solving. 
    You will be presented with a question and a proposed solution. 
    Your job is to carefully go over and analyze the solution. Follow the instructions.
"""
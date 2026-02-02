GEN_PROMPT_HUMANEVAL = """
    Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
    {problem}
"""
SYSTEM_STR_CODE = """
    You are a critical verifier tasked with evaluating code implementations. 
    You will be presented with a prompt and a code implementation. 
    Your job is to carefully go over and analyze the code. Follow the instructions.
"""
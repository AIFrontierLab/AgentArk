MULTICHOICE_TEMPLATE = """
    Answer the following multiple choice question. Think step by step before answering, and then output the answer in the format of \"The answer is (X)\" at the end, where X is the LETTER of the correct answer.

    QUESTION:
    {problem}

    Think step by step, then end with EXACTLY \"The answer is (X)\", where X is the LETTER of the correct answer. Do not include the answer text itself, only the letter.
    """

SYSTEM_STR_MULTIPLE_CHOICE = """
    You are a critical verifier tasked with evaluating multiple-choice question-answering. 
    You will be presented with a question, the multiple-choice options, and a proposed solution. 
    Your job is to carefully go over and analyze the solution. Follow the instructions.
"""
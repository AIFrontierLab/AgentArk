GEN_PROMPT_MAIN = """
    Solve the problem concisely. Include only the final answer. 
    Problem: {problem}
    Answer:
"""  
SYSTEM_STR_MAIN = """
    You are a critical verifier tasked with evaluating question-answering. 
    You will be presented with a question and a proposed answer. 
    Your job is to carefully go over and analyze the answer. Follow the instructions.
"""

VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER:"

# For verifiers other than direct approval, we ask a follow up message since it is sometimes unclear what the verifier decided
VERA_ASK_FOR_APPROVAL_ONLY_PROMPT = f"To clarify, based on the above analysis, reply with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not include any other text in your response."

VERA_NAMES_TO_PROMPTS = {
    "math_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Go over each step in the proposed solution and check whether it is mathematically correct. Think out load. "
        f"If you reach a step that is incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'."
        f"If you get to the end of all the steps and each step was correct, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "logic_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Go over each step in the proposed solution and check whether it is logically sound. Think out load. "
        f"If you reach a step that is not logically sound, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you get to the end of all the steps and each step was logically sound, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "facts_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Go over each step in the proposed solution and check whether the facts presented are correct. Think out load. "
        f"If you reach a step with incorrect facts, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you get to the end of all the steps and each step had correct facts, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "units_steps": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the units are handled correctly in each step of the solution. Think out loud. "
        f"If you find any issues with the units, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If all units are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_direct": (
        "{prefix}"
        f"INSTRUCTIONS: \n"
        f"Is this solution correct for the given question? "
        f"Respond with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not provide any explanation or additional text."
    ),
    "general_summarize": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Summarize the solution in your own words, explore anything you think may be incorrect. Think out load. "
        f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_diff": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Explain the solution in a different way than it was presented. "
        "Try to find any flaws in the solution. Think out load. "
        f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_edge": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the solution handles edge cases and boundary conditions, test extreme values or special cases. Think out loud. "
        f"If any boundary conditions or edge cases fail, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If all boundary conditions and edge cases are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_mistakes": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the solution has any common mistakes, calculation errors, or misconceptions that typically found in this type of problem. Think out loud. "
        f"If you find any common mistakes, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If no common mistakes are found, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
    "general_domain": (
        "{prefix}"
        "INSTRUCTIONS: \n"
        f"Check if the solution correctly applies relevant domain-knowledge, established theories, and standard practices for this type of problem. Think out loud. "
        f"If any domain knowledge is misapplied or violated, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
        f"If all domain-specific knowledge is correctly applied, reply '{VERA_ANSWER_SYMBOL}True'."
    ),
}
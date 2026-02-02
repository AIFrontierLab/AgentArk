import re
from termcolor import colored
from .prompt_main import VERA_ANSWER_SYMBOL

def compute_aggregated_verification_score(vera_approvals_for_solution, veras):
    num_positive_approvals = 0
    for vera_model, vera_name in veras:
        approval_key = f"{vera_model}_{vera_name}"
        if approval_key in vera_approvals_for_solution:
            num_positive_approvals += vera_approvals_for_solution[approval_key]
        else:
            raise ValueError(f"Missing approval_key {approval_key} in solution_vera_approvals")
    aggregated_verification_score = num_positive_approvals / len(veras)
    return aggregated_verification_score

def extract_verifier_approval(verifier_response: str) -> bool:
    """Extract the verifier's approval from the response."""
    # Get the last answer
    vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
    last_index = verifier_response.lower().rfind(vera_answer_symbol)
    answer = verifier_response[last_index + len(vera_answer_symbol):].strip() if last_index != -1 else None
    
    if not answer:
        print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                      f"and full verifier_response (length {len(verifier_response)}): "
                      f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
        return False
    
    answer = answer.replace("*", "")  # Remove any asterisks (bolding)
    answer = answer.strip().lower()
    if answer == "true" or answer == "true.":
        return True
    elif answer == "false" or answer == "false.":
        return False
    else:
        # Check if 'true' or 'false' is in the first word
        print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                      f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
        first_word = answer.split()[0]
        if "true" in first_word:
            print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
            return True
        elif "false" in first_word:
            print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
            return False
        else:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                          f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                          f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False
        
def last_boxed_only_string(string):
    # Find the location of the last occurrence of the \boxed command
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    # Iterate over the string until find a matching right curly bracket or end of string
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    # Determine whether the matching right curly bracket was successfully found
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)
    
def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)
    
def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
    
def find_code(completion):
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    extracted_answer = extracted_answer[
                               extracted_answer.find(":\n    ") + 2:
                               ]  # remove signature
    return extracted_answer
import ast
import math
import random
import re
import os
from typing import Dict, List, Tuple, Any
from sacrebleu import sentence_bleu

# Import from MAS base
from methods.mas_base import MAS
from .utils_humaneval import *

class DyLAN_HumanEval(MAS):
    """
    Dylan implementation for HumanEval problems, restructured to work with the MAS base class.
    All functionality from LLMNeuron, JudgeNeuron, and CoLLMLP is integrated here.
    """
    
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_humaneval" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
               
        self.seed = self.method_config.get('random_seed', 0)
        self.num_agents = self.method_config.get('num_agents', 4)
        self.agent_roles = self.method_config.get('agent_roles', ["PythonAssistant", "AlgorithmDeveloper", "ComputerScientist", "Programmer"])
        self.num_judges = self.method_config.get('num_judges', 4)
        self.judge_roles = self.method_config.get('judge_roles', ["Tester", "Reflector", "Debugger", "QualityManager"])
        self.num_agentsrounds = self.method_config.get('num_rounds', 3)
        self.activation = self.method_config.get('activation', "listwise") 
        
        # Ensure we have the correct number of roles
        if len(self.agent_roles) != self.num_agents:
            self.agent_roles = self.agent_roles[:self.num_agents] if len(self.agent_roles) > self.num_agents else self.agent_roles + ["PythonAssistant"] * (self.num_agents - len(self.agent_roles))
        
        if len(self.judge_roles) != self.num_judges:
            self.judge_roles = self.judge_roles[:self.num_judges] if len(self.judge_roles) > self.num_judges else self.judge_roles + ["Tester"] * (self.num_judges - len(self.judge_roles))
        
        # Initialize system state
        self.nodes = []
        self.edges = []
    
    def inference(self, sample):
        query = sample["query"]
        random.seed(self.seed)
        
        prompt, entry_point = self.parse_human_eval_query(query)
        
        if not entry_point:
            # Try to extract entry point from prompt
            for line in prompt.split("\n"):
                if line.startswith("def "):
                    potential_entry_point = line.split("def ")[1].split("(")[0].strip()
                    if potential_entry_point:
                        entry_point = potential_entry_point
                        break
        
        if not entry_point:
            raise ValueError("Error: Could not determine the entry point for the function. Please provide a valid HumanEval problem.")
        
        # Reset network state
        self.init_nodes()
        self.zero_grad()
        self.deactivate_all_nodes()
        assert self.num_agentsrounds > 2
        
        unit_tests = []
        
        # First round: activate initial agents in random order
        loop_indices = list(range(self.num_agents))
        random.shuffle(loop_indices)

        activated_agent_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self.activate_llm_node(node_idx, prompt)
            activated_agent_indices.append(node_idx)

        # Activate judges in the first round
        for idx, node_idx in enumerate(range(self.num_agents, self.num_agents + self.num_judges)):
            self.activate_judge_node(node_idx, prompt)

            # Collect unit tests from tester nodes
            if self.nodes[node_idx]["role"] == "Tester":
                unit_tests.extend(self.nodes[node_idx]["unit_tests"])

        # Second round: activate agents in random order
        loop_indices = list(range(self.num_agents + self.num_judges, self.num_agents * 2 + self.num_judges))
        random.shuffle(loop_indices)

        activated_agent_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self.activate_llm_node(node_idx, prompt)
            activated_agent_indices.append(node_idx)
        
            # Check for early consensus after activating a sufficient number of agents
            if idx >= math.floor(2/3 * self.num_agents):
                active_nodes = [self.nodes[i] for i in activated_agent_indices]
                reached, reply = self.check_consensus(active_nodes, list(range(self.num_agents)), prompt, entry_point)
                if reached:
                    agent_nodes = [node for node in self.nodes if node["type"] == "agent" and node["active"]]
                    return {'response': self.all_tests_and_get_final_result(prompt, unit_tests, entry_point, agent_nodes)}

        # Process remaining rounds
        idx_mask = list(range(self.num_agents))
        active_agent_indices = list(range(self.num_agents + self.num_judges, self.num_agents * 2 + self.num_judges))
        
        for rid in range(2, self.num_agentsrounds):
            # Rank agent answers if there are enough agents
            if self.num_agents > 2:
                replies = [self.nodes[idx]["answer"] for idx in active_agent_indices]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                # Use activation function (listwise ranker) to select top answers
                tops = self.listwise_ranker_2(shuffled_replies, prompt)
                idx_mask = list(map(lambda x: active_agent_indices[indices[x]] % (self.num_agents + self.num_judges), tops))

            # Activate judges for this round
            judge_start_idx = (self.num_agents + self.num_judges) * rid - self.num_judges
            for idx, node_idx in enumerate(range(judge_start_idx, judge_start_idx + self.num_judges)):
                self.activate_judge_node(node_idx, prompt)

                # Collect more unit tests
                if self.nodes[node_idx]["role"] == "Tester":
                    unit_tests.extend(self.nodes[node_idx]["unit_tests"])

            # Activate agents for this round (selected by activation function)
            agent_start_idx = (self.num_agents + self.num_judges) * rid
            loop_indices = list(range(agent_start_idx, agent_start_idx + self.num_agents))
            random.shuffle(loop_indices)
            
            active_agent_indices = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    self.activate_llm_node(node_idx, prompt)
                    active_agent_indices.append(node_idx)
                    
                    # Check for consensus after activating sufficient agents
                    if len(active_agent_indices) > math.floor(2/3 * len(idx_mask)):
                        active_nodes = [self.nodes[i] for i in active_agent_indices]
                        reached, reply = self.check_consensus(active_nodes, idx_mask, prompt, entry_point)
                        if reached:
                            agent_nodes = [node for node in self.nodes if node["type"] == "agent" and node["active"]]
                            return {'response': self.all_tests_and_get_final_result(prompt, unit_tests, entry_point, agent_nodes)}
        
        # Return final result if no consensus was reached
        agent_nodes = [node for node in self.nodes if node["type"] == "agent" and node["active"]]
        response = self.all_tests_and_get_final_result(prompt, unit_tests, entry_point, agent_nodes)
        return {"response": response}
    
    def parse_human_eval_query(self, query):
        """Parse a query string to extract prompt and entry_point"""
        entry_point = None
        prompt = query
        
        # Try to extract entry point from the prompt
        for line in query.split("\n"):
            if line.startswith("def "):
                potential_entry_point = line.split("def ")[1].split("(")[0].strip()
                if potential_entry_point:
                    entry_point = potential_entry_point
                    break
        
        return prompt, entry_point
    
    # Message construction functions
    def construct_message(self, responses, question):
        if len(responses) == 0:
            qtemplate = "```python\n{}\n```"
            qtemplate = '''You must complete the python function I give you.
Be sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.
[function impl]:
{}\nOnce more, please follow the template by repeating the original function, then writing the completion.'''.format(qtemplate.format(question))
            return {"role": "user", "content": qtemplate}
        else:
            flag_ranker = False
            ranker = None
            for agent in responses:
                if agent["role"] == "Ranker":
                    flag_ranker = True
                    ranker = agent
                    break
            pre_impls = list(responses[0]["answer"].keys())
            if flag_ranker:
                ranks = ranker["answer"]
                pre_impls = [pre_impl for pre_impl in pre_impls if ranks[pre_impl]]
            
            prefix_string = ""
            for impl_id, pre_impl in enumerate(pre_impls, start=1):
                prefix_string += "[previous impl {}]:\n```python\n{}\n```\n\n".format(impl_id, pre_impl)
                for agent in responses:
                    if agent["role"] == "Ranker":
                        continue
                    reply = agent["answer"][pre_impl]
                    prefix_string += JUDGE_PREFIX[agent["role"]].format(impl_id) + reply + "\n\n"
            
            def connect_judges(judges):
                # connect the name by ',' and "and"
                judges = [judge for judge in judges if judge["role"] != "Ranker"]
                if len(judges) == 0:
                    raise ValueError("No judges")
                if len(judges) == 1:
                    return JUDGE_NAMES[judges[0]["role"]]
                if len(judges) == 2:
                    return JUDGE_NAMES[judges[0]["role"]] + " and " + JUDGE_NAMES[judges[1]["role"]]
                return ", ".join([JUDGE_NAMES[judge["role"]] for judge in judges[:-1]]) + " and " + JUDGE_NAMES[judges[-1]["role"]]

            prefix_string = prefix_string + "You must complete the python function I give you by rectifying previous implementations. Use the other information as a hint.\nBe sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.\n[improved impl]:\n```python\n{}\n```\n\nPlease follow the template by repeating the function signature and complete the new implementation in [improved impl]. If no changes are needed, simply rewrite the implementation in the Python code block.\nAlong with the new implementation, give a score ranged from 1 to 5 to {} in terms of helpfulness. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(question, connect_judges(responses), len(responses)-1)

            return {"role": "user", "content": prefix_string}

    def construct_judge_message(self, responses, question, role):
        if role == "Tester":
            prefix_string = "function signature:\n```python\n{}\n```\n\nunit tests:\n".format(question)
            return {"role": "user", "content": prefix_string}
        elif role == "Reflector":
            prefix_string = "Here are previous implementations of the same function.The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)
                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Write your reflection on these implementations in consideration of correctness, efficiency, and possible corner cases. Put your reflection of the n-th implementation after [reflection n].\nAlong with the reflections, give a score ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "Debugger":
            prefix_string = "Here are previous implementations of the same function.The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)
                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Debug this version of implementation and write your feedback as a debugger. Put your debug information of the n-th implementation after [bug fix n].\nPlease give a score ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "QualityManager":
            prefix_string = "Here are previous implementations of the same function.The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[previous impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)
                prefix_string = prefix_string + response
            prefix_string = prefix_string + "Write your code review on these implementations in multiple aspects. Put your review of the n-th implementation after [code review n].\nAlso, give a score ranged from 1 to 5 to each previous implementation. Put all {} scores in the form like [[1, 5, 2, ...]] at the end of your response.".format(len(responses))
            return {"role": "user", "content": prefix_string}
        elif role == "Ranker":
            prefix_string = "Here are some implementations of the same function. The function has a signature and a docstring explaining its functionality.\n\n"
            for aid, agent_response in enumerate(responses, start=1):
                response = "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)
                prefix_string = prefix_string + response
            prefix_string = prefix_string + "[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.".format(question)
            return {"role": "user", "content": prefix_string}
        else:
            raise NotImplementedError(f"Unknown role: {role}")
            
    def construct_ranking_message(self, responses, question):
        prefix_string = "Here are some implementations of the same function and the thoughts of them. The function has a signature and a docstring explaining its functionality.\n\n"
        for aid, agent_response in enumerate(responses, start=1):
            response = "[function impl {}]:\n```python\n{}\n```\n\n".format(aid, agent_response)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + """[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.""".format(question)
        return {"role": "user", "content": prefix_string}
    
    def listwise_ranker_2(self, responses, question):
        message = self.construct_ranking_message(responses, question)
        completion = self.call_llm(messages=[message])
        return parse_ranks(completion, max_num=len(responses))
    
    def parse_judge_attitude(self, agent_response, question, role, former_results):
        if role == "Passer":
            attitude = dict()
            for res_code in former_results:
                result = check_function_result(res_code)
                if result["passed"]:
                    attitude[res_code] = "The code doesn't have syntax error."
                else:
                    attitude[res_code] = "The code has syntax error: " + result["result"]
            return attitude
        elif role == "Tester":
            attitude = dict()
            def parse_tests(tests: str) -> List[str]:
                candidates = [test.strip() for test in tests.splitlines() if "assert" in test]
                candidates = [test for test in candidates if py_is_syntax_valid(test)]
                candidates = list(set(candidates))
                if len(candidates) > 10:
                    candidates = candidates[:10]
                return candidates
            tests = parse_tests(agent_response)
            if len(tests) == 0:
                for res_code in former_results:
                    attitude[res_code] = ""
                return (attitude, [])
            
            for res_code in former_results:
                passed_tests = []
                failed_tests = []
                for test in tests:
                    result = check_function_result(res_code + "\n" + test)
                    if result["passed"]:
                        passed_tests.append(test)
                    else:
                        failed_tests.append(test + " # output: " + "error")
                    attitude[res_code] = ""
                    if len(passed_tests) != 0:
                        attitude[res_code] += "Passed tests:\n"
                        for test in passed_tests:
                            attitude[res_code] += test + "\n"
                    if len(failed_tests) != 0:
                        if len(passed_tests) != 0:
                            attitude[res_code] += "\n"
                        attitude[res_code] += "Failed tests:\n"
                        for test in failed_tests:
                            attitude[res_code] += test + "\n"
            return (attitude, tests)
        elif role == "Reflector":
            def parse_reflection(response):
                reflections = []
                # Extract content between [reflection n] and [reflection n+1] or [reflection n] and EOF
                matches = re.findall(r'\[reflection \d+\]:\n(.*?)(?=\[reflection \d+\]:|$)', response, re.DOTALL)
                for match in matches:
                    # Strip each extracted reflection content
                    reflections.append(match.strip())
                return reflections
            reflections = parse_reflection(agent_response)
            if len(reflections) != len(former_results):
                reflections = ["No reflection"] * len(former_results)
            attitude = dict()
            for res_code, reflection in zip(former_results, reflections):
                attitude[res_code] = reflection
            return attitude
        elif role == "Debugger":
            def parse_reflection(response):
                reflections = []
                # Extract content between [reflection n] and [reflection n+1] or [reflection n] and EOF
                matches = re.findall(r'\[bug fix \d+\]:\n(.*?)(?=\[bug fix \d+\]:|$)', response, re.DOTALL)
                for match in matches:
                    # Strip each extracted reflection content
                    reflections.append(match.strip())
                return reflections
            reflections = parse_reflection(agent_response)
            if len(reflections) != len(former_results):
                reflections = ["No Bugs"] * len(former_results)
            attitude = dict()
            for res_code, reflection in zip(former_results, reflections):
                attitude[res_code] = reflection
            return attitude
        elif role == "QualityManager":
            def parse_reflection(response):
                reflections = []
                # Extract content between [reflection n] and [reflection n+1] or [reflection n] and EOF
                matches = re.findall(r'\[code review \d+\]:\n(.*?)(?=\[code review \d+\]:|$)', response, re.DOTALL)
                for match in matches:
                    # Strip each extracted reflection content
                    reflections.append(match.strip())
                return reflections
            reflections = parse_reflection(agent_response)
            if len(reflections) != len(former_results):
                reflections = ["No code review"] * len(former_results)
            attitude = dict()
            for res_code, reflection in zip(former_results, reflections):
                attitude[res_code] = reflection
            return attitude
        elif role == "Ranker":
            attitude = dict()
            tops = parse_ranks(agent_response, max_num=len(former_results)) if 2 < len(former_results) <= 4 else [0, 1]
            for res_id, res_code in enumerate(former_results):
                if res_id in tops:
                    attitude[res_code] = True
                else:
                    attitude[res_code] = False
            return attitude
        else:
            raise NotImplementedError(f"Unknown role: {role}")
    
    def find_array(self, text):
        # Find all matches of array pattern
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        if matches:
            # Take the last match and remove spaces
            last_match = matches[-1].replace(' ', '')
            # Convert the string to a list of integers
            try:
                ret = list(map(int, last_match.split(',')))
            except:
                ret = []
            return ret
        else:
            return []
    
    def find_array_for_judge(self, text, role, formers, answer):
        if role == "Ranker":
            results = []
            for former in formers:
                if answer[former]:
                    results.append(5)
                else:
                    results.append(0)
            return results
        if role == "Passer":
            results = []
            for former in formers:
                if answer[former] == "The code doesn't have syntax error.":
                    results.append(5)
                else:
                    results.append(0)
            return results
        if role == "Tester":
            results = []
            for former in formers:
                if "Passed tests:\n" not in answer[former]:
                    results.append(0)
                else:
                    total_tests = len(answer[former].splitlines()) - 1
                    if "Failed tests:\n" in answer[former]:
                        total_tests -= 2
                    flag_pass = False
                    pass_tests = 0
                    for line in answer[former].splitlines():
                        if flag_pass:
                            if "Failed tests:" in line:
                                break
                            pass_tests += 1
                        if "Passed tests:" in line:
                            flag_pass = True
                    pass_tests -= 1
                    results.append(math.ceil(pass_tests / total_tests * 5))
            return results

        if role != "Reflector" and role != "Debugger" and role != "QualityManager":
            raise NotImplementedError("Error init role type")

        # Find all matches of array pattern
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        if matches:
            # Take the last match and remove spaces
            last_match = matches[-1].replace(' ', '')
            # Convert the string to a list of integers
            try:
                ret = list(map(int, last_match.split(',')))
            except:
                ret = []
            return ret
        else:
            return []
            
    def cut_def_question(self, func_code, question, entry_point):
        def parse_imports(src_code):
            res = []
            for line in src_code.split("\n"):
                if "import" in line:
                    res.append(line)
            res = ["    " + line.strip() for line in res]
            return res
        import_lines = parse_imports(func_code)

        def extract_functions_with_body(source_code):
            # Parse the source code to an AST
            tree = ast.parse(source_code)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if the function is nested inside another function
                    # We can determine this by checking the ancestors of the node
                    parents = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    nesting_level = sum(1 for parent in parents if
                                        parent.lineno <= node.lineno and parent.end_lineno >= node.end_lineno)
                    
                    if nesting_level == 1:  # Only top-level functions
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        function_body = source_code.splitlines()[start_line:end_line]
                        functions.append("\n".join(function_body))
                    
            return functions
        try:
            funcs = extract_functions_with_body(func_code)
        except:
            funcs = [func_code]

        def extract_func_def(src_code):
            for line in src_code.split("\n"):
                if "def" in line and entry_point in line:
                    return line
            return ""
        que_func = extract_func_def(question)

        for fiid, func_ins_code in enumerate(funcs):
            if question in func_ins_code:
                func_ins_code = func_ins_code.split(question)[-1]
            elif question.strip() in func_ins_code:
                func_ins_code = func_ins_code.split(question.strip())[-1]
            elif que_func in func_ins_code:
                # remove the line before def
                res_lines = func_ins_code.split("\n")
                func_ins_code = ""
                in_func = False
                for line in res_lines:
                    if in_func:
                        func_ins_code += line + "\n"
                    if "def" in line:
                        in_func = True
            else:
                continue

            other_funcs = []
            for other_func in funcs[:fiid] + funcs[fiid+1:]:
                other_func = other_func.split("\n")
                other_func = other_func[:1] + import_lines + other_func[1:]
                other_func = "\n".join(other_func)
                other_funcs.append(other_func)
                        
            return "\n".join(import_lines) + "\n" + func_ins_code + "\n" + "\n".join(other_funcs)
        
        res_lines = func_code.split("\n")
        func_code = ""
        in_func = False
        for line in res_lines:
            if in_func:
                func_code += line + "\n"
            if "def" in line:
                in_func = True
        
        return "\n".join(import_lines) + "\n" + func_code

    def check_consensus(self, active_nodes, idx_mask, question, entry_point):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [node["answer"] for node in active_nodes]
        python_codes = []
        backup = []
        for cand in candidates:
            result = check_function_result(cand)
            if result["passed"]:
                python_codes.append(cand)
            else:
                backup.append(cand)
        
        if len(python_codes) == 0:
            return False, None

        pred_answers = []
        for python_code in python_codes:
            python_code = self.cut_def_question(python_code, question, entry_point)
            pred_answers.append(python_code)

        # cmp_res function for code completion
        cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= CODE_THRESHOLD * 100
        consensus_answer, ca_cnt = most_frequent(pred_answers, cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def all_tests_and_get_final_result(self, question, unit_tests, entry_point, agent_nodes):
        candidates = [node["answer"] for node in agent_nodes if node["type"] == "agent" and node["active"]]
        candidates = [self.cut_def_question(cand, question, entry_point) for cand in candidates]
        python_codes = []
        for cand in candidates:
            passed_tests = []
            failed_tests = []
            for test in unit_tests:
                result = check_function_result(question + "\n" + cand + "\n" + test)
                if result["passed"]:
                    passed_tests.append(test)
                else:
                    failed_tests.append(test)
            python_codes.append((cand, len(passed_tests), passed_tests, failed_tests))
        
        # Sort the codes based on the number of passed tests in descending order
        sorted_codes = sorted(python_codes, key=lambda x: x[1], reverse=True)
        
        # Get the maximum number of passed tests
        max_passed_tests = sorted_codes[0][1] if sorted_codes else 0
        
        # Filter the codes that have the maximum number of passed tests
        top_codes = [code for code in sorted_codes if code[1] == max_passed_tests]
        if len(top_codes) < 5:
            top_codes = [code for code in sorted_codes if code[1] == max_passed_tests or code[1] == max_passed_tests - 1]
        
        # Randomly select one of the top codes
        selected_code = random.choice(top_codes) if top_codes else (sorted_codes[0] if sorted_codes else (question, 0, [], []))
        
        return selected_code[0]  # Return the code part of the tuple

    def init_nodes(self):
        """Initialize the network of nodes and edges based on configuration"""
        self.nodes = []
        # Create agent nodes for each round
        for rid in range(self.num_agentsrounds):
            for idx, role in enumerate(self.agent_roles):
                node_id = rid * (self.num_agents + self.num_judges) + idx
                self.nodes.append({
                    "id": node_id,
                    "type": "agent",
                    "role": role,
                    "reply": None,
                    "answer": "",
                    "active": False,
                    "question": None,
                    "from_edges": [],
                    "to_edges": [],
                    "importance": 0
                })
                
            # Create judge nodes for each round (except the last)
            if rid < self.num_agentsrounds - 1:
                for idx, role in enumerate(self.judge_roles):
                    node_id = rid * (self.num_agents + self.num_judges) + self.num_agents + idx
                    judge_node = {
                        "id": node_id,
                        "type": "judge",
                        "role": role,
                        "reply": None,
                        "answer": "",
                        "active": False,
                        "question": None,
                        "from_edges": [],
                        "to_edges": [],
                        "importance": 0,
                        "unit_tests": []
                    }
                    self.nodes.append(judge_node)
        
        # Create edges between nodes
        self.edges = []
        for rid in range(self.num_agentsrounds - 1):
            # Connect agents to judges in the same round
            agent_offset = rid * (self.num_agents + self.num_judges)
            judge_offset = agent_offset + self.num_agents
            next_agent_offset = (rid + 1) * (self.num_agents + self.num_judges)
            
            # Connect each agent to each judge in the same round
            for agent_idx in range(self.num_agents):
                agent_id = agent_offset + agent_idx
                for judge_idx in range(self.num_judges):
                    judge_id = judge_offset + judge_idx
                    edge = {"from": agent_id, "to": judge_id, "weight": 0}
                    self.edges.append(edge)
                    self.nodes[agent_id]["to_edges"].append(len(self.edges) - 1)
                    self.nodes[judge_id]["from_edges"].append(len(self.edges) - 1)
            
            # Connect each judge to each agent in the next round
            for judge_idx in range(self.num_judges):
                judge_id = judge_offset + judge_idx
                for agent_idx in range(self.num_agents):
                    next_agent_id = next_agent_offset + agent_idx
                    edge = {"from": judge_id, "to": next_agent_id, "weight": 0}
                    self.edges.append(edge)
                    self.nodes[judge_id]["to_edges"].append(len(self.edges) - 1)
                    self.nodes[next_agent_id]["from_edges"].append(len(self.edges) - 1)
    
    def deactivate_all_nodes(self):
        """Deactivate all nodes in the network"""
        for node in self.nodes:
            node["active"] = False
            node["reply"] = None
            node["answer"] = ""
            node["question"] = None
            node["importance"] = 0
            if node["type"] == "judge" and node["role"] == "Tester":
                node["unit_tests"] = []

    def zero_grad(self):
        """Reset all edge weights to zero"""
        for edge in self.edges:
            edge["weight"] = 0
    
    def activate_llm_node(self, node_id, question):
        """Activate an LLM node (agent)"""
        node = self.nodes[node_id]
        node["question"] = question
        node["active"] = True
        
        # Get context from incoming edges
        contexts = []
        
        # Get system prompt based on whether there are incoming edges
        if len(node["from_edges"]) == 0:
            sys_prompt = ROLE_MAP_INIT[node["role"]]
        else:
            sys_prompt = ROLE_MAP[node["role"]]
        
        contexts = [{"role": "system", "content": sys_prompt}]
        
        # Get context from previous nodes
        formers = []
        for edge_idx in node["from_edges"]:
            edge = self.edges[edge_idx]
            from_node = self.nodes[edge["from"]]
            if from_node["reply"] is not None and from_node["active"]:
                formers.append((from_node, edge_idx))
        
        # Shuffle formers for diversity
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]

        # Construct message
        contexts.append(self.construct_message(formers, question))
        
        # Call LLM with context
        node["reply"] = self.call_llm(messages=contexts)
        
        # Parse answer
        node["answer"] = parse_code_completion(node["reply"], question)
        
        # Parse weights
        weights = self.find_array(node["reply"])
        if len(weights) != len(formers) - 1:
            weights = [0 for _ in range(len(formers))]
        else:
            res_weights = []
            if formers[0]["role"] == "Ranker":
                res_weights.append(3)
            for wid, weight in enumerate(weights):
                res_weights.append(weight)
                if formers[wid + 1]["role"] == "Ranker":
                    res_weights.append(3)
            weights = res_weights
        
        # Update edge weights
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.edges[eid]["weight"] = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
            
        # Normalize weights
        total = sum([self.edges[eid]["weight"] for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.edges[eid]["weight"] /= total
        else:
            for _, eid in formers:
                self.edges[eid]["weight"] = 1 / len(formers)

    def activate_judge_node(self, node_id, question):
        """Activate a Judge node"""
        node = self.nodes[node_id]
        node["question"] = question
        node["active"] = True
        
        # Get context from incoming edges
        sys_prompt = JUDGE_MAP[node["role"]]
        contexts = [{"role": "system", "content": sys_prompt}]
        
        # Get context from previous nodes
        formers = []
        for edge_idx in node["from_edges"]:
            edge = self.edges[edge_idx]
            from_node = self.nodes[edge["from"]]
            if from_node["reply"] is not None and from_node["active"]:
                formers.append((from_node["answer"], edge_idx))
        
        # Shuffle formers for diversity
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]

        # Handle different judge types
        if node["role"] == "Ranker" and len(formers) <= 2:
            node["reply"] = "[1, 2]"
        elif node["role"] not in TOOL_LIST:
            # Construct message for judge
            contexts.append(self.construct_judge_message(formers, question, node["role"]))
            
            # Call LLM with context
            node["reply"] = self.call_llm(messages=contexts)
        else:
            node["reply"] = formers

        # Parse answer
        node["answer"] = self.parse_judge_attitude(node["reply"], question, node["role"], formers)
        if node["role"] == "Tester":
            node["answer"], node["unit_tests"] = node["answer"]

        # Parse weights
        weights = self.find_array_for_judge(node["reply"], node["role"], formers, node["answer"])
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]
        
        # Update edge weights
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.edges[eid]["weight"] = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
            
        # Normalize weights
        total = sum([self.edges[eid]["weight"] for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.edges[eid]["weight"] /= total
        else:
            for _, eid in formers:
                self.edges[eid]["weight"] = 1 / len(formers)

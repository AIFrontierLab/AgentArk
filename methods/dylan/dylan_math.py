import math
import re
import os
from typing import Dict, List, Tuple, Any

from methods.mas_base import MAS
from .utils_math import get_examples

class DyLAN_MATH(MAS):
    """Dylan-Math: Dynamic ListwIse AggregatioN for Math
    A method that uses multiple agents to solve math problems with a debate procedure.
    """

    def __init__(self, general_config, method_config_name=None):
        """Initialize Dylan-Math with configurations."""
        method_config_name = "config_math" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        # Set configuration variables
        self.seed = self.method_config.get("random_seed", 0)
        self.num_agents = self.method_config.get("num_agents", 4)
        self.mode = self.method_config.get("mode", "complex")  # simple or complex
        self.examples = get_examples(self.mode)
        self.system_prompt = "It's a debate. Explain your reasons at each round thoroughly."
        
        if self.mode == "complex":
            self.system_prompt += "\nFollow the given examples and answer the mathematics problem."
    
    def inference(self, sample):

        query = sample["query"]
        
        # Prepare the question with examples
        question = self._prepare_question(query)
        
        # Initialize agent contexts
        agent_contexts = [[{"role": "system", "content": self.system_prompt}, 
                          {"role": "user", "content": question}] 
                         for _ in range(self.num_agents)]
        
        store_contexts = [[{"role": "system", "content": self.system_prompt}] 
                         for _ in range(self.num_agents)]
        
        # First round: each agent provides an initial solution
        consensus = False
        for i, agent_context in enumerate(agent_contexts):
            completion = self.call_llm(messages=agent_context)
            
            assistant_message = {"role": "assistant", "content": completion}
            agent_context.append(assistant_message)
            store_contexts[i].extend(agent_context[1:])
            
            # Check for consensus after at least 2/3 of agents have responded
            if i >= math.floor(2/3 * len(agent_contexts)) and self._check_reach_consensus(agent_contexts[:i+1]):
                consensus = True
                break
        
        if consensus:
            return {'response': self._extract_final_answer(agent_contexts)}
        
        # Second round: debate based on other agents' solutions
        consensus = False
        message = self._construct_message(agent_contexts, question)
        for i, agent_context in enumerate(agent_contexts):
            # Reset the conversation for the next round
            agent_context.pop()
            agent_context.pop()
            agent_context.append({"role": "user", "content": message})
            
            completion = self.call_llm(messages=agent_context)
            
            assistant_message = {"role": "assistant", "content": completion}
            agent_context.append(assistant_message)
            store_contexts[i].extend(agent_context[1:])
            
            # Check consensus again
            if i >= math.floor(2/3 * len(agent_contexts)) and self._check_reach_consensus(agent_contexts[:i+1]):
                consensus = True
                break
        
        if consensus:
            return {'response': self._extract_final_answer(agent_contexts)}
        
        # Third round: ranking and selecting best solutions
        message = self._construct_ranking_message(agent_contexts, question)
        completion = self.call_llm(messages=[{"role": "user", "content": message}])
        
        tops = self._parse_ranks(completion)
        agent_contexts = [agent_contexts[top] for top in tops]
        
        if self._check_reach_consensus(agent_contexts):
            return {'response': self._extract_final_answer(agent_contexts)}
        
        # Final round: debate with selected best solutions
        message = self._construct_message(agent_contexts, question)
        for i, agent_context in enumerate(agent_contexts):
            agent_context.pop()
            agent_context.pop()
            agent_context.append({"role": "user", "content": message})
            
            completion = self.call_llm(messages=agent_context)
            
            assistant_message = {"role": "assistant", "content": completion}
            agent_context.append(assistant_message)
            store_contexts[i].extend(agent_context[1:])
        
        answer = self._extract_final_answer(agent_contexts)
        wrapped_answer = r"\boxed{" + answer + "}"          # This is for fair comparison between evaluation protocols
        return {'response': wrapped_answer}
    
    def _prepare_question(self, query: str) -> str:
        """Prepare the question with appropriate examples."""
        if self.mode == "complex":
            return self.examples + f"\n\nPlease solve the problem below.\nProblem: {query}\nAnswer:"
        else:  # simple
            return self.examples + f"\n\nPlease solve the problem below.\nProblem: {query}\nAnswer:"
    
    def _construct_message(self, agents: List[Dict], question: str) -> str:
        """Construct a message for agents to debate."""
        prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question + "\n\nThese are the solutions to the problem from other agents: "
        
        for agent in agents:
            agent_response = agent[-1]["content"]
            response = f"\n\nOne agent solution: ```{agent_response}```"
            prefix_string += response
        
        prefix_string += "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that the former answers might be all wrong."
        
        return prefix_string
    
    def _construct_ranking_message(self, agents: List[Dict], question: str) -> str:
        """Construct a message for ranking agent solutions."""
        prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question + "\n\nThese are the solutions to the problem from other agents: "
        
        for aid, agent in enumerate(agents, 1):
            agent_response = agent[-1]["content"]
            response = f"\n\nAgent solution {aid}: ```{agent_response}```"
            prefix_string += response
        
        prefix_string += "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
        
        return prefix_string
    
    def _parse_ranks(self, completion: str) -> List[int]:
        """Parse ranking results from completion."""
        content = completion
        pattern = r'\[([1234]),\s*([1234])\]'
        matches = re.findall(pattern, content)
        
        try:
            match = matches[-1]
            tops = [int(match[0])-1, int(match[1])-1]
            
            def clip(x):
                if x < 0:
                    return 0
                if x > 3:
                    return 3
                return x
                
            tops = [clip(x) for x in tops]
        except:
            tops = [0, 1]  # Default to first two agents if parsing fails
        
        return tops
    
    def _extract_math_answer(self, pred_str: str) -> str:
        """Extract the answer from a math solution string."""
        if 'The answer is ' in pred_str:
            pred = pred_str.split('The answer is ')[-1].strip()
        elif 'the answer is ' in pred_str:
            pred = pred_str.split('the answer is ')[-1].strip()
        elif 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            if len(ans) == 0:
                return ""
                
            if ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
                
            a = self._strip_string(a)
            pred = a
        else:
            pattern = r'-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
                
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred[-1] == "/":
                pred = pred[:-1]
                
        pred = self._strip_string(pred)
        
        if 'boxed' in pred:
            ans = pred.split('boxed')[-1]
            if ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
                
            a = self._strip_string(a)
            pred = a
            
        return pred
    
    def _strip_string(self, string: str) -> str:
        """Clean and strip a string for comparison."""
        # linebreaks
        string = string.replace("\n", "")
        
        # remove inverse spaces
        string = string.replace("\\!", "")
        
        # replace \\ with \
        string = string.replace("\\\\", "\\")
        
        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        
        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        
        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        
        # remove dollar signs
        string = string.replace("\\$", "")
        
        # remove units (on the right)
        string = self._remove_right_units(string)
        
        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        
        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        
        # if empty, return empty string
        if len(string) == 0:
            return string
            
        if string[0] == ".":
            string = "0" + string
        
        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]
        
        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)
        
        # remove spaces
        string = string.replace(" ", "")
        
        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)
        
        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"
        
        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = self._fix_a_slash_b(string)
        
        return string
    
    def _fix_fracs(self, string: str) -> str:
        """Fix fraction formatting in strings."""
        substrs = string.split("\\frac")
        new_str = substrs[0]
        
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if len(substr) == 0:
                    continue
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
                            
        return new_str
    
    def _fix_a_slash_b(self, string: str) -> str:
        """Fix a/b formatting to \\frac{a}{b}."""
        if len(string.split("/")) != 2:
            return string
            
        a = string.split("/")[0]
        b = string.split("/")[1]
        
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string
    
    def _remove_right_units(self, string: str) -> str:
        """Remove units from the right of a string."""
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) >= 2
            return splits[0]
        else:
            return string
    
    def _fix_sqrt(self, string: str) -> str:
        """Fix square root formatting in strings."""
        if "\\sqrt" not in string:
            return string
            
        splits = string.split("\\sqrt")
        new_string = splits[0]
        
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
            
        return new_string
    
    def _is_equiv(self, str1: str, str2: str, verbose: bool = False) -> bool:
        """Check if two strings are equivalent after cleaning."""
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False
        
        try:
            ss1 = self._strip_string(str1)
            ss2 = self._strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except:
            return str1 == str2
    
    def _check_reach_consensus(self, agent_contexts: List[Dict]) -> bool:
        """Check if agents have reached a consensus."""
        pred_solutions = [context[-1]["content"] for context in agent_contexts]
        pred_answers = []
        
        for pred_solution in pred_solutions:
            pred_answer = self._extract_math_answer(pred_solution)
            if pred_answer:
                pred_answers.append(pred_answer)
        
        if len(pred_answers) == 0:
            return False
        
        # Find most frequent answer
        consensus_answer, counter = self._most_frequent(pred_answers)
        
        if counter > math.floor(2/3 * len(agent_contexts)):
            return True
        
        return False
    
    def _most_frequent(self, answer_list: List[str]) -> Tuple[str, int]:
        """Find the most frequent answer in a list."""
        if not answer_list:
            return "", 0
            
        counter = 0
        num = answer_list[0]
        
        for i in answer_list:
            current_frequency = sum(self._is_equiv(i, item) for item in answer_list)
            if current_frequency > counter:
                counter = current_frequency
                num = i
        
        return num, counter
    
    def _extract_final_answer(self, agent_contexts: List[Dict]) -> str:
        """Extract the final answer based on consensus."""
        pred_solutions = [context[-1]["content"] for context in agent_contexts]
        pred_answers = []
        
        for pred_solution in pred_solutions:
            pred_answer = self._extract_math_answer(pred_solution)
            if pred_answer:
                pred_answers.append(pred_answer)
        
        if not pred_answers:
            return "No consensus reached. Unable to determine answer."
            
        # Return the most frequent answer as the consensus
        consensus_answer, _ = self._most_frequent(pred_answers)
        return consensus_answer
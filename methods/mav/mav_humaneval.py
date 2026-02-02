import os
import random
from collections import defaultdict

from .util import *
from .prompt_main import *
from .prompt_humaneval import *
from ..mas_base import MAS
from ..utils import load_config

class MAV_HumanEval(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        self.method_config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml"))
        self.vera_temp = self.method_config['vera_temp']
        self.n_solutions = self.method_config['n_solutions']
        self.veras = [
            (self.model_name, "math_steps"),
            (self.model_name, "logic_steps"),
            (self.model_name, "facts_steps"),
            (self.model_name, "units_steps"),
            (self.model_name, "general_direct"),
            (self.model_name, "general_diff"),
            (self.model_name, "general_edge"),
            (self.model_name, "general_domain"),
            (self.model_name, "general_summarize")
        ]   # TODO: the official code uses different models for verifications, here we use same model for fair comparison
        
        # Shuffle and select the previous n_max_verifiers. Implemented by authors of UniMAS
        random.shuffle(self.veras)
        self.veras = self.veras[:self.method_config['n_max_verifiers']]

    def inference(self, sample):
        """
        query: Query to be passed to the MAS
        """
        query = sample['query']
        data = defaultdict(list)
        prompt = GEN_PROMPT_HUMANEVAL.format(problem = query)
        for i in range(self.n_solutions):
            solution = self.call_llm(prompt=prompt)
            answer = find_code(solution)
            data['solutions'].append(solution)
            data['extracted_answers'].append(answer)
            data['all_solution_vera_approvals'].append({})  # Initialize empty approvals for new solution
            data['all_solution_vera_responses'].append({})

        for solution_index, solution in enumerate(data['solutions'][:self.n_solutions]):  
            for vera_model, vera_name in self.veras:
                approval_key = f"{vera_model}_{vera_name}"
            
                # This vera approval has not yet been generated
                system_str = SYSTEM_STR_CODE
                prefix = f"""{system_str}\n\n
                QUESTION:
                {query}\n\n
                PROPOSED SOLUTION:
                {solution}\n\n"""
                user_prompt = VERA_NAMES_TO_PROMPTS[vera_name].format(prefix=prefix)
                response = self.call_llm(model_name=vera_model, prompt=user_prompt, temperature=self.vera_temp)
                messages = [
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": response}, 
                    {"role": "user", "content": VERA_ASK_FOR_APPROVAL_ONLY_PROMPT}
                ]
                approval_response = self.call_llm(model_name=vera_model, messages=messages, temperature=self.vera_temp)
                approval_bool = extract_verifier_approval(approval_response)
                data['all_solution_vera_approvals'][solution_index][approval_key] = approval_bool
                data['all_solution_vera_responses'][solution_index][approval_key] = [response, approval_response]
        
        sampled_indices = list(range(self.n_solutions))
        extracted_answers = []
        for answer in data['extracted_answers']:
            if answer is None:
                extracted_answers.append("")
            else:
                extracted_answers.append(answer)
        extracted_answers = [extracted_answers[i] for i in sampled_indices]
        solution_vera_approvals = [data['all_solution_vera_approvals'][i] for i in sampled_indices]
        # Get the best-rated answer by the verifiers
        agg_score_key = lambda i: compute_aggregated_verification_score(solution_vera_approvals[i], self.veras)
        best_solution_index = max(range(len(solution_vera_approvals)), key=agg_score_key)
        best_extracted_answer = extracted_answers[best_solution_index]
        return {"response": best_extracted_answer}

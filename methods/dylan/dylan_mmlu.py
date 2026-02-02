import os
import random
import re
import math

from methods.mas_base import MAS
from .utils_mmlu import SYSTEM_PROMPT_MMLU, ROLE_MAP, ACTIVATION_MAP, parse_ranks, parse_single_choice, most_frequent

class DyLAN_MMLU(MAS):
    
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_mmlu" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        # Set random seed for reproducibility
        self.seed = self.method_config.get('random_seed', 0)
        
        # Set default roles if not provided
        self.roles = self.method_config.get('roles', ["Assistant", "Mathematician", "Economist", "Psychologist"])
        self.agents_count = len(self.roles)
        self.num_rounds = self.method_config.get('num_rounds', 3)
        self.activation = self.method_config.get('activation', 'listwise')
        self.qtype = self.method_config.get('type', 'single_choice')
        
        # Node attributes (from LLMNeuron)
        self.nodes = []
        for _ in range(self.agents_count * self.num_rounds):
            self.nodes.append({
                'role': None,
                'reply': None,
                'answer': "",
                'active': False,
                'importance': 0,
                'to_edges': [],
                'from_edges': [],
                'question': None
            })
            
        # Initialize nodes and edges
        self.init_network()
    
    def inference(self, sample):

        query = sample["query"]
        
        random.seed(self.seed)
        
        # Reset gradients
        self.zero_grad()
        
        # Reset all nodes
        self.set_allnodes_deactivated()
        assert self.num_rounds > 2

        # First round
        loop_indices = list(range(self.agents_count))
        random.shuffle(loop_indices)

        activated_indices = []
        for node_idx in loop_indices:
            print(0, node_idx)
            self.activate_node(node_idx, query)
            activated_indices.append(node_idx)
        
            if len(activated_indices) >= math.floor(2/3 * self.agents_count):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents_count)))
                if reached:
                    return {'response': reply}

        # Second round
        loop_indices = list(range(self.agents_count, self.agents_count*2))
        random.shuffle(loop_indices)

        activated_indices = []
        for node_idx in loop_indices:
            print(1, node_idx)
            self.activate_node(node_idx, query)
            activated_indices.append(node_idx)
        
            if len(activated_indices) >= math.floor(2/3 * self.agents_count):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents_count)))
                if reached:
                    return {'response': reply}

        # Remaining rounds
        idx_mask = list(range(self.agents_count))
        idxs = list(range(self.agents_count, self.agents_count*2))
        
        for rid in range(2, self.num_rounds):
            if self.agents_count > 3:
                replies = [self.nodes[idx]['reply'] for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                tops = self.listwise_ranker_2(shuffled_replies, query)
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents_count, tops))

            loop_indices = list(range(self.agents_count*rid, self.agents_count*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    print(rid, idx)
                    self.activate_node(node_idx, query)
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return {'response': reply}
        response = most_frequent([self.nodes[idx]['answer'] for idx in idxs], lambda x, y: x == y)[0]
        return {'response': response}
    
    def init_network(self):
        activation = ACTIVATION_MAP[self.activation]
        
        # Set roles for each node
        for rid in range(self.num_rounds):
            for aid in range(self.agents_count):
                node_idx = rid * self.agents_count + aid
                self.nodes[node_idx]['role'] = self.roles[aid]
                
        # Create edges
        agents_last_round = list(range(self.agents_count))
        for rid in range(1, self.num_rounds):
            for idx in range(self.agents_count):
                curr_node = rid * self.agents_count + idx
                for prev_node in agents_last_round:
                    edge = {'weight': 0, 'a1': prev_node, 'a2': curr_node}
                    self.nodes[prev_node]['to_edges'].append(edge)
                    self.nodes[curr_node]['from_edges'].append(edge)
            agents_last_round = [rid * self.agents_count + i for i in range(self.agents_count)]
            
    def deactivate_node(self, idx):
        node = self.nodes[idx]
        node['active'] = False
        node['reply'] = None
        node['answer'] = ""
        node['question'] = None
        node['importance'] = 0

    def set_allnodes_deactivated(self):
        for idx in range(len(self.nodes)):
            self.deactivate_node(idx)

    def zero_grad(self):
        for node in self.nodes:
            for edge in node['to_edges']:
                edge['weight'] = 0

    def weights_parser(self, text):
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        if matches:
            last_match = matches[-1].replace(' ', '')
            def convert(x):
                try:
                    return int(x)
                except:
                    return 0
            try:
                ret = list(map(convert, last_match.split(',')))
            except:
                ret = []
            return ret
        else:
            return []

    def construct_message(self, responses, question):
        if len(responses) == 0:
            prefix_string = "Here is the question:\n" + question + "\n\nPut your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."
            return {"role": "user", "content": prefix_string}

        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D). Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))

        return {"role": "user", "content": prefix_string}

    def get_context(self, node_idx):
        sys_prompt = ROLE_MAP[self.nodes[node_idx]['role']] + "\n" + SYSTEM_PROMPT_MMLU
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(self.nodes[edge['a1']]['reply'], eid) 
                  for eid, edge in enumerate(self.nodes[node_idx]['from_edges']) 
                  if self.nodes[edge['a1']]['reply'] is not None and self.nodes[edge['a1']]['active']]
        return contexts, formers

    def activate_node(self, node_idx, question):
        node = self.nodes[node_idx]
        node['question'] = question
        node['active'] = True
        
        # get context and generate reply
        contexts, formers = self.get_context(node_idx)
        
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        
        contexts.append(self.construct_message(formers, question))
        
        # Using call_llm
        response = self.call_llm(messages=contexts)
        node['reply'] = response
        print(node['reply'])
        
        # parse answer
        node['answer'] = parse_single_choice(node['reply'])
        weights = self.weights_parser(node['reply'])
        
        if len(weights) != len(formers):
            print("miss match!")
            weights = [0 for _ in range(len(formers))]

        # Process weights
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights = [weight for _, weight, _ in sorted_pairs]

        # Update edge weights
        edges = node['from_edges']
        for eid, weight in enumerate(weights):
            edges[eid]['weight'] = weight / 5 if 0 < weight <= 5 else (1 if weight > 5 else 0)
            
        print([edge['weight'] for edge in edges])
        
        # normalize weights
        total = sum(edge['weight'] for edge in edges)
        if total > 0:
            for edge in edges:
                edge['weight'] /= total
        else:
            for edge in edges:
                edge['weight'] = 1 / len(edges)

        print(node['answer'])
        print([edge['weight'] for edge in edges])

    def check_consensus(self, idxs, idx_mask):
        candidates = [self.nodes[idx]['answer'] for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, lambda x, y: x == y)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            print("Consensus answer: {}".format(consensus_answer))
            return True, consensus_answer
        return False, None

    def listwise_ranker_2(self, responses, question):
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."

        message = {"role": "user", "content": prefix_string}
        
        completion = self.call_llm(messages=[message])
        
        return parse_ranks(completion, max_num=len(responses))

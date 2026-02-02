import math
import random
import re
import os

from methods.mas_base import MAS

class DyLAN_Main(MAS):
    """
    Dynamic LLM Agent Network (DyLAN) implementation.
    Inherits from MAS base class.
    """
    
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        """
        Initialize the DyLAN method.
        
        Args:
            general_config: A dictionary containing the model configuration
            method_config_name: Name of the method configuration file
        """
        super().__init__(general_config, method_config_name)
        
        self.seed = self.method_config.get("random_seed", 0)
        
        # Setup DyLAN specific parameters
        self.num_agents = self.method_config.get("num_agents", 4)
        self.num_rounds = self.method_config.get("num_rounds", 3)
        self.activation = self._get_activation_map()[self.method_config.get("activation", "listwise")]
        self.qtype = "open-ended"  # Only support open-ended mode
        self.roles = self.method_config.get("roles", ["Assistant"] * self.num_agents)
        
        # Initialize comparison and parsing functions
        self.cmp_res = lambda x, y: self._sentence_bleu(x, [y], lowercase=True) >= 0.9 * 100
        self.ans_parser = lambda x: x

        # Initialize network
        self.nodes = []  # List of node states
        self.edges = []  # List of edge weights and connections
        self._init_network()

    def inference(self, sample):
        query = sample["query"]
        random.seed(self.seed)
        
        self._zero_grad()
        self._set_allnodes_deactivated()
        assert self.num_rounds > 2

        # First round
        loop_indices = list(range(self.num_agents))
        random.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self._activate_node(node_idx, query)
            activated_indices.append(node_idx)
            if idx >= math.floor(2/3 * self.num_agents):
                reached, reply = self._check_consensus(activated_indices, list(range(self.num_agents)))
                if reached:
                    return {'response': reply}

        # Second round
        loop_indices = list(range(self.num_agents, self.num_agents*2))
        random.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self._activate_node(node_idx, query)
            activated_indices.append(node_idx)
            if idx >= math.floor(2/3 * self.num_agents):
                reached, reply = self._check_consensus(activated_indices, list(range(self.num_agents)))
                if reached:
                    return {'response': reply}

        # Subsequent rounds
        idx_mask = list(range(self.num_agents))
        idxs = list(range(self.num_agents, self.num_agents*2))
        for rid in range(2, self.num_rounds):
            if self.num_agents > 3:
                replies = [self.nodes[idx]['reply'] for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
                
                if self.activation == 0:  # listwise
                    tops = self._listwise_ranker(shuffled_replies, query)
                    idx_mask = list(map(lambda x: idxs[indices[x]] % self.num_agents, tops))

            loop_indices = list(range(self.num_agents*rid, self.num_agents*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    self._activate_node(node_idx, query)
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self._check_consensus(idxs, idx_mask)
                        if reached:
                            return {'response': reply}
        response = self._most_frequent([self.nodes[idx]['answer'] for idx in idxs], self.cmp_res)[0]
        return {'response': response}

    def _get_activation_map(self):
        """Map activation names to their corresponding values"""
        return {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1}

    def _init_network(self):
        """Initialize the network structure with nodes and edges"""
        # Initialize first round nodes
        for idx in range(self.num_agents):
            self.nodes.append({
                'role': self.roles[idx],
                'reply': None,
                'answer': "",
                'active': False,
                'importance': 0,
                'to_edges': [],
                'from_edges': [],
                'question': None
            })
        
        first_round_nodes = list(range(self.num_agents))
        
        # Initialize subsequent rounds
        for rid in range(1, self.num_rounds):
            start_idx = rid * self.num_agents
            for idx in range(self.num_agents):
                node_idx = start_idx + idx
                self.nodes.append({
                    'role': self.roles[idx],
                    'reply': None,
                    'answer': "",
                    'active': False,
                    'importance': 0,
                    'to_edges': [],
                    'from_edges': [],
                    'question': None
                })
                # Connect to previous round
                for prev_idx in first_round_nodes:
                    edge = {'weight': 0, 'from': prev_idx, 'to': node_idx}
                    self.edges.append(edge)
                    self.nodes[prev_idx]['to_edges'].append(len(self.edges) - 1)
                    self.nodes[node_idx]['from_edges'].append(len(self.edges) - 1)
            first_round_nodes = list(range(start_idx, start_idx + self.num_agents))

    def _zero_grad(self):
        """Reset edge weights"""
        for edge in self.edges:
            edge['weight'] = 0

    def _deactivate_node(self, idx):
        """Deactivate a node and reset its state"""
        self.nodes[idx]['active'] = False
        self.nodes[idx]['reply'] = None
        self.nodes[idx]['answer'] = ""
        self.nodes[idx]['question'] = None
        self.nodes[idx]['importance'] = 0

    def _set_allnodes_deactivated(self):
        """Deactivate all nodes"""
        for idx in range(len(self.nodes)):
            self._deactivate_node(idx)

    def _get_role_map(self):
        """Get role descriptions for agents"""
        return {
            "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
            "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
            "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
            "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
            "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient's age, lifestyle and medical history when providing your recommendations.",
            "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
            "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history."
        }

    def _construct_message(self, responses, question):
        """Construct message for LLM"""
        if len(responses) == 0:
            return {"role": "user", "content": question}

        prefix_string = question + "\n\nThese are the responses from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent response " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the answer from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Please answer the question in detail. Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))

        return {"role": "user", "content": prefix_string}

    def _construct_ranking_message(self, responses, question):
        """Construct message for ranking responses"""
        prefix_string = question + "\n\nThese are the responses from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent response " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 answers and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."

        return {"role": "user", "content": prefix_string}

    def _parse_ranks(self, completion, max_num=4):
        """Parse rank information from completion"""
        pattern = r'\[([1234567]),\s*([1234567])\]'
        matches = re.findall(pattern, completion)

        try:
            match = matches[-1]
            tops = [int(match[0])-1, int(match[1])-1]
            def clip(x):
                if x < 0:
                    return 0
                if x > max_num-1:
                    return max_num-1
                return x
            tops = [clip(x) for x in tops]
        except:
            print("error in parsing ranks")
            tops = random.sample(list(range(max_num)), 2)

        return tops

    def _find_array(self, text):
        """Find array patterns in text"""
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        if matches:
            # Take the last match and remove spaces
            last_match = matches[-1].replace(' ', '')
            def convert(x):
                try:
                    return int(x)
                except:
                    return 0
            # Convert the string to a list of integers
            try:
                ret = list(map(convert, last_match.split(',')))
            except:
                ret = []
            return ret
        else:
            return []

    def _activate_node(self, idx, question):
        """Activate a node and process its response"""
        node = self.nodes[idx]
        node['question'] = question
        node['active'] = True
        
        # Get context and generate reply
        sys_prompt = self._get_role_map()[node['role']] + "\n"
        contexts = [{"role": "system", "content": sys_prompt}]
        
        # Get previous responses
        formers = []
        for edge_idx in node['from_edges']:
            edge = self.edges[edge_idx]
            prev_node = self.nodes[edge['from']]
            if prev_node['reply'] is not None and prev_node['active']:
                formers.append((prev_node['reply'], edge_idx))

        # Shuffle and process responses
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]

        contexts.append(self._construct_message(formers, question))
        node['reply'] = self.call_llm(messages=contexts)
        
        # Parse answer and weights
        node['answer'] = self.ans_parser(node['reply'])
        weights = self._find_array(node['reply'])
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]

        # Update edge weights
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights = [weight for _, weight, _ in sorted_pairs]

        for eid, weight in zip(original_idxs, weights):
            self.edges[eid]['weight'] = weight / 5 if 0 < weight <= 5 else (1 if weight > 5 else 0)

        # Normalize weights
        total = sum(self.edges[eid]['weight'] for eid in original_idxs)
        if total > 0:
            for eid in original_idxs:
                self.edges[eid]['weight'] /= total
        else:
            for eid in original_idxs:
                self.edges[eid]['weight'] = 1 / len(original_idxs)

    def _check_consensus(self, idxs, idx_mask):
        """Check for consensus among nodes"""
        candidates = [self.nodes[idx]['answer'] for idx in idxs]
        consensus_answer, ca_cnt = self._most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def _listwise_ranker(self, responses, question):
        """Rank responses and return top indices"""
        assert 2 < len(responses)
        message = self._construct_ranking_message(responses, question)
        completion = self.call_llm(messages=[message])
        return self._parse_ranks(completion, max_num=len(responses))

    def _most_frequent(self, clist, cmp_func):
        """Find most frequent item in list using comparison function"""
        counter = 0
        num = clist[0]

        for i in clist:
            current_frequency = sum(cmp_func(i, item) for item in clist)
            if current_frequency > counter:
                counter = current_frequency
                num = i

        return num, counter

    def _sentence_bleu(self, reference, hypotheses, lowercase=False):
        """Simplified BLEU score calculation for comparing text similarity"""
        if lowercase:
            reference = reference.lower()
            hypotheses = [h.lower() for h in hypotheses]
            
        ref_tokens = reference.split()
        scores = []
        
        for hyp in hypotheses:
            hyp_tokens = hyp.split()
            matches = sum(1 for token in hyp_tokens if token in ref_tokens)
            precision = matches / len(hyp_tokens) if hyp_tokens else 0
            scores.append(precision * 100)
            
        return max(scores)


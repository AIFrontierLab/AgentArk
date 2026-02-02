import os
import copy
import math
import random

from ..mas_base import MAS
from .prompt_main import *

class Node:
    def __init__(self, node_id: int, env, temperature: float = 0.2) -> None:
        self.id: int = node_id
        self.predecessors: list[Node] = []
        self.successors: list[Node] = []
        self.suggestions = None
        self.pre_answers = {}  
        self.generated_answer: str = ""  
        self.temperature: float = temperature
        self.system_message: str = ""
        self.depth: int = 0
        self.env = env

    def interact(self, query: str, previous_answer: str):
        
        # print(f"Node {self.id} is inteacting......")
        self.suggestions = "None."
        if previous_answer !='':
            critic_prompt = INSTRUCTOR_PROMPT.format(
                query=query,
                previous_answer=previous_answer
            )
            self.suggestions = self.env.call_llm(
                prompt=critic_prompt,
                system_prompt=self.system_message,
                temperature=self.temperature 
            )
        
        actor_prompt = ASSISTANT_PROMPT.format(
            query=query,
            previous_answer=previous_answer,
            suggestions=self.suggestions
        )
        
        interacted_answer = self.env.call_llm(
            prompt=actor_prompt,
            system_prompt=self.system_message,
            temperature=self.temperature
        )
        
        return interacted_answer
    
    def aggregate_answers(self, answer_dict: dict) -> str:
        # print(f"Node {self.id} is aggregating......")
        if len(answer_dict) == 1:
            return next(iter(answer_dict.values()))
        aggregation_prompt = CC_PROMPT
        for node_id, answer in answer_dict.items():
            aggregation_prompt += f"[Node {node_id}'s answer:]\n{answer}\n\n"
        
        return self.env.call_llm(
            prompt=aggregation_prompt,
            system_prompt=self.system_message,
            temperature=self.temperature
        )
    
    
class MacNet_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agent_num = self.method_config["agent_num"]
        self.topology = self.method_config["topology"]
        self.reverse = self.method_config["reverse"]
        self.type = self.method_config["type"] if self.method_config["type"]!='default' else "default"
        self.aggregate_retry_limit = self.method_config["aggregate_retry_limit"]
        self.aggregate_unit_num = self.method_config["aggregate_unit_num"]
        self.node_in_id = self.method_config["node_in_id"]
        self.node_out_id = self.method_config["node_out_id"]
        self.edges = []
        self.node_in = Node(node_id=self.node_in_id,env=self)
        self.node_out = Node(node_id=self.node_out_id,env=self)
        self.nodes = {self.node_in_id: self.node_in, self.node_out_id: self.node_out}
        self.depth = None
        self.input_layer = []
        self.output_layer = []
    
    def inference(self, sample):
        query = sample['query']
        random.seed(2025)   # Fix the random seed for reproducibility
        topo = self.generate_graph_topology()
        response = self.graph_inference(query, topo)
        return {"response": response}

    def generate_graph_topology(self):
        generate_topo_methods = {"chain": self.generate_chain,
                    "star": self.generate_star,
                    "tree": self.generate_tree,
                    "net": self.generate_net,
                    "mlp": self.generate_mlp,
                    "random": self.generate_random
                    }
        try:
            generate_topo_method = generate_topo_methods[self.topology]
        except:
            raise ValueError("Invalid topology type specified.")
        generate_topo_method()
        if self.reverse:
            self.edges = [(edge[1], edge[0]) for edge in self.edges]
        graph_structure = [f"{edge[0]}->{edge[1]}" for edge in self.edges]
        return graph_structure
        
    def generate_chain(self):
        for i in range(self.agent_num - 1):
            self.edges.append((i, i + 1))
        assert len(self.edges) == self.agent_num - 1

    def generate_star(self):
        for i in range(1, self.agent_num):
            self.edges.append((0, i))
        assert len(self.edges) == self.agent_num - 1
    
    def generate_tree(self):
        i = 0
        while True:
            self.edges.append((i, 2 * i + 1))
            if len(self.edges) >= self.agent_num - 1:
                break
            self.edges.append((i, 2 * i + 2))
            if len(self.edges) >= self.agent_num - 1:
                break
            i += 1
        assert len(self.edges) == self.agent_num - 1
    
    def generate_net(self):
        for u in range(self.agent_num):
            for v in range(self.agent_num):
                if u < v:
                    self.edges.append((u, v))
        assert len(self.edges) == self.agent_num * (self.agent_num - 1) / 2

    def generate_mlp(self):
        layer_num = int(math.log(self.agent_num, 2))
        layers = [self.agent_num // layer_num for _ in range(layer_num)]
        layers[0] += self.agent_num % layer_num

        end_ids, start_ids = [layers[0]], [0]
        for i in range(1, len(layers)):
            start_ids.append(end_ids[-1])
            end_ids.append(end_ids[-1] + layers[i])

        for i in range(len(layers) - 1):
            for u in range(start_ids[i], end_ids[i]):
                for v in range(start_ids[i + 1], end_ids[i + 1]):
                    self.edges.append((u, v))

    def generate_random(self):
        edge_num = random.randint(self.agent_num-1, self.agent_num*(self.agent_num-1)/2)
        edges_space = [(u, v) for u in range(self.agent_num) for v in range(self.agent_num) if u < v]
        random.shuffle(edges_space)
        for i in range(edge_num):
            (u, v) = edges_space[i]
            self.edges.append((u, v))

    def graph_inference(self, query, topo):
        
        # 1. Get the graph's structure
        for raw_line in topo:
            # Convert string format to node range tuple, e.g. “1-3->5” to [[(1,3)], [(5,5)]]
            line = [
                [
                    tuple(map(int, sub_part.split('-'))) if '-' in sub_part else (int(sub_part), int(sub_part))
                    for sub_part in part.split(',')
                ]
                for part in raw_line.split('->')
            ]
            # Dealing with isolated nodes (single node layer)
            if len(line) == 1:  
                for node_id in range(line[0][0][0], line[0][0][1] + 1):
                    if node_id not in self.nodes:
                        self.nodes[node_id] = Node(node_id,self)

            # Establishment of predecessor-successor relationships between nodes
            for i in range(len(line) - 1):
                from_node_list = line[i]
                to_node_list = line[i + 1]
                for from_node_tuple in from_node_list:
                    for from_node_id in range(from_node_tuple[0], from_node_tuple[1] + 1):
                        for to_node_tuple in to_node_list:
                            for to_node_id in range(to_node_tuple[0], to_node_tuple[1] + 1):

                                if from_node_id not in self.nodes:
                                    self.nodes[from_node_id] = Node(from_node_id,self)
                                if to_node_id not in self.nodes:
                                    self.nodes[to_node_id] = Node(to_node_id,self)
                                self.nodes[from_node_id].successors.append(self.nodes[to_node_id])
                                self.nodes[to_node_id].predecessors.append(self.nodes[from_node_id])
        self.input_layer = self.get_input_layer()
        self.output_layer = self.get_output_layer()

        #Set successor nodes for input nodes
        for input_nodes in self.input_layer:
            if (input_nodes.id != self.node_in.id) and (input_nodes.id != self.node_out.id):
                self.nodes[self.node_in.id].successors.append(self.nodes[input_nodes.id])
                self.nodes[input_nodes.id].predecessors.append(self.nodes[self.node_in.id])
        #Set predecessor nodes for output nodes
        for output_nodes in self.output_layer:
            if output_nodes.id != self.node_out.id and output_nodes.id != self.node_in.id:
                self.nodes[output_nodes.id].successors.append(self.nodes[self.node_out.id])
                self.nodes[self.node_out.id].predecessors.append(self.nodes[output_nodes.id])

        if self.circular_check():
            print("ERROR: The graph has circular dependency!")
            exit(1)
        
        # 2. Assign prompt and temperature to each node
        new_graph = copy.deepcopy(self)
        layer = -1
        layers = []
        while True:
            input_nodes = new_graph.get_input_layer()
            if len(input_nodes) == 0:
                self.depth = layer
                cur_depth = 0
                for Layer in layers:
                    # Assign agent system prompt and temperature to nodes at each layer
                    for node in Layer:
                        self.nodes[node.id].depth = cur_depth
                        self.nodes[node.id].temperature = 1 - cur_depth / self.depth

                        if self.type == 'default':
                            self.nodes[node.id].system_message = "You are helpful an assistant."
                        else:
                            profile_num = random.randint(1, 99)
                            try:
                                prompt=SYSTEM_PROMPT[profile_num]
                                self.nodes[node.id].system_message=prompt
                            except FileNotFoundError:
                                pass
                           
                    cur_depth += 1
                break
            layers.append(input_nodes)

            # Remove the current layer node and prepare to process the next layer
            visited_edges, next_nodes = set(), set()
            for cur_node in input_nodes:
                for next_node in cur_node.successors:
                    visited_edges.add((cur_node.id, next_node.id))
                    next_nodes.add(next_node.id)
            layer += 1
            for edge in visited_edges:
                new_graph.nodes[edge[0]].successors.remove(new_graph.nodes[edge[1]])
                new_graph.nodes[edge[1]].predecessors.remove(new_graph.nodes[edge[0]])
            for cur_node in input_nodes:
                del new_graph.nodes[cur_node.id]

        # 3. The graph starts executing
        while True:
            input_nodes = self.get_input_layer()
            if len(input_nodes) == 0:
                break

            visited_edges, next_nodes = set(), set()
            for cur_node in input_nodes :
          
                for next_node in cur_node.successors:  
                    # Gather predecessors' answers and interact 
                    pre_answer=""
                    predecessor_answers = {
                        p.id: p.generated_answer for p in cur_node.predecessors
                        }
                    for node_id, answer in predecessor_answers.items():
                        pre_answer += f"[Node {node_id}'s answer]\n{answer}\n\n"
                    
                    interacted_answer = next_node.interact(
                        query=query,
                        previous_answer=pre_answer
                        )
                    
                    # Record the results of the interaction
                    next_node.pre_answers[cur_node.id] = interacted_answer
                    visited_edges.add((cur_node.id, next_node.id))
                    next_nodes.add(next_node.id)

            for node_id in next_nodes:
                node = self.nodes[node_id]
                # If the aggregation condition is satisfied, the aggregation algorithm is executed and all pre-answers are merged
                if len(node.pre_answers) == len(node.predecessors) and len(node.pre_answers) >= self.aggregate_unit_num:
                    aggregated = node.aggregate_answers(node.pre_answers)
                    node.generated_answer = aggregated
                else:
                    node.generated_answer = node.pre_answers[list(node.pre_answers.keys())[0]]
                    # print(f"Node {node.id} has insufficient predecessors, uses pre_solution.")
            
            #Remove edges and nodes from the previous layer
            for edge in visited_edges:
                self.nodes[edge[0]].successors.remove(self.nodes[edge[1]])
                self.nodes[edge[1]].predecessors.remove(self.nodes[edge[0]])
            for cur_node in input_nodes:
                del self.nodes[cur_node.id]

        final_answer = self.node_out.generated_answer
        return final_answer

    def get_input_layer(self):
        """Get the input layer of the graph."""
        input_layer = []
        for node in self.nodes.values():
            if len(node.predecessors) == 0:
                input_layer.append(node)
        return input_layer

    def get_output_layer(self):
        """Get the output layer of the graph."""
        output_layer = []
        for node in self.nodes.values():
            if len(node.successors) == 0:
                output_layer.append(node)
        return output_layer
    
    #Check if the graph has a circular dependency.
    def circular_check(self):
        visited = set()
        path = set()
        def dfs(n):
            if n in path: return True
            if n in visited: return False
            visited.add(n)
            path.add(n)
            if any(dfs(s) for s in n.successors): return True
            path.remove(n)
            return False
        return any(n not in visited and dfs(n) for n in self.nodes.values())
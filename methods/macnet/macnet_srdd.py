import os
import re
import copy
import math
import time
import random
import shutil
import signal
import subprocess

from termcolor import colored
from typing import Tuple

from .prompt_srdd import *
from ..mas_base import MAS
from ..utils import load_config

class Node:
    def __init__(self, node_id: int, env,temperature: float = 0.2) -> None:
        """Initialize a Node."""
        self.id: int = node_id
        self.predecessors: list[Node] = []
        self.successors: list[Node] = []
        self.suggestions =None
        self.pre_solutions= {}
        self.solution = {}
        self.generated_content: str 
        self.temperature: float = temperature
        self.system_message: str = " "
        self.pool = None
        self.depth: int = 0
        self.env=env

    #Formatting the contents of a code file into a Markdown-style code block document
    def _get_codes(self) -> str:
        content = ""
        for filename in self.solution.keys():
            content += "{}\n```{}\n{}\n```\n\n".format(filename,
                                                       "python" if filename.endswith(".py") else filename.split(".")[
                                                           -1], self.solution[filename])
        return content
    
    #Intelligently extract code snippets from structured text and index code repositories
    def code_process(self,generated_content):
        def extract_filename_from_line(lines):
            file_name = ""
            for candidate in re.finditer(r"(\w+\.\w+)", lines, re.DOTALL):
                file_name = candidate.group()
                file_name = file_name.lower()
            return file_name

        def extract_filename_from_code(code):
            file_name = ""
            regex_extract = r"class (\S+?):\n"
            matches_extract = re.finditer(regex_extract, code, re.DOTALL)
            for match_extract in matches_extract:
                file_name = match_extract.group(1)
            file_name = file_name.lower().split("(")[0] + ".py"
            return file_name

        if generated_content != "":
            regex = r"(.+?)\n```.*?\n(.*?)```"
            matches = re.finditer(regex, generated_content, re.DOTALL)
            for match in matches:
                code = match.group(2)
                if "CODE" in code:
                    continue
                group1 = match.group(1)
                filename = extract_filename_from_line(group1)
                if "__main__" in code:
                    filename = "main.py"
                if filename == "":  # post-processing
                    filename = extract_filename_from_code(code)
                assert filename != "" and filename != ".py", print(group1, generated_content)
                if filename is not None and code is not None and len(filename) > 0 and len(code) > 0:
                    self.solution[filename] = "\n".join([line for line in code.split("\n") if len(line.strip()) > 0])

    def write_codes_to_hardware(self, name):
        directory=f"results/{self.env.dataset_name}/{self.env.model_name}/MacNet/{name}"
        if not os.path.exists(f"{directory}"):
            os.mkdir(f"{directory}")
        for filename in self.solution.keys():
            filepath = os.path.join(directory, filename)
            with open(filepath, "w", encoding="utf-8") as writer:
                writer.write(self.solution[filename])
        files = os.listdir(directory)
        for file in files:
            if file not in self.solution.keys():
                shutil.rmtree(os.path.join(directory, file)) if os.path.isdir(os.path.join(directory, file)) else os.remove(os.path.join(directory, file))

    def exist_bugs(self, directory: str) -> Tuple[bool, str]:
        #Check if there are bugs in the software.
        success_info = "The software run successfully without errors."
        try:
            if os.name == 'nt':
                command = f"cd {directory} && dir && python main.py"
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                command = f"cd {directory} && python3 main.py;"
                process = subprocess.Popen(command,
                                        shell=True,
                                        preexec_fn=os.setsid,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE
                                        )
            time.sleep(3)
            return_code = process.returncode
            if process.poll() is None:
                if "killpg" in dir(os):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    os.kill(process.pid, signal.SIGTERM)
                    if process.poll() is None:
                        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
            if return_code == 0:
                return False, success_info
            else:
                error_output = process.stderr.read().decode('utf-8')
                if error_output:
                    if "Traceback".lower() in error_output.lower():
                        errs = error_output.replace(directory + "/", "")
                        return True, errs
                else:
                    return False, success_info
        except subprocess.CalledProcessError as e:
            return True, f"Error: {e}"
        except Exception as ex:
            return True, f"An error occurred: {ex}"
        return False, success_info

    def interact(self, query: str, pre_solution: str, name: str):
        """Optimize a single solution."""
        success_info = "The software run successfully without errors."
        error_info = "The software run failed with errors."
        self.suggestions = "None."

        if pre_solution != '':
            instructor_prompt = INSTRUCTOR_PROMPT.format(query=query, pre_solution=pre_solution)
            self.suggestions = self.env.call_llm(prompt=instructor_prompt,system_prompt=self.system_message,temperature=self.temperature)
            if self.suggestions.startswith("<API>compile()</API>"):
                self.code_process(pre_solution)
                self.write_codes_to_hardware(name)
                dir_name = f"./results/{self.env.dataset_name}/{self.env.model_name}/MacNet/{name}"
                compiler_flag, compile_info = self.exist_bugs(dir_name)

                if not compiler_flag:
                    self.suggestions = success_info + "\n" + self.suggestions
                else:
                    self.suggestions = error_info + "\n" + compile_info + "\n" + self.suggestions
                    instructor_prompt = "Compiler's feedback: " + error_info + "\n" + compile_info + \
                                        "pre_comments:" + self.suggestions + "\n" + instructor_prompt
                    self.suggestions = self.env.call_llm(prompt=instructor_prompt,system_prompt=self.system_message,temperature=self.temperature)
                    self.suggestions = error_info + "\n" + compile_info + "\n" + self.suggestions

        assistant_prompt = ASSISTANT_PROMPT.format(query=query, pre_solution=pre_solution,suggestions=self.suggestions)
        response=self.env.call_llm(prompt=assistant_prompt,system_prompt=self.system_message,temperature=self.temperature)
        response = response.replace("```", "\n```").replace("'''", "\n'''")
        newnode=Node(node_id=-100,env=None)
        newnode.code_process(response)
        return response,newnode
    
    def aggregate(self, query: str, retry_limit: int, unit_num: int, layer_directory: str, graph_depth: int,
                  store_dir: str) -> int:
        """Aggregate solutions from predecessors."""

        cc_prompt = self.system_message + CC_PROMPT

        for file in self.pre_solutions:
            with open(layer_directory + "/solution_{}.txt".format(file), "w") as f:
                for key in self.pre_solutions[file].keys():
                    f.write(str(key) + '\n\n' + self.pre_solutions[file][key] + '\n\n')
        from .pool import Pool
        pool=Pool(unit_num, self.env)
        for i in range(retry_limit):
            new_codes = pool.state_pool_add(cc_prompt,query,store_dir,temperature=1 - self.depth / graph_depth,)
            if new_codes is None:
                print(f"Retry Aggregation at round {i}!")
            else:
                self.solution = new_codes.solution
                return 0
        # print(colored(f"ERROR: Node {self.id} has reached the retry limit!\n",'red'))
        return 1
    
class MacNet_SRDD(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.dataset_name=general_config["test_dataset_name"]
        self.model_name= self.model_dict_list[0]['model_name']
        self.agent_num = self.method_config["agent_num"]
        self.topology = self.method_config["topology"]
        self.reverse = self.method_config["reverse"]
        self.type = self.method_config["type"] if self.method_config["type"]!='default' else "default"
        self.aggregate_retry_limit=self.method_config["aggregate_retry_limit"]
        self.aggregate_unit_num=self.method_config["aggregate_unit_num"]
        self.node_in_id=self.method_config["node_in_id"]
        self.node_out_id=self.method_config["node_out_id"]
        self.edges = []
        self.node_in = Node(node_id=self.node_in_id,env=self)
        self.node_out = Node(node_id=self.node_out_id,env=self)
        self.nodes = {self.node_in_id: self.node_in, self.node_out_id: self.node_out}
        self.depth=None
        self.input_layer = []
        self.output_layer = []
     

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
                
    def inference(self, sample):
        query = sample['query']
        random.seed(2025)
        topo = self.generate_graph_topology()
        self.graph_inference(query, topo)
        return {"response": None}
    
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
            self.edges=[(edge[1], edge[0]) for edge in self.edges]
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

    def graph_inference(self,query,topo):
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
                            self.nodes[node.id].system_message = "You are an experienced programmer."
                        else:
                            profile_num = random.randint(1, 99)
                            try:
                                prompt=SYSTEM_PROMPT.get(self.type, [])[profile_num]
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

        #3. The graph starts executing 
        layer = 0
        
        directory = f"./results/{self.dataset_name}/{self.model_name}/MacNet"
        while True:
            input_nodes = self.get_input_layer()
            if len(input_nodes) == 0:
                break
            cur_layer_dir = directory + f"/Layer {layer}"
            os.makedirs(cur_layer_dir, exist_ok=True)
            if layer == 0 and not os.path.exists(cur_layer_dir + "/Node -1"):
                os.makedirs(cur_layer_dir + "/Node -1")
                with open(cur_layer_dir + "/Node -1/solution.txt", "w", encoding="utf-8") as f:
                    f.write(query)

            visited_edges, next_nodes = set(), set()
            for cur_node in input_nodes :
                with open(cur_layer_dir + f"/Node {cur_node.id}/profile.txt", "w", encoding="utf-8") as f:
                    f.write(cur_node.system_message)

                for next_node in cur_node.successors:
                    # Gather predecessors' solutions and interact 
                    response,newnode = next_node.interact(query=query,pre_solution=cur_node._get_codes(),name=self.type)
                    # Record the results of the interaction
                    next_node.pre_solutions[cur_node.id] = newnode.solution
                    with open(cur_layer_dir + f"/Node {cur_node.id}/suggestions.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n\n{next_node.id}'s suggestion on {cur_node.id}'s solution:\n{next_node.suggestions}\n\n")
                    visited_edges.add((cur_node.id, next_node.id))
                    next_nodes.add(next_node.id)

            for node_id in next_nodes:
                node = self.nodes[node_id]
                node_directory = directory + f"/Layer {layer + 1}/Node {node.id}"
                os.makedirs(node_directory, exist_ok=True)
                os.makedirs(node_directory + "/pre_solutions", exist_ok=True)
                for prev_node in node.pre_solutions.keys():
                    with open(node_directory + f"/pre_solutions/solution_{prev_node}.txt", "w") as f:
                        for key in node.pre_solutions[prev_node].keys():
                            f.write(f"{key}\n\n{node.pre_solutions[prev_node][key]}\n\n")

                if len(os.listdir(node_directory + "/pre_solutions")) != len(node.pre_solutions):
                    print(colored("Error: the number of solutions is not equal to the number of files!",'red'))
                    exit(1)
                # If the aggregation condition is satisfied, the aggregation algorithm is executed and all pre-solutions are merged
                if len(node.pre_solutions) == len(node.predecessors) and len(node.pre_solutions) >= self.aggregate_unit_num:
                    
                    agg_layer_dir = node_directory + "/pre_solutions"
                    error_flag = node.aggregate(query, self.aggregate_retry_limit, self.aggregate_unit_num,
                                                agg_layer_dir, self.depth, node_directory + "/solution.txt")
                    
                    if error_flag:
                        node.solution = node.pre_solutions[list(node.pre_solutions.keys())[0]]
                        with open(node_directory + "/solution.txt", "w") as f:
                            for key in node.solution.keys():
                                f.write(f"{key}\n\n{node.solution[key]}\n\n")
                        print(f"Node {node.id} failed aggregating pre_solutions.")
                #If not, the node will use the first pre_solution as its solution
                else:
                    node.solution = node.pre_solutions[list(node.pre_solutions.keys())[0]]
                    with open(node_directory + "/solution.txt", "w") as f:
                        for key in node.solution.keys():
                            f.write(f"{key}\n\n{node.solution[key]}\n\n")
                    # print(colored(f"Node {node.id} has insufficient predecessors, uses pre_solution.","yellow"))

            #Remove edges and nodes from the previous layer
            for edge in visited_edges:
                self.nodes[edge[0]].successors.remove(self.nodes[edge[1]])
                self.nodes[edge[1]].predecessors.remove(self.nodes[edge[0]])
            for cur_node in input_nodes:
                del self.nodes[cur_node.id]

            layer += 1
        self.node_out.write_codes_to_hardware(self.type)

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
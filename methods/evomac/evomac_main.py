import os
import time
import signal
import tempfile
import subprocess
from collections import defaultdict, deque

from methods.mas_base import MAS
from methods.evomac.codes import Codes
from methods.evomac.graph import Organization
from methods.evomac.prompt import INITIAL_CODING_ROLE, INITIAL_CODING, ORGANIZER, ORGANIZING, PROGRAMMER, SUBCODECOMPLETE, TESTORGANIZING, TESTCODECOMPLETE, UPDATING

class EvoMAC_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.iteration = self.method_config['iteration']
        self.language = self.method_config["language"]

        self.codes: Codes = Codes()
        self.test_codes: Codes = Codes()
        self.organization: Organization = Organization()
        self.test_organization: Organization = Organization(predifined_filename="test_organization.json")

    def inference(self, sample):
        
        query = sample['query']

        # 1. Initial coding
        initial_coding = self.format_messages(INITIAL_CODING_ROLE, INITIAL_CODING.format(task=query))
        initial_coding_codes = self.call_llm(messages=initial_coding)
        self.codes._update_codes(initial_coding_codes)
        
        # 2. Generate the workflow for the quey
        organizing = self.format_messages(ORGANIZER.format(task=query), ORGANIZING.format(task=query, language=self.language, codes=self.codes._get_codes()))
        organization = self.call_llm(messages=organizing)
        self.organization._update_orgs(organization)

        # 3. Generate and validate the generated code
        self.excute_workflow(query)
        has_bug_in_tests, test_reports = self.excute_test_workflow(query)
        
        # 4. Update the workflow if bugs are found iteratively
        for i in range(self.iteration-1):
            if not has_bug_in_tests:
                return {'response': self.codes._get_raw_codes()}
            else:
                updating = self.format_messages(ORGANIZER.format(task=query), UPDATING.format(task=query, codes=self.codes._get_codes(), issues=test_reports))
                organization = self.call_llm(messages=updating)
                self.organization._update_orgs(organization)

                self.excute_workflow(query)
                has_bug_in_tests, test_reports = self.excute_test_workflow(query)

        return {'response': self.codes._get_raw_codes()}

    def excute_workflow(self, query):
        """
        Complete the subtask in the genrated workflow to generate code for the problem.

        Args:
            query (str): The task or problem description to be solved.
        """
        org = self.organization._get_orgs()[0]
        composition, workflow = org.get('composition',{}), org.get('workflow',{})
        topo_order = self.topological_sort(workflow)
        for phase in topo_order:
            # Generate code to complete the subtask
            subtask = composition[phase]
            subtask_messages = self.format_messages(PROGRAMMER.format(task=query), SUBCODECOMPLETE.format(task=query, language=self.language, codes=self.codes._get_codes(), subtask=subtask))
            subtask_codes = self.call_llm(messages=subtask_messages)
            # print(f"[Programmer Prompt]:\n{self.format_print(subtask_messages)}\n\n")
            # print(f"[Programmer Response]:\n{subtask_codes}\n\n************************\n\n")
            self.codes._update_codes(subtask_codes)
        
    def excute_test_workflow(self, query):
        """
        Generate and complete the test workflow to validate the generated code.

        Args:
            query (str): The task or problem description to be tested.

        Returns:
            tuple: A tuple containing:
                - has_bug_in_tests (bool): True if bugs are found, False otherwise
                - test_reports (str): Detailed report of test results
        """
        # 1. Generate the workflow for test cases
        test_organizing = self.format_messages(ORGANIZER.format(task=query), TESTORGANIZING.format(task=query, language=self.language))
        test_organization = self.call_llm(messages=test_organizing)
        # print(f"[Test Organizer Prompt]:\n{self.format_print(test_organizing)}\n\n")
        # print(f"[Test Organizer Response]:\n{test_organization}\n\n************************\n\n")
        self.test_organization._update_orgs(test_organization)

        test_org = self.test_organization._get_orgs()[0]
        test_composition, test_workflow = test_org.get('composition',{}), test_org.get('workflow',{})
        test_topo_order = self.topological_sort(test_workflow)
        test_reports = ""
        has_bug_in_tests = False
        for idx, phase in enumerate(test_topo_order):
            # 2. Generate test cases for the subtask
            test_subtask = test_composition[phase]
            test_subtask_messages = self.format_messages(PROGRAMMER.format(task=query), TESTCODECOMPLETE.format(task=query, codes=self.codes._get_codes(), subtask=test_subtask, test_file_name=f'test_requirement_{idx}.py'))
            test_subtask_codes = self.call_llm(messages=test_subtask_messages)
            # print(f"[Test Programmer Prompt]:\n{self.format_print(test_subtask_messages)}\n\n")
            # print(f"[Test Programmer Response]:\n{test_subtask_codes}\n\n************************\n\n")
            self.test_codes._update_codes(test_subtask_codes, target_file=f'test_requirement_{idx}.py')

            # 3. Run the test cases and collect the results
            has_bug, result = self.test_bugs(test_file_name=f'test_requirement_{idx}.py')
            if has_bug:
                has_bug_in_tests = True
                test_reports += f'Requirement: {test_subtask}\nTest Report: {result}\n'

        if not has_bug_in_tests:
            test_reports = "No bugs found in the test cases."

        # print(f"[Test Reports]:\n{test_reports}\n\n************************\n\n")
        return has_bug_in_tests, test_reports
    
    def test_bugs(self, test_file_name):
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename, code in self.codes.codebooks.items():
                code_path = os.path.join(temp_dir, filename)
                with open(code_path, "w") as f:
                    f.write(code)

            test_path = os.path.join(temp_dir, test_file_name)
            with open(test_path, "w") as f:
                f.write(self.test_codes.codebooks[test_file_name])
            has_bug, result = self._test_bugs(temp_dir, test_file_name)
        return has_bug, result
    
    
    def _test_bugs(self, directory, test_file):
        success_info = "The codes run successfully without errors."
        try:
            # check if we are on windows or linux
            if os.name == 'nt':
                command = "cd {} && dir && python {}".format(directory, test_file)
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                command = "cd {}; ls -l; python3 {};".format(directory, test_file)
                process = subprocess.Popen(command,
                                        shell=True,
                                        preexec_fn=os.setsid,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE
                                        )
            time.sleep(3)
            return_code = process.returncode
            # Check if the software is still running
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
    
    def format_messages(self, role, content):
        return [
            { "role": "system", "content": role },
            { "role": "user", "content": content }
        ]

    def format_print(self, messages):
        print_str = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            print_str += f"**{role}**\n{content}\n-----------\n"
        return print_str

    def topological_sort(self, workflow):
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)
        for node, dependencies in workflow.items():
            for dep in dependencies:
                adj_list[dep].append(node)
                in_degree[node] += 1

        queue = deque([node for node in workflow if in_degree[node] == 0])
        topo_order = []
        while queue:
            current_node = queue.popleft()
            topo_order.append(current_node)

            # Reduce the in-degree of each neighbor by 1
            for neighbor in adj_list[current_node]:
                in_degree[neighbor] -= 1
                # If in-degree becomes 0, add it to the queue
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if there was a cycle in the graph
        if len(topo_order) != len(workflow):
            raise ValueError("The workflow contains a cycle and cannot be topologically sorted.")

        return topo_order


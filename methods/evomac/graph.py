import re
import os
import time
import json
import ast


class Organization():
    def __init__(self, generated_content = "", parse = True, predifined_filename = None):
        self.directory: str = None
        self.generated_content = generated_content
        self.organization = {}
        self.filename = "organization.json" if predifined_filename is None else predifined_filename

        if generated_content != "":
            if parse:
                content = self._format_orgs(self.generated_content)
            else:
                content = self.generated_content
            self.organization[self.filename] = content
            
    def _format_orgs(self, generated_content):
        regex = r"(.+?)\n```.*?\n(.*?)```"
        matches = re.finditer(regex, generated_content, re.DOTALL)
        composition = {}
        workflow = {}
        for match in matches:
            content = match.group(2).strip()
            if "COMPOSITION" in match.group(1):
                composition = self._format_composition(content)
            elif "WORKFLOW" in match.group(1):
                workflow = self._format_workflow(content)
        organization = {"composition": composition, "workflow": workflow}
        return organization

    def _update_orgs(self, generated_content, parse = True, predifined_filename = ""):
        new_orgs = Organization(generated_content, parse, self.filename)
        for key in new_orgs.organization.keys():
            if key not in self.organization.keys() or self.organization[key] != new_orgs.organization[key]:
                # print("{} updated.".format(key))
                # print("------Old:\n{}\n------New:\n{}".format(self.organization[key] if key in self.organization.keys() else "# None", new_orgs.organization[key]))
                self.organization[key] = new_orgs.organization[key]

    def _rewrite_orgs(self):
        directory = self.directory
        if not os.path.exists(directory):
            os.mkdir(directory)
            # print("{} Created.".format(directory))
        for filename in self.organization.keys():
            with open(os.path.join(directory, filename), "w", encoding="utf-8") as writer:
                json.dump(self.organization[filename], writer, indent=4)
                # print(os.path.join(directory, filename), "Writen")

    def _get_orgs(self):
        return list(self.organization.values())
    
    def _format_composition(self, generated_content):
        '''
        Input:
            composition: str, the composition part of the organization
        Output:
            composition: dict, the formatted composition part of the organization
        '''
        if "Programmer" in generated_content:
            if '\n\n' in generated_content:
                regex = r'Programmer \d+:.*?(\n\n|\Z)'
            else:
                regex = r'Programmer \d+:.*?(\n|\Z)'
        else:
            if '\n\n' in generated_content:
                regex = r'Task \d+:.*?(\n\n|\Z)'
            else:
                regex = r'Task \d+:.*?(\n|\Z)'
        matches = re.finditer(regex, generated_content, re.DOTALL)
        composition = {}
        for match in matches:
            task_detail = match.group().strip().split(":")
            task = task_detail[0].strip()
            if len(task_detail) > 2:
                task_info = ' '.join(task_detail[1:]).strip()
            else:
                task_info = task_detail[1].strip()
            composition.update({task.replace('Task', 'Programmer').strip(): task_info.strip()})
        return composition

    def _format_workflow(self, generated_content):
        '''
        Input:
            workflow: str, the workflow part of the organization
        Output:
            workflow: dict, the formatted workflow part of the organization
        '''
        if "Programmer" in generated_content:
            if '\n\n' in generated_content:
                regex = r'Programmer \d+:.*?(\n\n|\Z)'
            else:
                regex = r'Programmer \d+:.*?(\n|\Z)'
        else:
            if '\n\n' in generated_content:
                regex = r'Task \d+:.*?(\n\n|\Z)'
            else:
                regex = r'Task \d+:.*?(\n|\Z)'
        matches = re.finditer(regex, generated_content, re.DOTALL)
        workflow = {}
        programmers = []
        for match in matches:
            task_detail = match.group().strip().split(":")
            task = task_detail[0].strip()
            if len(task_detail) > 2:
                task_dependency = ' '.join(task_detail[1:]).strip()
            else:
                task_dependency = task_detail[1].strip()
            task_dependency = task_dependency.strip().strip('[]').replace('\'','').replace('\"','')
            task_dependency = task_dependency.split(',') if len(task_dependency) > 0 else []
            programmer = task.replace('Task', 'Programmer').strip()
            programmers_dependency = [x.replace('Task', 'Programmer').strip() for x in task_dependency] if len(task_dependency) > 0 else []
            workflow[programmer] = programmers_dependency
            programmers.extend(programmers_dependency)
        programmers = list(set(programmers))
        for programmer in programmers:
            if programmer not in workflow.keys():
                workflow[programmer] = []
        return workflow

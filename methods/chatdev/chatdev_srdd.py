import os
import re
import subprocess
import signal
import time
from datetime import datetime
import wikipediaapi  # pip install Wikipedia-API
import difflib
from ..mas_base import MAS
from .prompt_srdd import Role_configs, Phase_configs, background_prompt

from typing import List, Dict

class ChatDev_SRDD(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_srdd" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        
        self.version = 1.0
        self.gui_design = self.method_config['gui_design']
        self.history = []
        output_dir = f"./results/{general_config['test_dataset_name']}/{general_config['model_name']}/{general_config['method_name']}"
        self.env_dict = {
            "directory": output_dir,
            "task_prompt": "",
            "task_description":"",
            "modality": "",
            "ideas": "",
            "language": "",
            "pyfiles": [],
            "codes": {},
            "review_comments": "",
            "error_summary": "",
            "test_reports": "",
            "docs": {},
        }
        self.role_configs = Role_configs
        self.phase_configs = Phase_configs
        self.background_prompt = background_prompt
        
    
    def inference(self, sample):
        
        query = sample['query']
        ## Demand Analysis
        self.env_dict["task_prompt"] = query
        query_prefix = query[:10].replace(" ", "_")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{query_prefix}_{current_time}"
        self.env_dict["directory"] = os.path.join(self.env_dict["directory"], folder_name)
        Chief_Executive_Officer_system_message = "\n".join(self.role_configs["Chief Executive Officer"]).format(chatdev_prompt = self.background_prompt,
                                                                                                              task = query)
        Chief_Product_Officer_system_message = "\n".join(self.role_configs["Chief Product Officer"]).format(chatdev_prompt = self.background_prompt,
                                                                                                              task = query)
        self.env_dict["modality"] = self.DemandAnalysis(Chief_Executive_Officer_system_message, Chief_Product_Officer_system_message)
        # self.env_dict["task_description"] = self.modal_trans(query)
        
        ## Choose language
        Chief_Technology_Officer_system_message = "\n".join(self.role_configs["Chief Technology Officer"]).format(chatdev_prompt = self.background_prompt,
                                                                                                              task = query)
        self.env_dict["language"] = self.LanguageChoose(Chief_Executive_Officer_system_message, Chief_Technology_Officer_system_message)
        ## Coding
        Programmer_system_message = "\n".join(self.role_configs["Programmer"]).format(chatdev_prompt = self.background_prompt,
                                                                                                              task = query)
        self.env_dict["codes"] = self.Coding(Chief_Technology_Officer_system_message, Programmer_system_message)
        self.rewrite_codes()
        self.env_dict["pyfiles"] = [filename for filename in os.listdir(self.env_dict['directory']) if filename.endswith(".py")]

        ## CodingCompleteAll
        self.CodeComplete(Chief_Technology_Officer_system_message, Programmer_system_message)
        
        ## CodeReview
        Code_Reviewer_system_message = "\n".join(self.role_configs["Code Reviewer"]).format(chatdev_prompt = self.background_prompt,
                                                                                                              task = query)
        final_response = self.CodeReview(Programmer_system_message, Code_Reviewer_system_message)
        
        ##Test
        if self.gui_design:
            Software_Test_engineer_system_message = "\n".join(self.role_configs["Software Test Engineer"]).format(chatdev_prompt = self.background_prompt,
                                                                                                                    task = query)
            self.Test(Software_Test_engineer_system_message, Programmer_system_message)

        ##EnvironmentDoc
        self.EnvironmentDoc(Chief_Technology_Officer_system_message,Programmer_system_message)
        ##Manual
        self.Manual(Chief_Executive_Officer_system_message, Chief_Product_Officer_system_message)
        
        return {"response": final_response}

    def DemandAnalysis(self, Chief_Executive_Officer_system_message, Chief_Product_Officer_system_message , max_turn_step = 10):
        Chief_Executive_Officer_history = []  ##user
        Chief_Product_Officer_history = []  ##assistant
        
        ##Chief_Executive_Officer to Chief_Product_Officer
        phase_prompt = "\n\n".join(self.phase_configs["DemandAnalysis"]["phase_prompt"]).format(assistant_role = "Chief Product Officer")
        Chief_Product_Officer_messages = self.construct_messages(Chief_Product_Officer_system_message, Chief_Product_Officer_history, phase_prompt)
        Chief_Product_Officer_response = self.call_llm(None, None, Chief_Product_Officer_messages)
        Chief_Executive_Officer_history.append({"role": "assistant", "content": phase_prompt})
        Chief_Product_Officer_history.append({"role": "user", "content": phase_prompt})
        self.history.append({"role": "Chief_Executive_Officer", "content": phase_prompt})
        if Chief_Product_Officer_response.split("\n")[-1].startswith("<INFO>"):
            return Chief_Product_Officer_response.split("<INFO>")[-1].lower().replace(".", "").strip()
        max_turn_step -= 1
        while max_turn_step > 0:  
            ##Chief_Product_Officer to Chief_Executive_Officer
            Chief_Executive_Officer_messages = self.construct_messages(Chief_Executive_Officer_system_message, Chief_Executive_Officer_history, Chief_Product_Officer_response)
            Chief_Executive_Officer_response = self.call_llm(None, None, Chief_Executive_Officer_messages)
            Chief_Product_Officer_history.append({"role": "assistant", "content": Chief_Product_Officer_response})
            Chief_Executive_Officer_history.append({"role": "user", "content": Chief_Product_Officer_response})
            self.history.append({"role": "Chief_Product_Officer", "content": Chief_Product_Officer_response})
            if Chief_Executive_Officer_response.split("\n")[-1].startswith("<INFO>"):
                return Chief_Executive_Officer_response.split("<INFO>")[-1].lower().replace(".", "").strip()
            max_turn_step -= 1
            if max_turn_step > 0:
                Chief_Product_Officer_messages = self.construct_messages(Chief_Product_Officer_system_message, Chief_Product_Officer_history, Chief_Executive_Officer_response)
                Chief_Product_Officer_response = self.call_llm(None, None, Chief_Product_Officer_messages)
                Chief_Executive_Officer_history.append({"role": "assistant", "content": Chief_Executive_Officer_response})
                Chief_Product_Officer_history.append({"role": "user", "content": Chief_Executive_Officer_response})
                self.history.append({"role": "Chief_Executive_Officer", "content": Chief_Executive_Officer_response})
                if Chief_Product_Officer_response.split("\n")[-1].startswith("<INFO>"):
                    return Chief_Product_Officer_response.split("<INFO>")[-1].lower().replace(".", "").strip()
                max_turn_step -= 1
        
        return Chief_Executive_Officer_response

    def LanguageChoose(self, Chief_Executive_Officer_system_message, Chief_Technology_Officer_system_message, max_turn_step = 10):
        Chief_Executive_Officer_history = []
        Chief_Technology_Officer_history = []
        ##Chief_Executive_Officer to Chief_Technology_Officer
        phase_prompt = "\n\n".join(self.phase_configs["LanguageChoose"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                                modality = self.env_dict["modality"],
                                                                                                ideas = self.env_dict["ideas"],
                                                                                                assistant_role = "Chief Technology Officer")
        Chief_Technology_Officer_messages = self.construct_messages(Chief_Technology_Officer_system_message, Chief_Technology_Officer_history, phase_prompt)
        Chief_Technology_Officer_response = self.call_llm(None, None, Chief_Technology_Officer_messages)
        Chief_Executive_Officer_history.append({"role": "assistant", "content": phase_prompt})
        Chief_Technology_Officer_history.append({"role": "user", "content": phase_prompt})
        self.history.append({"role": "Chief_Executive_Officer", "content": phase_prompt})
        if Chief_Technology_Officer_response.split("\n")[-1].startswith("<INFO>"):
            # return Chief_Technology_Officer_response.split("<INFO>")[-1].lower().replace(".", "").strip()
            return Chief_Technology_Officer_response.split("<INFO>")[-1]
        max_turn_step -= 1
        while max_turn_step > 0:
            ##Chief_Technology_Officer to Chief_Executive_Officer
            Chief_Executive_Officer_messages = self.construct_messages(Chief_Executive_Officer_system_message, Chief_Executive_Officer_history, Chief_Technology_Officer_response)
            Chief_Executive_Officer_response = self.call_llm(None, None, Chief_Executive_Officer_messages)
            Chief_Technology_Officer_history.append({"role": "assistant", "content": Chief_Technology_Officer_response})
            Chief_Executive_Officer_history.append({"role": "user", "content": Chief_Technology_Officer_response})
            self.history.append({"role": "Chief_Technology_Officer", "content": Chief_Technology_Officer_response})
            if Chief_Executive_Officer_response.split("\n")[-1].startswith("<INFO>"):
                # return Chief_Executive_Officer_response.split("<INFO>")[-1].lower().replace(".", "").strip()
                return Chief_Executive_Officer_response.split("<INFO>")[-1]
            max_turn_step -= 1
            if max_turn_step > 0:
                Chief_Technology_Officer_messages = self.construct_messages(Chief_Technology_Officer_system_message, Chief_Technology_Officer_history, Chief_Executive_Officer_response)
                Chief_Technology_Officer_response = self.call_llm(None, None, Chief_Technology_Officer_messages)
                Chief_Executive_Officer_history.append({"role": "assistant", "content": Chief_Executive_Officer_response})
                Chief_Technology_Officer_history.append({"role": "user", "content": Chief_Executive_Officer_response})
                self.history.append({"role": "Chief_Executive_Officer", "content": Chief_Executive_Officer_response})
                if Chief_Technology_Officer_response.split("\n")[-1].startswith("<INFO>"):
                    # return Chief_Technology_Officer_response.split("<INFO>")[-1].lower().replace(".", "").strip()
                    return Chief_Technology_Officer_response.split("<INFO>")[-1]
                max_turn_step -= 1
            
        return Chief_Executive_Officer_response

    def Coding(self, Chief_Technology_Officer_system_message, Programmer_system_message, max_turn_step = 1):
        Chief_Technology_Officer_history = []
        Programmer_history = []
        ##Chief_Technology_Officer to Programmer
        phase_prompt = "\n\n".join(self.phase_configs["Coding"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                        description = self.env_dict["task_description"],
                                                                                        modality = self.env_dict["modality"],
                                                                                        language = self.env_dict["language"],
                                                                                        ideas = self.env_dict["ideas"],
                                                                                        assistant_role = "Programmer",
                                                                                        gui = "" if not self.gui_design \
                                                                                            else "The software should be equipped with graphical user interface (GUI) so that user can visually and graphically use it; so you must choose a GUI framework (e.g., in Python, you can implement GUI via tkinter, Pygame, Flexx, PyGUI, etc,).")
        Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, phase_prompt)
        Programmer_response = self.call_llm(None, None, Programmer_messages)
        Chief_Technology_Officer_history.append({"role": "assistant", "content": phase_prompt})
        Programmer_history.append({"role": "user", "content": phase_prompt})
        self.history.append({"role": "Chief_Technology_Officer", "content": phase_prompt})
        codebooks = self.init_codes(Programmer_response)
        max_turn_step -= 1
        while max_turn_step > 0:
            ##Programmer to Chief_Technology_Officer
            Chief_Technology_Officer_messages = self.construct_messages(Chief_Technology_Officer_system_message, Chief_Technology_Officer_history, Programmer_response)
            Chief_Technology_Officer_response = self.call_llm(None, None, Chief_Technology_Officer_messages)
            Programmer_history.append({"role": "assistant", "content": Chief_Technology_Officer_response})
            Chief_Technology_Officer_history.append({"role": "user", "content": Chief_Technology_Officer_response})
            self.history.append({"role": "Programmer", "content": Chief_Technology_Officer_response})
            max_turn_step -= 1
            if max_turn_step > 0:
                ##Chief_Technology_Officer to Programmer
                Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, Chief_Technology_Officer_response)
                Programmer_response = self.call_llm(None, None, Programmer_messages)
                Chief_Technology_Officer_history.append({"role": "assistant", "content": Chief_Technology_Officer_response})
                Programmer_history.append({"role": "user", "content": Chief_Technology_Officer_response})
                self.history.append({"role": "Chief_Technology_Officer", "content": Chief_Technology_Officer_response})
                self.update_codes(Programmer_response)
                max_turn_step -= 1

        return codebooks

    def CodeComplete(self, Chief_Technology_Officer_system_message, Programmer_system_message, cycleNum = 10):
        ##Chief_Technology_Officer to Programmer
        # Chief_Technology_Officer_history = []
        Programmer_history = []
        for i in range(cycleNum):
            unimplemented_file = "" 
            for filename in self.env_dict["pyfiles"]:
                code_content = open(os.path.join(self.env_dict['directory'], filename)).read()
                lines = [line.strip() for line in code_content.split("\n") if line.strip() == "pass"]    
                if len(lines) > 0:
                    unimplemented_file = filename
                    break       
            if unimplemented_file == "":
                break
            phase_prompt = "\n\n".join(self.phase_configs["CodeCompleteAll"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                                     modality = self.env_dict["modality"],
                                                                                                     language = self.env_dict["language"],
                                                                                                     codes = self.get_codes(),
                                                                                                     unimplemented_file = unimplemented_file,
                                                                                                     assistant_role = "Programmer",
                                                                                                    )
            Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, phase_prompt)
            Programmer_response = self.call_llm(None, None, Programmer_messages)
            # Chief_Technology_Officer_history.append({"role": "assistant", "content": phase_prompt})
            Programmer_history.append({"role": "user", "content": phase_prompt})
            Programmer_history.append({"role": "assistant", "content": Programmer_response})
            self.history.append({"role": "Chief_Technology_Officer", "content": phase_prompt})
            self.history.append({"role": "Programmer", "content": Programmer_response})
            self.update_codes(Programmer_response)
            self.rewrite_codes()


    def CodeReview(self, Programmer_system_message, Code_Reviewer_system_message, cycleNum = 3):

        programmer_response = ""
        for i in range(cycleNum):
            Code_Reviewer_history = []
            Programmer_history = []
            ##CodeReviewComment Programmer to Code Reviewer
            CodeReviewComment_phase_prompt = "\n".join(self.phase_configs["CodeReviewComment"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                                                modality = self.env_dict["modality"],
                                                                                                                language = self.env_dict["language"],
                                                                                                                ideas = self.env_dict["ideas"],
                                                                                                                codes = self.get_codes(),
                                                                                                                assistant_role = "Code Reviewer",
                                                                                                                )
            Code_Reviewer_messages = self.construct_messages(Code_Reviewer_system_message, Code_Reviewer_history, CodeReviewComment_phase_prompt)
            Code_Reviewer_response = self.call_llm(None, None, Code_Reviewer_messages)
            Code_Reviewer_history.append({"role": "assistant", "content": CodeReviewComment_phase_prompt})
            Programmer_history.append({"role": "user", "content": CodeReviewComment_phase_prompt})
            self.history.append({"role": "Programmer", "content": CodeReviewComment_phase_prompt})
            self.env_dict["review_comments"] = Code_Reviewer_response.split("<INFO>")[-1]
            # if "<INFO> Finished".lower() in Code_Reviewer_response.lower():
            #     break
            ##CodeReviewModification Code Reviewer to Programmer
            Code_Reviewer_history = []
            Programmer_history = []
            CodeReviewModification_phase_prompt = "\n".join(self.phase_configs["CodeReviewModification"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                                                    modality = self.env_dict["modality"],
                                                                                                                    language = self.env_dict["language"],
                                                                                                                    ideas = self.env_dict["ideas"],
                                                                                                                    codes = self.get_codes(),
                                                                                                                    comments = self.env_dict["review_comments"],
                                                                                                                    assistant_role = "Programmer",
                                                                                                                    )
            Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, CodeReviewModification_phase_prompt)
            programmer_response = self.call_llm(None, None, Programmer_messages)
            Code_Reviewer_history.append({"role": "assistant", "content": CodeReviewModification_phase_prompt})
            Programmer_history.append({"role": "user", "content": CodeReviewModification_phase_prompt})
            self.history.append({"role": "Code_Reviewer", "content": CodeReviewModification_phase_prompt})
            self.update_codes(programmer_response)
            self.rewrite_codes()
            if "<INFO> Finished".lower() in programmer_response.lower():
                break
        if programmer_response == "":
            return Code_Reviewer_response
        return  programmer_response

    def Test(self, Software_Test_Engineer_system_message, Programmer_system_message, cycleNum = 3):
        ##Software_Test_Engineer to Programmer
        for i in range(cycleNum):
            Programmer_history = []
            bug_flag, self.env_dict["test_reports"] = self.exist_bugs()
            # print("bug_flag:", bug_flag)
            # print("test_reports:", self.env_dict["test_reports"])
            if "ModuleNotFoundError" in self.env_dict["test_reports"]:
                self.fix_module_not_found_error(self.env_dict["test_reports"])
                self.env_dict["error_summary"] = "nothing need to do"
            else:
                TestErrorSummary_phase_prompt = "\n\n".join(self.phase_configs["TestErrorSummary"]["phase_prompt"]).format(language = self.env_dict["language"],
                                                                                                                        codes = self.get_codes(),
                                                                                                                        test_reports = self.env_dict["test_reports"],
                                                                                                                        )
                Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, TestErrorSummary_phase_prompt)
                Programmer_response = self.call_llm(None, None, Programmer_messages)
                Programmer_history.append({"role": "user", "content": TestErrorSummary_phase_prompt})
                Programmer_history.append({"role": "assistant", "content": Programmer_response})
                self.history.append({"role": "Software_Test_Engineer", "content": TestErrorSummary_phase_prompt})
                self.history.append({"role": "Programmer", "content": Programmer_response})
                self.env_dict["error_summary"] = Programmer_response
                if not bug_flag:
                    # print(f"**[Test Info]**\n\nAI User (Software Test Engineer):\nTest Pass!\n")
                    break
            ##Software_Test_Engineer to Programmer
            Programmer_history = []
            TestModification_phase_prompt = "\n\n".join(self.phase_configs["TestModification"]["phase_prompt"]).format(language = self.env_dict["language"],
                                                                                                                    codes = self.get_codes(),
                                                                                                                    test_reports = self.env_dict["test_reports"],
                                                                                                                    error_summary = self.env_dict["error_summary"], 
                                                                                                                    assistant_role = "Programmer",)
            Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, TestModification_phase_prompt)
            Programmer_response = self.call_llm(None, None, Programmer_messages)
            Programmer_history.append({"role": "user", "content": TestModification_phase_prompt})
            Programmer_history.append({"role": "assistant", "content": Programmer_response})
            self.history.append({"role": "Software_Test_Engineer", "content": TestModification_phase_prompt})
            self.history.append({"role": "Programmer", "content": Programmer_response})
            self.update_codes(Programmer_response)
            self.rewrite_codes()

    def EnvironmentDoc(self, Chief_Technology_Officer_system_message, Programmer_system_message, max_turn_step = 1):
        Programmer_history = []
        Chief_Technology_Officer_history = []
        ## Chief_Technology_Officer to Programmer
        phase_prompt = "\n\n".join(self.phase_configs["EnvironmentDoc"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                                modality = self.env_dict["modality"],
                                                                                                language = self.env_dict["language"],
                                                                                                ideas = self.env_dict["ideas"],
                                                                                                codes = self.get_codes(),
                                                                                                assistant_role = "Programmer",
                                                                                                )
        for i in range(max_turn_step):
            Programmer_messages = self.construct_messages(Programmer_system_message, Programmer_history, phase_prompt)
            Programmer_response = self.call_llm(None, None, Programmer_messages)
            Chief_Technology_Officer_history.append({"role": "assistant", "content": phase_prompt})
            Programmer_history.append({"role": "user", "content": phase_prompt})
            self.history.append({"role": "Programmer", "content": phase_prompt})
            if "<INFO>" not in Programmer_response:
                Programmer_response = "<INFO>" + Programmer_response
            self.update_docs(Programmer_response)
            self.rewrite_docs()

    def Manual(self, Chief_Executive_Officer_system_message, Chief_Product_Officer_system_message, max_turn_step = 1):
        Chief_Executive_Officer_history = []
        Chief_Product_Officer_history = []
        ##Chief_Executive_Officer to Chief_Product_Officer
        phase_prompt = "\n\n".join(self.phase_configs["Manual"]["phase_prompt"]).format(task = self.env_dict["task_prompt"],
                                                                                        modality = self.env_dict["modality"],
                                                                                        language = self.env_dict["language"],
                                                                                        ideas = self.env_dict["ideas"],
                                                                                        codes = self.get_codes(),
                                                                                        requirements = self.env_dict["docs"]["requirements.txt"],
                                                                                        assistant_role = "Chief Product Officer",
                                                                                        )
        for i in range(max_turn_step):
            Chief_Product_Officer_messages = self.construct_messages(Chief_Product_Officer_system_message, Chief_Product_Officer_history, phase_prompt)
            Chief_Product_Officer_response = self.call_llm(None, None, Chief_Product_Officer_messages)
            Chief_Executive_Officer_history.append({"role": "assistant", "content": phase_prompt})
            Chief_Product_Officer_history.append({"role": "user", "content": phase_prompt})
            self.history.append({"role": "Chief_Executive_Officer", "content": phase_prompt})
            self.update_docs(Chief_Product_Officer_response, "manual.md")
            self.rewrite_docs()

    def modal_trans(self, query):
        try:
            task_in ="'" + query + \
                "'Just give me the most important keyword about this sentence without explaining it and your answer should be only one keyword."
            messages = [{"role": "user", "content": task_in}]
            response = self.call_llm(None, None, messages)
            spider_content = self.get_wiki_content(response)
            # time.sleep(1)
            task_in = "'" + spider_content + \
                "',Summarize this paragraph and return the key information."
            messages = [{"role": "user", "content": task_in}]
            response = self.call_llm(None, None, messages)
            # print("web spider content:", response)
        except:
            response = ''
            # print("the content is none")
        return response

    def init_codes(self, generated_content):
        codebooks = {}
        if generated_content != "":
            regex = r"(.+?)\n```.*?\n(.*?)```"
            matches = re.finditer(regex, generated_content, re.DOTALL)
            for match in matches:
                code = match.group(2)
                if "CODE" in code:
                    continue
                group1 = match.group(1)
                filename = self.extract_filename_from_line(group1)
                if "__main__" in code:
                    filename = "main.py"
                if filename == "":  # post-processing
                    filename = self.extract_filename_from_code(code)
                assert filename != ""
                if filename is not None and code is not None and len(filename) > 0 and len(code) > 0:
                    codebooks[filename] = self.format_code(code)
        return codebooks

    def extract_filename_from_line(self, lines):
        file_name = ""
        for candidate in re.finditer(r"(\w+\.\w+)", lines, re.DOTALL):
            file_name = candidate.group()
            file_name = file_name.lower()
        return file_name

    def extract_filename_from_code(self, code):
        file_name = ""
        regex_extract = r"class (\S+?):\n"
        matches_extract = re.finditer(regex_extract, code, re.DOTALL)
        for match_extract in matches_extract:
            file_name = match_extract.group(1)
        file_name = file_name.lower().split("(")[0] + ".py"
        return file_name

    def format_code(self, code):
        code = "\n".join([line for line in code.split("\n") if len(line.strip()) > 0])
        return code
    
    def get_codes(self, filename = None) -> str:
        content = ""
        if filename is None:
            for filename in self.env_dict["codes"].keys():
                content += "{}\n```{}\n{}\n```\n\n".format(filename,
                                                        "python" if filename.endswith(".py") else filename.split(".")[
                                                            -1], self.env_dict["codes"][filename])
        else:
            content = "{}\n```{}\n{}\n```\n\n".format(filename,
                                                    "python" if filename.endswith(".py") else filename.split(".")[
                                                        -1], self.env_dict["codes"][filename])
        return content

    def update_codes(self, generated_content):
        new_codes = self.init_codes(generated_content)
        differ = difflib.Differ()
        for key in new_codes.keys():
            if key not in self.env_dict["codes"].keys() or self.env_dict["codes"][key] != new_codes[key]:
                update_codes_content = "**[Update Codes]**\n\n"
                update_codes_content += "{} updated.\n".format(key)
                old_codes_content = self.env_dict["codes"][key] if key in self.env_dict["codes"].keys() else "# None"
                new_codes_content = new_codes[key]

                lines_old = old_codes_content.splitlines()
                lines_new = new_codes_content.splitlines()

                unified_diff = difflib.unified_diff(lines_old, lines_new, lineterm='', fromfile='Old', tofile='New')
                unified_diff = '\n'.join(unified_diff)
                update_codes_content = update_codes_content + "\n\n" + """```
'''

'''\n""" + unified_diff + "\n```"

                # print(f"update codes:", update_codes_content)
                self.env_dict["codes"][key] = new_codes[key]

    def rewrite_codes(self) -> None:
        directory = self.env_dict["directory"]
        rewrite_codes_content = "**[Rewrite Codes]**\n\n"
        if os.path.exists(directory) and len(os.listdir(directory)) > 0:
            self.version += 1.0
        if not os.path.exists(directory):
            os.makedirs(self.env_dict["directory"])
            rewrite_codes_content += "{} Created\n".format(directory)

        for filename in self.env_dict["codes"].keys():
            filepath = os.path.join(directory, filename)
            with open(filepath, "w", encoding="utf-8") as writer:
                writer.write(self.env_dict["codes"][filename])
                rewrite_codes_content += os.path.join(directory, filename) + " Wrote\n"
        # print(f"rewrite codes:", rewrite_codes_content)

    def update_docs(self, generated_content, predifined_filename = ""):
        docbooks = {}
        if predifined_filename == "":
            filename = "requirements.txt"
            regex = r"```\n(.*?)```"
            matches = re.finditer(regex, generated_content, re.DOTALL)
            for match in matches:
                doc = match.group(1)
                docbooks[filename] = doc
        else:
            filename = predifined_filename
            docbooks[filename] = generated_content

        for key in docbooks.keys():
            if key not in self.env_dict["docs"].keys() or self.env_dict["docs"][key] != docbooks[key]:
                # print("{} updated.".format(key))
                # print("------Old:\n{}\n------New:\n{}".format(self.env_dict["docs"][key] if key in self.env_dict["docs"].keys() else "# None", docbooks[key]))
                self.env_dict["docs"][key] = docbooks[key]

    def rewrite_docs(self):
        directory = self.env_dict["directory"]
        if not os.path.exists(directory):
            os.mkdir(directory)
            # print("{} Created.".format(directory))
        for filename in self.env_dict["docs"].keys():
            with open(os.path.join(directory, filename), "w", encoding="utf-8") as writer:
                writer.write(self.env_dict["docs"][filename])
                # print(os.path.join(directory, filename), "Writen")

    def exist_bugs(self):
        directory = self.env_dict['directory']

        success_info = "The software run successfully without errors."
        try:

            # check if we are on windows or linux
            if os.name == 'nt':
                command = "cd {} && dir && python main.py".format(directory)
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                command = "cd {}; ls -l; python3 main.py;".format(directory)
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
                # print("error_output:", error_output)
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

    def fix_module_not_found_error(test_reports):
        if "ModuleNotFoundError" in test_reports:
            for match in re.finditer(r"No module named '(\S+)'", test_reports, re.DOTALL):
                module = match.group(1)
                subprocess.Popen("pip install {}".format(module), shell=True).wait()
                # print("**[CMD Execute]**\n\n[CMD] pip install {}".format(module))

    def get_wiki_content(self, keyword):
        #  Wikipedia API ready
        wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        #the topic content which you want to spider
        search_topic = keyword
        # get the page content
        page_py = wiki_wiki.page(search_topic)
        # check the existence of the content in the page
        # if page_py.exists():
        #     print("Page - Title:", page_py.title)
        #     print("Page - Summary:", page_py.summary)
        # else:
        #     print("Page not found.")
        return page_py.summary

    def construct_messages(self, prepend_prompt: str, history: List[Dict], append_prompt: str):
        messages = []
        if prepend_prompt:
            messages.append({"role": "system", "content": prepend_prompt})
        if prepend_prompt == None:
            messages.append({"role": "system", "content": ""})
        if len(history) > 0:
            messages += history
        if append_prompt:
            messages.append({"role": "user", "content": append_prompt})
        if append_prompt == None:
            messages.append({"role": "user", "content": ""})
        return messages



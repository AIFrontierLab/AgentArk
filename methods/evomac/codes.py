import difflib
import re


class Codes:
    def __init__(self, generated_content="", target_file=None):
        self.directory: str = None
        self.version: float = 0.0
        self.generated_content: str = generated_content
        self.codebooks = {}

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
            if "```python" in generated_content:
                regex = r'(.*?)```python(.*?)```'
            elif "```FILENAME" in generated_content:
                regex = r"(.+?)\n```FILENAME.*?\n(.*?)```"
            matches = re.finditer(regex, self.generated_content, re.DOTALL)
            for match in matches:
                code = match.group(2)
                if "CODE" in code:
                    continue
                group1 = match.group(1)
                if target_file is None:
                    filename = extract_filename_from_line(group1)
                    if "__main__" in code:
                        if ("test" not in code) and ("check" not in code):
                            filename = "main.py"
                    if filename == "":  # post-processing
                        filename = extract_filename_from_code(code)
                else:
                    filename = target_file
                assert filename != ""
                if filename is not None and code is not None and len(filename) > 0 and len(code) > 0:
                    self.codebooks[filename] = self._format_code(code)

    def _format_code(self, code):
        # code = "\n".join([line for line in code.split("\n") if len(line.strip()) > 0])
        start_line = ''
        lines = code.split('\n')
        if len(lines) >= 1:
            if len(lines[0].strip()) > 0:
                if lines[0].lower() != 'python':
                    start_line = lines[0]+'\n'
        code = start_line+"\n".join([line for line in lines[1:] if len(line.strip()) > 0])
        return code

    def _update_codes(self, generated_content, target_file=None):
        new_codes = Codes(generated_content, target_file)
        differ = difflib.Differ()
        for key in new_codes.codebooks.keys():
            if key not in self.codebooks.keys() or self.codebooks[key] != new_codes.codebooks[key]:
                update_codes_content = "**[Update Codes]**\n\n"
                update_codes_content += "{} updated.\n".format(key)
                old_codes_content = self.codebooks[key] if key in self.codebooks.keys() else "# None"
                new_codes_content = new_codes.codebooks[key]

                lines_old = old_codes_content.splitlines()
                lines_new = new_codes_content.splitlines()

                unified_diff = difflib.unified_diff(lines_old, lines_new, lineterm='', fromfile='Old', tofile='New')

                unified_diff = '\n'.join(unified_diff)
                update_codes_content = update_codes_content + "\n\n" + """```
'''

'''\n""" + unified_diff + "\n```"

                self.codebooks[key] = new_codes.codebooks[key]

    def _get_codes(self) -> str:
        content = ""
        for filename in self.codebooks.keys():
            content += "{}\n```{}\n{}\n```\n\n".format(filename,
                                                       "python" if filename.endswith(".py") else filename.split(".")[
                                                           -1], self.codebooks[filename])
        return content

    def _get_codes_desinated_file(self, target_file) -> str:
        content = ""
        for filename in self.codebooks.keys():
            if filename == target_file:
                content += "{}\n```{}\n{}\n```\n\n".format(filename,
                                                        "python" if filename.endswith(".py") else filename.split(".")[
                                                            -1], self.codebooks[filename])
        return content
    

    def _get_raw_codes(self) -> str:
        content = ""
        for filename in self.codebooks.keys():
            if filename.startswith("test_requirement_"):
                continue
            content += "{}".format(self.codebooks[filename])
            return content

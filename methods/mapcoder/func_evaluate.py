import os
import time
import signal
import psutil
import tempfile
import subprocess
from multiprocessing import Process, Queue

def clear_process(process):
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
            # print(f"Killed child process: {child.pid}")
        parent.kill()
        # print(f"Killed parent process: {parent.pid}")
        time.sleep(0.1)
    except psutil.NoSuchProcess:
        # print("Process already terminated")
        pass

def func_exec(
    directory: str,
    timeout: int,
):
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
            command = "cd {}; python3 {};".format(directory, "main.py")
            process = subprocess.Popen(command,
                                    shell=True,
                                    preexec_fn=os.setsid,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                    )
        # Wait for the process to complete or timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            # If the process times out, terminate it
            clear_process(process)
            return False, f"Timeout: Process exceeded the timeout of {timeout} seconds"

        if return_code == 0:
            clear_process(process)
            return True, "pass"
        else:
            error_output = process.stderr.read().decode('utf-8')
            clear_process(process)
            if error_output:
                return False, error_output
            else:
                return True, "pass"
    except subprocess.CalledProcessError as e:
        clear_process(process)
        return False, f"Error: {e}"
    except Exception as ex:
        clear_process(process)
        return False, f"Error: {ex}"
        

def evaluate_functional_correctness(
    test_cases: list,
    completion: str,
    timeout: int = 1,
    stop_early: bool = False,
):
    test_log = ""
    passed = True
    for io in test_cases:
        # test_log += f"failed in test case: {io}\n"
        try:
            code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
                completion + "\n" + io + "\n"
            with tempfile.TemporaryDirectory() as temp_dir:
                code_path = os.path.join(temp_dir, "main.py")
                with open(code_path, "w") as f:
                    f.write(code)
                is_pass, _ = func_exec(
                    temp_dir,
                    timeout
                )
                # print(_)
            if is_pass:
                test_log += f"passed in test case: {io}\n"
            else:
                if stop_early:
                    return False, f"failed in test case: {io}\n"
                passed = False
                test_log += f"failed in test case: {io}\n"
        except Exception as e:
            if stop_early:
                return False, f"failed in test case: {io}\n"
            passed = False
            test_log += f"failed in test case: {io}\n"
    # passed = False
    return passed, test_log
                

if __name__ == "__main__":
    import json
    import concurrent.futures
    from tqdm import tqdm
    cnt = 0
    code = """
def truncate_number(n):
    if len(n) == 0:
        return None
    else:
        return n[0]"""
    err_code = """
def truncate_number(n):
    while True:
        pass"""
    test_cases = [
        "assert truncate_number([3.5]) == 3.5",
        "assert truncate_number([3]) == 3",
        "assert truncate_number([]) is None",
    ]
    # _ = evaluate_functional_correctness(test_cases, code)
    # print(_)
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        tasks = [(test_cases, code) for i in range(29)] 
        # tasks = [(data['test_cases'],  code, 1) for i in range(30)] 
        for i in range(3000):
            tasks.append((test_cases,  err_code, 1))
        for i in range(30):
            tasks.append((test_cases,  code))
        results = list(executor.map(lambda args: evaluate_functional_correctness(*args), tasks))
        for idx, _ in enumerate(results):
            print(idx, _)
                # passed, _ = evaluate_functional_correctness(data['test_cases'], data['generated_output'])
            # if passed:
            #     cnt += 1
    
    print(cnt)

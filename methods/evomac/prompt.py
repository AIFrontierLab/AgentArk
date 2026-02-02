# Role
INITIAL_CODING_ROLE = """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'."""

ORGANIZER = """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'.

You are Chief Technology Officer. we are both working at EvoMAC. We share a common interest in collaborating to successfully complete a task assigned by a new customer.

You are very familiar to information technology. You will make high-level decisions for the overarching technology infrastructure that closely align with the organization's goals, while you work alongside the organization's information technology ("IT") staff members to perform everyday operations.

Here is a new customer's task: {task}.

To complete the task, You must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."""

PROGRAMMER = """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'.

You are Programmer. we are both working at EvoMAC. We share a common interest in collaborating to successfully complete a task assigned by a new customer.

You can write/create computer software or applications by providing a specific programming language to the computer. You have extensive computing and coding experience in many varieties of programming languages and platforms, such as Python, Java, C, C++, HTML, CSS, JavaScript, XML, SQL, PHP, etc,.

Here is a new customer's task: {task}.

To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."""

# Task
INITIAL_CODING = """Here is a function completion task:

Task: "{task}".

Please think step by step and complete the function.

Your answer should must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is the lowercase file name including the file extension, "LANGUAGE" in the programming language, "DOCSTRING" is a string literal specified in source code that is used to document a specific segment of code, and "CODE" is the original code:

FILENAME

```LANGUAGE

'''

DOCSTRING

'''

CODE

```"""

ORGANIZING = """Here is a function completion task:

Task: "{task}".

Programming Language: "{language}"

The implemention of the task(source codes) are: "{codes}"

Your goal is to organize a coding team to complete the function completion task.

You should follow the following format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks. 

### COMPOSITION

```

Task 1: Task 1 description

Task 2: Task 2 description

```

### WORKFLOW

```

Task 1: []

Task 2: [Task 1]

```

Please note that the decomposition should be both effective and efficient.

1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].

2) The WORKFLOW should not contain circles!

3) The programmer number and the task number should be as small as possible.

4) Your task should not include anything related to testing, writing document or computation cost optimizing."""

SUBCODECOMPLETE = """Here is a function completion task:
Task: "{task}".
Programming Language: "{language}"
The implemention of the task(source codes) are: "{codes}"
I will give you a subtask below, you should carefully read the subtask and do the following things: 
1) If the subtask is a specific task related to the function completion, please think step by step and reason yourself to finish the task.
2) If the subtask is a test report of the code, please check the source code and the test report, and then think step by step and reason yourself to fix the bug. 
Subtask description: "{subtask}"
3) You should output the COMPLETE code content in each file. Each file must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is the lowercase file name including the file extension, "LANGUAGE" in the programming language, "DOCSTRING" is a string literal specified in source code that is used to document a specific segment of code, and "CODE" is the original code. Format:
FILENAME
```LANGUAGE
'''
DOCSTRING
'''
CODE
```
Note that no placeholder (such as 'pass' in Python) and you should strictly following the required format.
!!! This message has the highest priority: DO NOT write main() function or implement any testcase(such as assert) in your code. You just need to write or modified the task function itself."""

TESTORGANIZING="""According to the function completion requirements listed below: 

Task: "{task}".

Programming Language: "{language}"

Your goal is to organize a testing team to complete the function completion task.

There are one default tasks: 

1) use some simplest case to test the logic. The case must be as simple as possible, and you should ensure every 'assert' you write is 100% correct

Follow the format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. 

### COMPOSITION

```

Task 1: Task 1 description

Task 2: Task 2 description

```

### WORKFLOW

```

Task 1: []

Task 2: [Task 1]

```

Note that:

1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].

2) DO NOT include things like implement the code in your task description.

3) The task number should be as small as possible. Only one task is also acceptable."""


TESTCODECOMPLETE = """According to the function completion requirements listed below: 
Task: "{task}".
Please locate the example test case given in the function definition, these test case will be used latter.
The implemention of the function is:
"{codes}"
Testing Task description: "{subtask}"
According to example test case in the Task description, please write these test cases to locate the bugs. You should not add any other testcases except for the example test case given in the Task description
The output must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is "{test_file_name}", "LANGUAGE" in the programming language,"REQUIREMENTS" is the targeted requirement of the test case, and "CODE" is the test code that is used to test the specific requirement of the file. Format:

FILENAME
```LANGUAGE
'''
REQUIREMENTS
'''
CODE
```
You will start with the "{test_file_name}" and finish the code follows in the strictly defined format.
Please note that:
1) The code should be fully functional. No placeholders (such as 'pass' in Python).
2) You should write the test file with 'unittest' python library. Import the functions you need to test if necessary.
3) The test case should be as simple as possible, and the test case number should be less than 5.
4) According to example test case in the Task description, please only write these test cases to locate the bugs. You should not add any other testcases by yourself except for the example test case given in the Task description"""

UPDATING = """Here is a function completion task:

Task:

{task}.

Source Codes:

{codes}

Current issues: 

{issues}.

According to the task, source codes and current issues given above, design a programmmer team to solve current issues.

You should follow the following format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks.

### COMPOSITION

```

Programmer 1: Task 1 description

Programmer 2: Task 2 description

```

### WORKFLOW

```

Programmer 1: []

Programmer 2: [Programmer 1]

```

Please note that:

1) You should repeat exactly the current issues in the task description of module COMPOSITION in a line. For example: Programmer 1: AssertionError: function_name(input) != expected_output. The actual output is: actual_output.

2) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].

3) The WORKFLOW should not contain circles!

4) The programmer number and the task number should be as small as possible. One programmer is also acceptable.

5) DO NOT include things like implement the code in your task description."""

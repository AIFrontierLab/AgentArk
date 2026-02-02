INPUT_KB_EXEMPLARS = """Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{query}

# Exemplars:
Recall {k} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall {k} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
<description>
# Describe the problem.
</description>
<code>
# Let's think step by step to solve this problem in {language} programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
</problem>

# similarly add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
</algorithm>
</root>
"""

ALGORITHM_PROMPT = "## Relevant Algorithm to solve the next problem:\n{algorithm}"

SAMPLE_IO_PROMPT = "## Sample Test cases: \n{sample_io}\n"

PLANNING = """Given a competitive programming problem generate a concrete planning to solve the problem.
# Problem:
{example_problem}
# Planning:
{example_planning}
{algorithm_prompt}
## Problem to be solved:
{prompt}
{sample_io_prompt}
## Planning:

----------------
Important: You should give only the planning to solve the problem. Do not add extra explanation or words."""

PLANNING_FOR_VERIFICATION = """Given a competitive programming problem and a plan to solve the problem in {language}, tell whether the plan is correct to solve this problem.

# Problem:
{query}
# Planning:
{planning}

----------------
Important: Your response must follow the following xml format-```
<root>
<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>
<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>
</root>"""

FINAL_CODE_GENARATION = """Given a competitive programming problem generate {language} code to solve the problem.
{algorithm_prompt}
## Problem to be solved:
{prompt}
## Planning:
{planning}
{sample_io_prompt}
## Let's think step by step.

----------------
Important:
{std_input_prompt}
## Your response must contain only the {language} code to solve this problem. Do not add extra explanation or words."""


IMPROVING_CODE = """Given a competitive programming problem you have generated {language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.
{algorithm_prompt}
## Problem to be solved:
{prompt}
{response}
## Test Report:
{test_log}
## Modified Planning:
## Let's think step by step to modify {language} Code for solving this problem.

----------------
Important:
{std_input_prompt}
## Your response must contain the modified planning and then the {language} code inside ``` block to solve this problem."""


                
import re

from ..mas_base import MAS
from .mad_utils import Agent
from .mad_prompt import *

NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
]

class MAD_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        self.config = {}
        
        self.num_players = self.method_config["num_players"]
        self.max_round = self.method_config["max_round"]

        self.player_meta_prompt = ""
        self.moderator_prompt = ""
        self.affirmative_prompt = ""
        self.judge_prompt_last2 = ""

    def init_prompt(self, debate_topic):
        """initialize the prompt"""
        self.player_meta_prompt = PLAYER_META_PROMPT.replace("##debate_topic##", debate_topic)
        self.moderator_prompt = MODERATOR_META_PROMPT.replace("##debate_topic##", debate_topic)
        self.affirmative_prompt = AFFIRMATIVE_PROMPT.replace("##debate_topic##", debate_topic)
        self.judge_prompt_last2 = JUDGE_PROMPT_LAST2.replace("##debate_topic##", debate_topic)

    def create_agents(self):
        """create players and moderator"""
        self.players = [Agent(name) for name in NAME_LIST]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        """initialize player_meta_prompt, and start the first round of debate"""
        self.affirmative.set_meta_prompt(self.player_meta_prompt)
        self.negative.set_meta_prompt(self.player_meta_prompt)
        self.moderator.set_meta_prompt(self.moderator_prompt)

        # An affirmative agent starts the debate
        self.affirmative.add_event(self.affirmative_prompt)
        # self.aff_ans = self.affirmative.ask()
        self.aff_ans = self.call_llm(messages=self.affirmative.memory_lst)
        self.affirmative.add_memory(self.aff_ans)
        self.base_answer = self.aff_ans  

        # A negative agent responds to the affirmative agent
        self.negative.add_event(NEGATIVE_PROMPT.replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.call_llm(messages=self.negative.memory_lst)
        self.negative.add_memory(self.neg_ans)

        # A moderator evaluates the answers from both sides
        self.moderator.add_event(
            MODERATOR_PROMPT.replace('##aff_ans##', self.aff_ans)
            .replace('##neg_ans##', self.neg_ans)
            .replace('##round##', 'first')
        )
        self.mod_ans = self.call_llm(messages=self.moderator.memory_lst)
        self.mod_ans = re.sub(r"```json|```", "", self.mod_ans).strip()
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = eval(self.mod_ans)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct.get(num, f"{num}th")

    def print_answer(self, debate_topic):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(debate_topic)
        print("\n----- Base Answer -----")
        print(self.base_answer)
        print("\n----- Debate Answer -----")
        print(self.debate_answer)
        print("\n----- Debate Reason -----")
        print(self.config.get("Reason", "No reason provided."))

    def inference(self, sample):
        """inference function for MAD"""
        debate_topic = sample["query"]
        self.init_prompt(debate_topic)
        self.create_agents()
        self.init_agents()

        for round in range(self.max_round - 1):
            if self.mod_ans["debate_answer"]:   # if the debate is done, stop the loop
                break
            else:
                # set the prompt for the affirmative side and update memory
                self.affirmative.add_event(DEBATE_PROMPT.replace('##oppo_ans##', self.neg_ans))
                self.aff_ans = self.call_llm(messages=self.affirmative.memory_lst)
                self.affirmative.add_memory(self.aff_ans)

                # set the prompt for the negative side and update memory
                self.negative.add_event(DEBATE_PROMPT.replace('##oppo_ans##', self.aff_ans))
                self.neg_ans = self.call_llm(messages=self.negative.memory_lst)
                self.negative.add_memory(self.neg_ans)

                # set the prompt for the moderator and update memory
                self.moderator.add_event(
                    MODERATOR_PROMPT.replace('##aff_ans##', self.aff_ans)
                    .replace('##neg_ans##', self.neg_ans)
                    .replace('##round##', self.round_dct(round+2))
                )
                self.mod_ans = str(self.call_llm(messages=self.moderator.memory_lst))
                self.mod_ans = re.sub(r"```json|```", "", self.mod_ans).strip()
                self.moderator.add_memory(self.mod_ans)
                self.mod_ans = eval(self.mod_ans)

        if self.mod_ans["debate_answer"]:
            self.debate_answer = self.mod_ans["debate_answer"]
            self.config.update(self.mod_ans)
            self.config['success'] = True
        else:
            # let the judge decide the debate
            judge_player = Agent(name='Judge')
            aff_ans = self.affirmative.memory_lst[2]['content']
            neg_ans = self.negative.memory_lst[2]['content']

            # set the prompt for the judge and update memory
            judge_player.set_meta_prompt(self.moderator_prompt)
            judge_player.add_event(JUDGE_PROMPT_LAST1.replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            ans = self.call_llm(messages=judge_player.memory_lst)
            judge_player.add_memory(ans)

            # let the judge decide the debate and give the final answer
            judge_player.add_event(self.judge_prompt_last2)
            ans = self.call_llm(messages=judge_player.memory_lst)
            judge_player.add_memory(ans)

            ans = eval(ans)
            if ans["debate_answer"]:
                self.debate_answer = ans["debate_answer"]
                self.config['success'] = True

            self.config.update(ans)
            self.players.append(judge_player)

        return {"response": self.debate_answer}
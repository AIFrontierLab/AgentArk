from methods.mas_base import MAS

class CoT(MAS):
    def __init__(self, general_config, method_config_name=None):
        super().__init__(general_config)
    
    def inference(self, sample):
        
        prompt = sample["query"] + "\n\nLet's think step by step. Finally put your final answer in A, B, C, or D."
        
        # response = self.call_llm(prompt=prompt)
        extra_body = {"guided_choice": ["A", "B", "C", "D"]}

        response = self.call_llm(prompt=prompt, extra_body=extra_body)

        return {"response": response}

    # def __init__(self, general_config, method_config_name=None):
    #     method_config_name = "config_main" if method_config_name is None else method_config_name
    #     super().__init__(general_config, method_config_name)

    #     self.agents_num = self.method_config["agents_num"]
    #     self.rounds_num = self.method_config["rounds_num"]
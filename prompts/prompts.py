class PromptTemplate():
    def __init__(self, language: str, setting: str):
        self.language = language.upper()
        self.setting = setting
    
    def get_init_msg(self):
        pass

    def get_query_msg(self, query):
        pass

    def get_obs_msg(self, observation, reward):
        pass
    
    def get_retry_msg(self):
        return f"""No {self.language} code was found in your last response.

Your response should be a {self.language} command. Format your {self.language} command as follows:

```{self.language}
Your {self.language} code here
```
"""
    
class TemplateV2(PromptTemplate):        
    def get_init_msg(self):
        self.explore_msg = f"""
Try ```sql
SHOW TABLES``` or ```sql
DESCRIBE <table_name> to learn more about the database```.

"""
        return f"""## TASK DESCRIPTION
You are a {self.language} code generator helping me answer a question using {self.language}. 
I will ask you a question, and your task is to interact with a {self.setting} system using {self.language} commands to come up with the answer. 

## RESPONSE FORMAT
Your response should be a {self.language} command. Format your {self.language} command as follows:
```{self.language}
Your {self.language} code here
```

DO NOT WRITE ANYTHING EXCEPT FOR CODE in your response.
{self.explore_msg}

## OUTPUT DESCRIPTION
Given your {self.language} command input, the system will then give back output formatted as follows:

Output: <string>
Reward: [0, 1]

The output is the standard output from executing your {self.language} command.
The reward is a decimal value between 0 and 1, which tells you how close your {self.language} command is to the correct answer. 
The closer the reward is to 1, the closer your {self.language} command is to the correct answer.

You have to try to maximize the reward.
"""
    
    def get_query_msg(self, query):
        self.query = query
        return f"""Query: \"{query}\".
Do not generate any output or reward.
"""
    
    def get_obs_msg(self, observation, reward):
        if isinstance(observation, str) and observation == "" or isinstance(observation, list) and len(observation) == 0:
            observation = "No output"
        return f"""{self.setting} Output: {observation}
Reward: {reward}
Here is the query again: \"{self.query}\"
Try something different to generate {self.language} command to get a reward of 1.
Do not generate any output or reward.
"""
    

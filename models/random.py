import sys
sys.path.append("..")
from models.agent_base import AgentBase
import random


class RandomAgent(AgentBase):
    def __init__(self, dic_agent_conf, dic_sumo_env_conf, dic_path, cnt_round, best_round=None):
        super(RandomAgent, self).__init__(dic_agent_conf, dic_sumo_env_conf, dic_path)

    def choose_action(self, count, state):
        ''' choose the best action for current state '''
        action = random.randrange(self.dic_sumo_env_conf["NUM_PHASES"])
        return action


import os
import copy
from config import DIC_AGENTS, DIC_ENVS
import traceback


class Generator:
    def __init__(self, cnt_round, cnt_gen,
                 dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 best_round=None):
        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.agents = [None] * dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_exp_conf["MODEL_NAME"]
            agent = DIC_AGENTS[agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round,
                intersection_id=str(i)
            )
            self.agents[i] = agent

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf)

    def generate(self):
        done = False
        state = self.env.reset()
        step_num = 0
        print("[RUN] start generator file for : {0}".format(self.dic_traffic_env_conf['TRAFFIC_FILE']))
        while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"] / self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                if self.dic_exp_conf["MODEL_NAME"] in ["DGN", "GCN", "STGAT", "SimpleDQNOne"]:
                    one_state = state
                    if self.dic_exp_conf["MODEL_NAME"] == 'DGN' or self.dic_exp_conf["MODEL_NAME"] == 'STGAT':
                        action, _ = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == 'GCN':
                        action = self.agents[i].choose_action(step_num, one_state)
                    else:  # simpleDQNOne
                        if True:
                            action = self.agents[i].choose_action(step_num, one_state)
                        else:
                            action = self.agents[i].choose_action_separate(step_num, one_state)
                    action_list = action
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)
            try:
                next_state, reward, done, _ = self.env.step(action_list)
            except:
                print('traceback.format_exc():\n%s' % traceback.format_exc())
            state = next_state
            step_num += 1
        self.env.bulk_log_multi_process()
        self.env.end_sumo()
        print("[RUN] end generator file : {0}".format(self.dic_traffic_env_conf['TRAFFIC_FILE']))


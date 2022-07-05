import os, time, copy
from models.agent_dqn import DQNAgent
from models.agent_deeplight import DeeplightAgent
from models.agent_lit import LitAgent
from models.fixedtime import FixedtimeAgent
from models.fixedtimeoffset import FixedtimeOffsetAgent
from models.formula import FormulaAgent
from models.maxpressure import MaxPressureAgent
from models.random import RandomAgent
from models.sliding_formula import SlidingFormulaAgent
from models.sotl import SOTLAgent
# from env_cf import AnonENV
from envs.env_cf_cmp_zx import AnonENV
# from envs.sumoenv import SumoEnv

PATH_TO_ALL_DATA = "/Users/pengyuquan/Desktop/All_code/cityflow_data/"
PATH_TO_ALL_WORK = "/Users/pengyuquan/Desktop/All_code/run_exp_results/"

DIC_EXP_CONF = {
    "RUN_COUNTS": 3600,
    "TRAFFIC_FILE": ["cross.2phases_rou01_equal_450.xml"],
    "MODEL_NAME": "DQNAgent",
    "NUM_ROUNDS": 200,
    "NUM_GENERATORS": 3,
    "LIST_MODEL": ["Fixedtime", "Deeplight", "DQNAgent","SimpleDQN",
                   "TransferDQN","TransferDQNPress","TransferDQNPressOne","TransferDQNOne"],
    "LIST_MODEL_NEED_TO_UPDATE": ["Deeplight", "DQNAgent","SimpleDQN", "DGN","GCN",
                                  "SimpleDQNOne","STGAT", "TransferDQN","TransferDQNPress",
                                  "TransferDQNOne","TransferDQNPressOne"],
    "MODEL_POOL": False,
    "NUM_BEST_MODEL": 3,
    "PRETRAIN": True,
    "PRETRAIN_MODEL_NAME": "Random",
    "PRETRAIN_NUM_ROUNDS": 10,
    "PRETRAIN_NUM_GENERATORS": 10,
    "AGGREGATE": False,
    "DEBUG": False,
    "EARLY_STOP": False,
    "MULTI_TRAFFIC": False,
    "MULTI_RANDOM": False,
}

DIC_TRAFFIC_ENV_CONF = {
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 2,
    "NUM_LANES": 1,
    "ACTION_DIM": 2,
    "MEASURE_TIME": 10,
    "IF_GUI": True,
    "DEBUG": False,

    "INTERVAL": 1,
    "THREADNUM": 8,
    "SAVEREPLAY": True,
    "RLTRAFFICLIGHT": True,

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE = (4,),
        D_LEAVING_VEHICLE = (4,),
        # D_COMING_VEHICLE=(12,),
        # D_LEAVING_VEHICLE=(12,),

        D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
        D_CUR_PHASE=(1,),
        D_NEXT_PHASE=(1,),
        D_TIME_THIS_PHASE=(1,),
        D_TERMINAL=(1,),
        D_LANE_SUM_WAITING_TIME=(4,),
        D_VEHICLE_POSITION_IMG=(4, 60,),
        D_VEHICLE_SPEED_IMG=(4, 60,),
        D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

        D_PRESSURE=(1,),
        D_ADJACENCY_MATRIX=(2,),
        D_ADJACENCY_MATRIX_LANE = (6,),

    ),

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_been_stopped_thres01",
        "lane_num_vehicle_been_stopped_thres1",
        "lane_queue_length",
        "lane_num_vehicle_left",
        "lane_sum_duration_vehicle_left",
        "lane_sum_waiting_time",
        "terminal",
        "coming_vehicle",
        "leaving_vehicle",
        "pressure",
        "adjacency_matrix",
        "adjacency_matrix_lane",
        "connectivity",
        "inter_cmp_state",
    ],

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
        "sum_duration_vehicle_left": 0,
        "sum_num_vehicle_been_stopped_thres01": 0,
        "sum_num_vehicle_been_stopped_thres1": -0.25,
        "pressure": 0,
        "sum_lane_num_vehicle": 0,
        "sum_delta_lane_num_vehicle": 0,
    },

    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 1,
        "STRAIGHT": 1
    },

    "PHASE": [
        'WSES',
        'NSSS',
        'WLEL',
        'NLSL',
        # 'WSWL',
        # 'ESEL',
        # 'NSNL',
        # 'SSSL',
    ],

}

DIC_PATH = {
    "PATH_TO_MODEL": PATH_TO_ALL_WORK + "model/default",
    "PATH_TO_WORK_DIRECTORY": PATH_TO_ALL_WORK + "records/default",
    "PATH_TO_DATA": PATH_TO_ALL_DATA + "data/template",
    "PATH_TO_PRETRAIN_MODEL": PATH_TO_ALL_WORK + "model/default",
    "PATH_TO_PRETRAIN_WORK_DIRECTORY": PATH_TO_ALL_WORK + "records/default",
    "PATH_TO_PRETRAIN_DATA": PATH_TO_ALL_DATA + "data/template",
    "PATH_TO_AGGREGATE_SAMPLES": PATH_TO_ALL_WORK + "records/initial",
    "PATH_TO_ERROR": PATH_TO_ALL_WORK + "errors/default"
}

DIC_ENVS = {
    # "sumo": SumoEnv,
    "anon": AnonENV
}

DIC_AGENTS = {
    "DQNAgent": DQNAgent,
    "Deeplight": DeeplightAgent,
    "LIT": LitAgent,
    "Fixedtime": FixedtimeAgent,
    "FixedtimeOffset": FixedtimeOffsetAgent,
    "Formula": FormulaAgent,
    "MaxPressure": MaxPressureAgent,
    "Random": RandomAgent,
    "SlidingFormula": SlidingFormulaAgent,
    "SOTL": SOTLAgent,
}

DIC_DQNAGENT_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

# ======================这些是baseline config信息=================
DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [
        30,
        30
    ],
}

DIC_FIXEDTIME = {
    100: [3, 3],
    200: [3, 3],
    300: [7, 7],
    400: [12, 12],
    500: [15, 15],
    600: [28, 28],
    700: [53, 53]
    }

DIC_FIXEDTIME_MULTI_PHASE = {
    100: [4 for _ in range(4)],
    200: [4 for _ in range(4)],
    300: [5 for _ in range(4)],
    400: [8 for _ in range(4)],
    500: [9 for _ in range(4)],
    600: [13 for _ in range(4)],
    700: [23 for _ in range(4)],
    750: [30 for _ in range(4)]
}

DIC_FIXEDTIMEOFFSET_AGENT_CONF = {
    100: [3, 3],
    200: [3, 3],
    300: [7, 7],
    400: [12, 12],
    500: [15, 15],
    600: [28, 28],
    700: [53, 53]
}

DIC_FORMULA_AGENT_CONF = {
    "DAY_TIME": 3600,
    "UPDATE_PERIOD": 3600,
    "FIXED_TIME": [30, 30],
    "ROUND_UP": 5,
    "PHASE_TO_LANE": [[0, 1], [2, 3]],
    "MIN_PHASE_TIME": 5,
    "TRAFFIC_FILE": [
        "cross.2phases_rou01_equal_450.xml"
    ],
}

DIC_MAXPRESSURE_AGENT_CONF = {
    100: [3, 3],
    200: [3, 3],
    300: [7, 7],
    400: [12, 12],
    500: [15, 15],
    600: [28, 28],
    700: [53, 53]
}

DIC_SLIDINGFORMULA_AGENT_CONF = {
    "DAY_TIME": 3600,
    "UPDATE_PERIOD": 300,
    "FIXED_TIME": [30, 30],
    "ROUND_UP": 5,
    "PHASE_TO_LANE": [[0, 1], [3, 4], [6, 7], [9, 10]],
    "MIN_PHASE_TIME": 5,
    "TRAFFIC_FILE": [ "cross.2phases_rou01_equal_450.xml" ]
}

DIC_SOTL_AGENT_CONF = {
    "PHI": 5,
    "MIN_GREEN_VEC": 3,
    "MAX_RED_VEC": 6,
}

DIC_DEEPLIGHT_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "UPDATE_PERIOD": 300,
    "SAMPLE_SIZE": 300,
    "SAMPLE_SIZE_PRETRAIN": 3000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "EPOCHS_PRETRAIN": 500,
    "SEPARATE_MEMORY": True,
    "PRIORITY_SAMPLING": False,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "GAMMA_PRETRAIN": 0,
    "MAX_MEMORY_LEN": 1000,
    "PATIENCE": 10,
    "PHASE_SELECTOR": True,
    "KEEP_OLD_MEMORY": 0,
    "DDQN": False,
    "D_DENSE": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,

}

DIC_LIT_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "UPDATE_PERIOD": 300,
    "SAMPLE_SIZE": 300,
    "SAMPLE_SIZE_PRETRAIN": 3000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "EPOCHS_PRETRAIN": 500,
    "SEPARATE_MEMORY": True,
    "PRIORITY_SAMPLING": False,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "GAMMA_PRETRAIN": 0,
    "MAX_MEMORY_LEN": 1000,
    "PATIENCE": 10,
    "PHASE_SELECTOR": True,
    "KEEP_OLD_MEMORY": 0,
    "DDQN": False,
    "D_DENSE": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0
}

# ======================这些是baseline config信息=================

def create_config(traffic_file, args, num_rounds, ENVIRONMENT):
    NUM_COL = int(args.road_net.split('_')[0])
    NUM_ROW = int(args.road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL

    dic_exp_conf_extra = {
        "RUN_COUNTS": args.cnt,
        "MODEL_NAME": args.agent,
        "TRAFFIC_FILE": [traffic_file],  # here: change to multi_traffic
        "ROADNET_FILE": "roadnet_{0}.json".format(args.road_net),
        "NUM_ROUNDS": num_rounds,
        "NUM_GENERATORS": args.gen,
        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 3,
        "PRETRAIN": False,  #
        "PRETRAIN_MODEL_NAME": args.agent,
        "PRETRAIN_NUM_ROUNDS": 0,
        "PRETRAIN_NUM_GENERATORS": 15,
        "AGGREGATE": False,
        "DEBUG": False,
        "EARLY_STOP": False,
        "SPARSE_TEST": False,
    }

    dic_agent_conf_extra = {
        "EPOCHS": 100,
        "SAMPLE_SIZE": 1000,
        "MAX_MEMORY_LEN": 10000,
        "UPDATE_Q_BAR_EVERY_C_ROUND": False,
        "UPDATE_Q_BAR_FREQ": 5,
        "PRIORITY": False,
        "N_LAYER": 2,
        "TRAFFIC_FILE": traffic_file,
    }

    dic_traffic_env_conf_extra = {
        "USE_LANE_ADJACENCY": False,
        "TOP_K_ADJACENCY": 5,
        "ADJACENCY_BY_CONNECTION_OR_GEO": False,
        "TOP_K_ADJACENCY_LANE": 5,
        "ONE_MODEL": args.onemodel,
        "TRANSFER": False,
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "ACTION_PATTERN": "set",
        "MEASURE_TIME": 10,
        "IF_GUI": False,
        "DEBUG": False,
        "SIMULATOR_TYPE": ENVIRONMENT,
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": False,
        "SEPARATE_TEST": False,
        "NEIGHBOR": False,  # 为什么一直是false！！！！必须是True
        "MODEL_NAME": args.agent,
        "SAVEREPLAY": False,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(args.road_net),

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),
            D_LANE_NUM_VEHICLE_DOWNSTREAM=(4,),
            D_DELTA_LANE_NUM_VEHICLE=(4,),
            D_NUM_TOTAL_VEH = (1,),
            D_COMING_VEHICLE = (12,),
            D_LEAVING_VEHICLE = (12,),
            D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
            D_CUR_PHASE=(1,),
            D_NEXT_PHASE=(1,),
            D_TIME_THIS_PHASE=(1,),
            D_TERMINAL=(1,),
            D_LANE_SUM_WAITING_TIME=(4,),
            D_VEHICLE_POSITION_IMG=(4, 60,),
            D_VEHICLE_SPEED_IMG=(4, 60,),
            D_VEHICLE_WAITING_TIME_IMG=(4, 60,),
            D_PRESSURE=(4,),
            D_ADJACENCY_MATRIX=(2,),
            D_ADJACENCY_MATRIX_LANE=(6,),
        ),

        "LANE_NUM": {
            "LEFT": 1,
            "RIGHT": 1,
            "STRAIGHT": 1
        },

        "PHASE": {
            "sumo": {
                0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0] # 'NLSL',
            },
            "anon": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                4: [0, 0, 0, 0, 1, 0, 1, 0] # 'NLSL',
            },
        },

        "list_lane_order": ["WL", "WT", "EL", "ET", "SL", "ST", "NL", "NT"],
        "PHASE_LIST": [
            'WT_ET',
            'NT_ST',
            'WL_EL',
            'NL_SL',
        ],

    }

    if args.data_type == 'syn':
        data_path = "syn/road_lsr"
    elif args.data_type == 'real':
        data_path = "real"
    else:
        print("DATA TYPE ERROR")
        exit(-1)

    # ================== mode ============== 0-IRL 1-segmentRL 2-allFeatureRL 3-NewRL
    # arterial presslight
    # 大多数情况下根据实验需要调整需要的特征 奖励函数的设计 这两个需要考虑到环境env文件中特征提取和奖励函数的计算
    # 同时也可以在这里更新需要的参数的维度和特殊的参数的值
    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"] = ["cur_phase", "coming_vehicle", "leaving_vehicle"]
    dic_traffic_env_conf_extra["DIC_REWARD_INFO"] = {
        "pressure": -0.25,
        "cmp_pressure_xz":0,
    }
    if args.pressure == 'cmp':
        dic_traffic_env_conf_extra["DIC_REWARD_INFO"] = {
            "pressure": 0,
            "cmp_pressure_xz": -0.25,
        }
    if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,) # 8
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join(PATH_TO_ALL_WORK + "model", args.memo, traffic_file + "_" +
                                      time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join(PATH_TO_ALL_WORK + "records", args.memo,traffic_file + "_" +
                                               time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join(PATH_TO_ALL_DATA, data_path, str(args.road_net)),
        "PATH_TO_PRETRAIN_MODEL": os.path.join(PATH_TO_ALL_WORK + "model", "initial", traffic_file),
        "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join(PATH_TO_ALL_WORK + "records", "initial", traffic_file),
        "PATH_TO_ERROR": os.path.join(PATH_TO_ALL_WORK + "errors", args.memo)
    }

    deploy_dic_exp_conf = merge(DIC_EXP_CONF, dic_exp_conf_extra)
    # 不同的agent是不一样的合并
    deploy_dic_agent_conf = merge(DIC_DQNAGENT_AGENT_CONF, dic_agent_conf_extra)
    deploy_dic_traffic_env_conf = merge(DIC_TRAFFIC_ENV_CONF, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(DIC_PATH, dic_path_extra)

    return data_path, deploy_dic_exp_conf, deploy_dic_agent_conf, deploy_dic_traffic_env_conf, deploy_dic_path


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


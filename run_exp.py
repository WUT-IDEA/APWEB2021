import argparse
import os
from config import create_config
from multiprocessing import Process
from pipeline.pipeline import Pipeline
from utils import prepare_traffice_files

multi_process = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='env_cmp_xz_test')
    parser.add_argument("--data_type", type=str, default="syn")  # syn or real
    parser.add_argument("--road_net", type=str, default='1_6')
    parser.add_argument("--agent", type=str, default="DQNAgent")
    parser.add_argument("--cnt", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("-onemodel", action="store_true", default=False)
    parser.add_argument("-multi_process", action="store_true", default=True)
    parser.add_argument("--pressure", type=str, default="cmp")

    args = parser.parse_args()
    return args


def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,
                   dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path
                   )
    global multi_process
    ppl.run(multi_process=multi_process)
    print("[RUN] pipeline_wrapper end for: {0}".format(dic_exp_conf['TRAFFIC_FILE']))
    return


def main(args=None):
    ENVIRONMENT = ["sumo", "anon"][1]
    traffic_file_list, _, lane = prepare_traffice_files(data_type=args.data_type, road_net=args.road_net)
    num_rounds = 600  # 在这里修改一次实验需要的轮数
    global multi_process
    multi_process = args.multi_process
    process_list = []

    for traffic_file in traffic_file_list:
        data_path, \
        deploy_dic_exp_conf, \
        deploy_dic_agent_conf, \
        deploy_dic_traffic_env_conf, \
        deploy_dic_path = create_config(traffic_file, args, num_rounds, ENVIRONMENT)
        # config文件中可以配置不同实验需要的配置信息
        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if multi_process:
        for i in range(0, len(process_list), args.workers):
            i_max = min(len(process_list), i + args.workers)
            for j in range(i, i_max):
                print("[RUN] start_traffic {0}".format(j))
                process_list[j].start()
                print("[RUN] after_traffic {0}".format(j))
            for k in range(i, i_max):
                print("[RUN] traffic to join ", k)
                process_list[k].join()
                print("[RUN] traffic finish join ", k)
    else:
        print("[EXIT] multi_process false")
        exit(-1)
    return args.memo


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["KERAS_BACKEND"] = "tensorflow"
    main(args)

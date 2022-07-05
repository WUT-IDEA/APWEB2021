import pickle, json, sys, os, time, threading
import numpy as np
import pandas as pd
from multiprocessing import Process, Pool
from copy import deepcopy
import engine
import traceback

class RoadNet:
    def __init__(self, roadnet_file):
        self.roadnet_dict = json.load(open(roadnet_file, "r"))
        self.net_edge_dict = {}
        self.net_node_dict = {}
        self.net_lane_dict = {}
        self.generate_node_dict()
        self.generate_edge_dict()
        self.generate_lane_dict()

    def generate_node_dict(self):
        # node dict has key as node id, value could be the dict of input nodes and output nodes
        for node_dict in self.roadnet_dict['intersections']:
            node_id = node_dict['id']
            road_links = node_dict['roads']
            input_nodes = []
            output_nodes = []
            input_edges = []
            output_edges = {}
            for road_link_id in road_links:
                road_link_dict = self._get_road_dict(road_link_id)
                if road_link_dict['startIntersection'] == node_id:
                    end_node = road_link_dict['endIntersection']
                    output_nodes.append(end_node)
                    # todo add output edges
                elif road_link_dict['endIntersection'] == node_id:
                    input_edges.append(road_link_id)
                    start_node = road_link_dict['startIntersection']
                    input_nodes.append(start_node)
                    output_edges[road_link_id] = set()
                    pass

            # update roadlinks
            actual_roadlinks = node_dict['roadLinks']
            for actual_roadlink in actual_roadlinks:
                output_edges[actual_roadlink['startRoad']].add(actual_roadlink['endRoad'])

            net_node = {
                'node_id': node_id,
                'input_nodes': list(set(input_nodes)),
                'input_edges': list(set(input_edges)),
                'output_nodes': list(set(output_nodes)),
                'output_edges': output_edges# should be a dict, with key as an input edge, value as output edges
            }
            if node_id not in self.net_node_dict.keys():
                self.net_node_dict[node_id] = net_node

    def _get_road_dict(self, road_id):
        for item in self.roadnet_dict['roads']:
            if item['id'] == road_id:
                return item
        print("Cannot find the road id {0}".format(road_id))
        sys.exit(-1)
        # return None

    def generate_edge_dict(self):
        # edge dict has key as edge id, value could be the dict of input edges and output edges
        for edge_dict in self.roadnet_dict['roads']:
            edge_id = edge_dict['id']
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']

            net_edge = {
                'edge_id': edge_id,
                'input_node': input_node,
                'output_node': output_node,
                'input_edges': self.net_node_dict[input_node]['input_edges'],
                'output_edges': self.net_node_dict[output_node]['output_edges'][edge_id],

            }
            if edge_id not in self.net_edge_dict.keys():
                self.net_edge_dict[edge_id] = net_edge

    def generate_lane_dict(self):
        lane_dict = {}
        for node_dict in self.roadnet_dict['intersections']:
            for road_link in node_dict["roadLinks"]:
                lane_links = road_link["laneLinks"]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in lane_links:
                    start_lane = start_road + "_" + str(lane_link['startLaneIndex'])
                    end_lane = end_road + "_" +str(lane_link["endLaneIndex"])
                    if start_lane not in lane_dict:
                        lane_dict[start_lane] = {
                            "output_lanes": [end_lane],
                            "input_lanes": []
                        }
                    else:
                        lane_dict[start_lane]["output_lanes"].append(end_lane)
                    if end_lane not in lane_dict:
                        lane_dict[end_lane] = {
                            "output_lanes": [],
                            "input_lanes": [start_lane]
                        }
                    else:
                        lane_dict[end_lane]["input_lanes"].append(start_lane)

        self.net_lane_dict = lane_dict


class Intersection:
    DIC_PHASE_MAP = {0: 1, 1: 2, 2: 3, 3: 4, -1: 0}
    # intersectioin_id, _adjacency_extraction()===light_id_dict
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        self.eng = eng
        self.fast_compute = dic_traffic_env_conf['FAST_COMPUTE']

        self.list_approachs = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        # enter and exit approaches generate
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach])
            for approach in self.list_approachs}

        # lane to grids
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)
        # phase info--anon/sumo
        self.list_phases = dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]
        # from approaches generate all lanes
        self.list_entering_lanes = []
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                         range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # neighbor info
        self.adjacency_row = light_id_dict['adjacency_row']
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']

        # previous & current env info
        self.dic_lane_vehicle_previous_step = {}  # laneId-vhicle
        self.dic_lane_waiting_vehicle_count_previous_step = {}  # laneId-waiting_vehicle_count
        self.dic_vehicle_speed_previous_step = {}  # {vehicle-speed}
        self.dic_vehicle_distance_previous_step = {}  # {vehicle-distance}
        self.list_lane_vehicle_previous_step = []  # [lane_vehicle]
        self.dic_feature_previous_step = {}  # {feature}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}
        self.list_lane_vehicle_current_step = []
        self.dic_feature = {}

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2
        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        self.next_phase_to_set_index = None

        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}
        self.dic_vehicle_arrive_leave_time = dict()

        # added code
        self.list_lane_vehicle_change_rate = {}
        for lane in self.list_entering_lanes:
            self.list_lane_vehicle_change_rate[lane] = 0.33

    # added code
    def _update_lane_veh_change_rate(self):
        # r_ = gamma * r + (1-gamma) * p
        # p = D_l_m / sum_D_l_k
        # for all phase of cur inter
        # for all enter lane 12
        # try exception
        temp = self.dic_entering_approach_to_edge
        for road_id in list(self.dic_entering_approach_to_edge.values()):
            road_total_veh_num = 0
            list_lanes = []
            list_lanes += [road_id + '_' + str(i) for i in
                                    range(sum(list(self.dic_traffic_env_conf["LANE_NUM"].values())))]
            for lane in list_lanes:
                road_total_veh_num += self.dic_lane_waiting_vehicle_count_current_step[lane]
            for lane in list_lanes:
                if road_total_veh_num != 0:
                    self.list_lane_vehicle_change_rate[lane] = self.list_lane_vehicle_change_rate[lane] * 0.9 + \
                        0.1 * (self.dic_feature['enter_lane_veh_change'][lane] / road_total_veh_num)
                else:
                    pass
        print("list_lane_vehicle_change_rate "+str(self.list_lane_vehicle_change_rate))
        pass

    # added code
    def _cal_new_pressure(self, net_edge_dict, dict_neighbors_obj, reset_flag):
        all_enter_car_queue = []
        for lane in self.list_entering_lanes:
            all_enter_car_queue.append(self.dic_lane_waiting_vehicle_count_current_step[lane])

        all_leaving_car_queue = []
        if reset_flag == True:
            for lane in self.list_exiting_lanes:
                all_leaving_car_queue.append(-self.dic_lane_waiting_vehicle_count_current_step[lane])
        else:
            for inter_id in dict_neighbors_obj.keys():
                if dict_neighbors_obj[inter_id] is None:
                    all_leaving_car_queue.append(0)
                else:
                    temp = 0
                    for lane in dict_neighbors_obj[inter_id].list_entering_lanes:
                        temp += dict_neighbors_obj[inter_id].dic_lane_waiting_vehicle_count_current_step[lane]
                    all_leaving_car_queue.append(-(temp // len(dict_neighbors_obj[inter_id].list_entering_lanes)))

        # for lane in self.list_exiting_lanes:
        #     # all_leaving_car_queue.append(self.dic_lane_waiting_vehicle_count_current_step[lane])
        #     a = lane.split('_')
        #     road_id_temp = a[0]+'_'+a[1]+'_'+a[2]+'_'+a[3]
        #     inter_id = net_edge_dict[road_id_temp]['output_node']
        #     if inter_id in dict_neighbors_obj.keys():
        #         neighbor = dict_neighbors_obj[inter_id]
        #         sum = 0
        #         for lane_nei in neighbor.list_entering_lanes:
        #             try:
        #                 sum += neighbor.dic_lane_waiting_vehicle_count_current_step[lane_nei]
        #             except:
        #                 sum += self.dic_lane_waiting_vehicle_count_current_step[lane]
        #         all_leaving_car_queue.append(-(sum//len(neighbor.list_entering_lanes)))
        #     else:
        #         all_leaving_car_queue.append(-self.dic_lane_waiting_vehicle_count_current_step[lane])
                # all_leaving_car_queue.append(0)
        return all_enter_car_queue + all_leaving_car_queue


    def _cal_lane_new_left_enter_veh(self, dict_pre, dict_cur):
        # :return: dict key:lane_id  value: veh_change_num
        temp = {}
        for lane in self.list_entering_lanes:
            temp[lane] = len(dict_pre[lane]) - len(dict_cur[lane])

        print("temp "+str(temp.values()))
        return temp

    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)  # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")
            elif action_pattern == "set":  # set to certain phase
                self.next_phase_to_set_index = self.DIC_PHASE_MAP[action]  # if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:  # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def _change_lane_vehicle_dic_to_list(self, dic_lane_vehicle):
        list_lane_vehicle = []
        for value in dic_lane_vehicle.values():
            list_lane_vehicle.extend(value)
        return list_lane_vehicle

    # not used
    def update_current_measurements(self, path_to_log):
        # need change, debug in seeing format
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = []  # = self.eng.get_lane_vehicles()
        # not implement
        flow_tmp = self.eng.get_lane_vehicles()
        self.dic_lane_vehicle_current_step = {key: None for key in self.list_entering_lanes}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = flow_tmp[lane]

        self.dic_lane_waiting_vehicle_count_current_step = self.eng.get_lane_waiting_vehicle_count()
        self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
        self.dic_vehicle_distance_current_step = self.eng.get_vehicle_distance()

        # get vehicle list
        self.list_lane_vehicle_current_step = self._change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = self._change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane, path_to_log)

        # update feature
        self._update_feature()

    def update_current_measurements_map(self, simulator_state, path_to_log, test_flag,
                                        net_edge_dict, neighbors, reset_flag):
        # need change, debug in seeing format
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] \
                = simulator_state["get_lane_waiting_vehicle_count"][lane]
        for lane in self.list_exiting_lanes:
            self.dic_lane_waiting_vehicle_count_current_step[lane] \
                = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        # get vehicle list
        self.list_lane_vehicle_current_step = self._change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = self._change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane, path_to_log)

        # update feature
        self._update_feature_map(simulator_state, test_flag, net_edge_dict, neighbors, reset_flag)

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                # sys.exit(-1)
                pass

    def log_one_vehicle(self, vehicle, path_to_log):
        inter = str(self.inter_id[0]) + '_' + str(self.inter_id[1])
        path_to_log_file = os.path.join(path_to_log, "vehicle_inter_{0}.csv".format(inter))
        df = [vehicle, self.dic_vehicle_arrive_leave_time[vehicle]["enter_time"],
              self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"]]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

    def _update_left_time(self, list_vehicle_left, path_to_log):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
                self.log_one_vehicle(vehicle, path_to_log)
                self.dic_vehicle_arrive_leave_time.pop(vehicle)
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    # not used
    def _update_feature(self):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None  # self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle(self.list_exiting_lanes)
        dic_feature["coming_vehicle"] = self._get_coming_vehicles()
        dic_feature["leaving_vehicle"] = self._get_leaving_vehicles()
        dic_feature["pressure"] = self._get_pressure()
        dic_feature["adjacency_matrix"] = None  # self._get_adjacency_row()

    def _update_feature_map(self, simulator_state, test_flag, net_edge_dict, neighbors, reset_flag):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None  # self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream(simulator_state)
        dic_feature["delta_lane_num_vehicle"] = \
            [dic_feature["lane_num_vehicle"][i] - dic_feature["lane_num_vehicle_downstream"][i]
            for i in range(len(dic_feature["lane_num_vehicle_downstream"]))]

        if self.fast_compute or test_flag:
            dic_feature["coming_vehicle"] = None
            dic_feature["leaving_vehicle"] = None
        else:
            dic_feature["coming_vehicle"] = self._get_coming_vehicles(simulator_state)
            dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(simulator_state)

        dic_feature["num_total_veh"] = simulator_state['num_total_veh']
        # dic_feature["pressure"] = self._get_pressure()
        dic_feature["pressure"] = self._cal_new_pressure(net_edge_dict, neighbors, reset_flag)
        dic_feature["lane_num_vehicle_been_stopped_thres01"] = None  # self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1,self.list_entering_lanes)
        dic_feature["lane_queue_length"] = None  # self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = None  # self._get_lane_sum_waiting_time(self.list_entering_lanes)
        dic_feature["terminal"] = None
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()

        # added code update added feature
        # if self.get_current_time() > 100:
        #     dic_feature['enter_lane_veh_change'] = \
        #         self._cal_lane_new_left_enter_veh(self.dic_lane_vehicle_previous_step
        #                                       , self.dic_lane_vehicle_current_step) # dict
        # 出道==下游路口的进道，车辆数的变化，前一步的，不是一段时间的累计值，不满足统计学上的要求
        # 错误2：需要的结果是dict key：进道  value：进道前后的车辆数差
        # 需要的是"步长"上 前一个-后一个 （等待的车辆的变化，不然就要考虑来的车辆对计算的影响）
        # 计算的步长为单个时间步的话，会造成更新的过于频繁，导致计算的结果可能一直都是0
        # dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream(simulator_state)

        self.dic_feature = dic_feature

    # not used
    def _add_suffix_to_dict_key(target_dict, suffix):
        keys = list(target_dict.keys())
        for key in keys:
            target_dict[key + "_" + suffix] = target_dict.pop(key)
        return target_dict

    def update_neighbor_info(self, neighbors, dic_feature):
        # print(dic_feature)
        none_dic_feature = deepcopy(dic_feature)
        for key in none_dic_feature.keys():
            if none_dic_feature[key] is not None:
                if "cur_phase" in key:
                    none_dic_feature[key] = [1] * len(none_dic_feature[key])
                elif "num_total_veh" in key:
                    none_dic_feature[key] = []
                else:
                    none_dic_feature[key] = [0] * len(none_dic_feature[key])
            else:
                none_dic_feature[key] = None
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            example_dic_feature = {}
            if neighbor is None:
                pass
                example_dic_feature["cur_phase_{0}".format(i)] = none_dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = none_dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = none_dic_feature["lane_num_vehicle"]
                example_dic_feature["lane_num_vehicle_downstream_{0}".format(i)] = \
                    none_dic_feature["lane_num_vehicle_downstream"]
                example_dic_feature["coming_vehicle_{0}".format(i)] = none_dic_feature["coming_vehicle"]
                example_dic_feature["leaving_vehicle_{0}".format(i)] = none_dic_feature["leaving_vehicle"]
            else:
                example_dic_feature["cur_phase_{0}".format(i)] = neighbor.dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = neighbor.dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = neighbor.dic_feature["lane_num_vehicle"]
                example_dic_feature["lane_num_vehicle_downstream_{0}".format(i)] = \
                    neighbor.dic_feature["lane_num_vehicle_downstream"]
                example_dic_feature["coming_vehicle_{0}".format(i)] = neighbor.dic_feature["coming_vehicle"]
                example_dic_feature["leaving_vehicle_{0}".format(i)] = neighbor.dic_feature["leaving_vehicle"]

            dic_feature.update(example_dic_feature)
        return dic_feature

    def _get_adjacency_row(self):
        return self.adjacency_row

    def _lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter

    def _get_coming_vehicles(self, simulator_state):
        coming_distribution = []
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        bins = np.linspace(0, 300, 4).tolist()
        for lane in self.list_entering_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            coming_distribution.extend(self._lane_position_mapper(coming_vehicle_position, bins))

        return coming_distribution

    def _get_leaving_vehicles(self, simulator_state):
        leaving_distribution = []
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        bins = np.linspace(0, 300, 4).tolist()
        for lane in self.list_exiting_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            leaving_distribution.extend(self._lane_position_mapper(coming_vehicle_position, bins))

        return leaving_distribution

    # you must need to change the computer way
    def _get_pressure(self):
        '''
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            all_enter_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]

        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            all_leaving_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]

        p = all_enter_car_queue - all_leaving_car_queue

        if p < 0:
            p = -p
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    # those _get_lane_xxx maybe used or not in _update_feature_map or others
    def _get_lane_queue_length(self, list_lanes):
        # queue length for each lane
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        # vehicle number for each lane
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    def _get_lane_num_vehicle_downstream(self, simulator_state):
        # vehicle number for each lane
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        return [len(lane_vid_mapping_dict[lane]) for lane in self.list_exiting_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        # waiting time for each lane
        raise NotImplementedError

    def _get_lane_list_vehicle_left(self, list_lanes):
        # get list of vehicles left at each lane
        # need to check
        raise NotImplementedError

    # non temporary
    def _get_lane_num_vehicle_left(self, list_lanes):
        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):
        ## not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_vehicle_position(self, list_lanes):
        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_vehicle_current_step[lane]
            for vec in list_vec_id:
                pos = int(self.dic_vehicle_distance_current_step[vec])
                pos_grid = min(pos // self.length_grid, self.num_grid)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # debug
    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_distance_current_step[veh_id]
            speed = self.dic_vehicle_speed_current_step[veh_id]
            return pos, speed
        except:
            return None, None

    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane] for lane in list_lanes]

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):
        raise NotImplementedError

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        try:
            dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in
                         list_state_features}
        except:
            print('traceback.format_exc():\n%s' % traceback.format_exc())
        return dic_state

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])
        dic_reward['pressure'] = np.absolute(np.sum(self.dic_feature["pressure"]))

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

class AnonENV:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.feature_name_for_neighbor = \
            self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            # raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        self.roadnet_temp = RoadNet('{0}'.format(file))

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0", "1", "2", "3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a - b) ** 2))

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                row = np.array([0] * total_inter_num)
                # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]['location']
                    dist = AnonENV._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['total_inter_num'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    if edge_id_dict[road_id]['to'] not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(edge_id_dict[road_id]['to'])

        return traffic_light_node_dict

    def reset(self):
        self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
                                 self.dic_traffic_env_conf["THREADNUM"],
                                 self.dic_traffic_env_conf["SAVEREPLAY"],
                                 self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                                 False)
        self.load_roadnet(self.dic_traffic_env_conf["ROADNET_FILE"])
        self.load_flow(self.dic_traffic_env_conf["TRAFFIC_FILE"])
        self.traffic_light_node_dict = self._adjacency_extraction()
        # initialize intersections (grid)
        self.list_intersection = [Intersection((i + 1, j + 1), self.dic_traffic_env_conf, self.eng,
                                self.traffic_light_node_dict["intersection_{0}_{1}".format(i + 1, j + 1)])
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]

        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i + 1, j + 1)] = count
                count += 1

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)):  # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,  # self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance(),
                                  }
        self.system_states["num_total_veh"] = np.sum(
            [len(self.system_states["get_lane_vehicles"][lane]) for lane in self.list_lanes])


        ########################
        try:
            print("55555===========")
            for inter in self.list_intersection:
                neighbor_inters_ = {}
                for neighbor_inter_id in inter.neighbor_ENWS:
                    if neighbor_inter_id is not None:
                        neighbor_inters_[neighbor_inter_id] = self.list_intersection[self.id_to_index[neighbor_inter_id]]
                    else:
                        neighbor_inters_[neighbor_inter_id] = None
                inter.update_current_measurements_map(self.system_states,self.path_to_log, False,
                                                      self.roadnet_temp.net_edge_dict, neighbor_inters_, True)
        except:
            print("55555")
            print('traceback.format_exc():\n%s' % traceback.format_exc())
        ########################  上面一行，要做传值操作，在inter自带的信息里面已经有了neighbor和adjacency的信息
        #

        # for inter in self.list_intersection:
        #     inter.update_current_measurements_map(self.system_states, self.path_to_log, False)

        # update neighbor's info
        try:
            print("66666666===========")
            if self.dic_traffic_env_conf["NEIGHBOR"]:
                for inter in self.list_intersection:
                    neighbor_inter_ids = inter.neighbor_ENWS
                    neighbor_inters = []
                    for neighbor_inter_id in neighbor_inter_ids:
                        if neighbor_inter_id is not None:
                            neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                        else:
                            neighbor_inters.append(None)
                    inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))
        except:
            print("6666")
            print('traceback.format_exc():\n%s' % traceback.format_exc())
        state, done = self.get_state()
        return state

    def reset_test(self):
        self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
                                 self.dic_traffic_env_conf["THREADNUM"],
                                 self.dic_traffic_env_conf["SAVEREPLAY"],
                                 self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                                 False)
        self.load_roadnet(self.dic_traffic_env_conf["ROADNET_FILE"])
        self.load_flow(self.dic_traffic_env_conf["TRAFFIC_FILE"])

        self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i + 1, j + 1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i + 1, j + 1)])
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]
        # no id_to_index = {}

        # get lanes list
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)):  # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        # get new measurements
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance(),
                                  }
        self.system_states["num_total_veh"] = np.sum(
            [len(self.system_states["get_lane_vehicles"][lane]) for lane in self.list_lanes])

        ########################
        for inter in self.list_intersection:
            neighbor_inter_ids_ = inter.neighbor_ENWS
            neighbor_inters_ = {}
            for neighbor_inter_id in neighbor_inter_ids_:
                if neighbor_inter_id is not None:
                    neighbor_inters_[neighbor_inter_id] = self.list_intersection[self.id_to_index[neighbor_inter_id]]
                else:
                    neighbor_inters_[neighbor_inter_id] = None
            inter.update_current_measurements_map(self.system_states,self.path_to_log, False,
                                                  self.roadnet_temp.net_edge_dict,neighbor_inters_, True)
        ########################

        # update neighbor's info
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inters = []
                for neighbor_inter_id in inter.neighbor_ENWS:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))

        state, done = self.get_state()
        return state

    def step(self, action, test_flag):
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"] - 1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0] * len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()
            before_action_feature = self.get_feature()

            self._inner_step(action_in_sec, test_flag)

            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

        next_state, done = self.get_state()
        return next_state, reward, done, average_reward_action_list

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])
                       for inter in self.list_intersection]
        return list_reward

    def get_state(self):
        # consider neighbor info
        try:
            list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
                          for inter in self.list_intersection]
            done = self._check_episode_done(list_state)
        except:
            print('traceback.format_exc():\n%s' % traceback.format_exc())
        return list_state, done

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def _check_episode_done(self, list_state):
        # ======== to implement ========
        return False

    def _inner_step(self, action, test_flag):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1 / self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        if self.dic_traffic_env_conf["FAST_COMPUTE"] or test_flag:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,  # self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance()
                                  }
        self.system_states["num_total_veh"] = \
            np.sum([len(self.system_states["get_lane_vehicles"][lane]) for lane in self.list_lanes])

        if self.dic_traffic_env_conf['DEBUG']:
            print("[DEBUG] Get system state time: {0}".format(time.time() - start_time))
        # get new measurements
        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        ########################
        for inter in self.list_intersection:
            neighbor_inters_ = {}
            for neighbor_inter_id in inter.neighbor_ENWS:
                if neighbor_inter_id is not None:
                    neighbor_inters_[neighbor_inter_id] = self.list_intersection[self.id_to_index[neighbor_inter_id]]
                else:
                    neighbor_inters_[neighbor_inter_id] = None
            inter.update_current_measurements_map(self.system_states,self.path_to_log, False,
                                                  self.roadnet_temp.net_edge_dict,neighbor_inters_, False)
        ########################

        # for inter in self.list_intersection:
        #     inter.update_current_measurements_map(self.system_states, self.path_to_log, test_flag)

        # update neighbor's info
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))

        if self.dic_traffic_env_conf['DEBUG']:
            print("[DEBUG] Update measurements time: {0}".format(time.time() - start_time))

        # self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        # self.log_phase()

    def load_roadnet(self, roadnetFile=None):
        start_time = time.time()
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        print("[RUN] successfully load roadnetFile:{0}, time: {1}".format(roadnetFile, time.time() - start_time))

    def load_flow(self, flowFile=None):
        start_time = time.time()
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        print("[RUN] successfully load flowFile: {0}, time: {1}".format(flowFile, time.time() - start_time))

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("[RUN] bulk_log_multi_process before START")
            p.start()
            print("[RUN] bulk_log_multi_process end    START")
            process_list.append(p)

        print("[RUN] bulk_log_multi_process before JOIN")
        for t in process_list:
            t.join()
        print("[RUN] bulk_log_multi_process end    JOIN")
        f = open(os.path.join(self.path_to_log, "bulk_log_done_mark.txt"), "a")
        f.close()

    def bulk_log(self):
        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()


    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()

    def log_hidden_state(self, hidden_states):
        path_to_log_file = os.path.join(self.path_to_log, "hidden_states.pkl")
        with open(path_to_log_file, "wb") as f:
            pickle.dump(hidden_states, f)

    # not used in log_xxx
    def _list_to_str(self, alist):
        new_str = ""
        for s in alist:
            new_str = new_str + str(s) + " "
        return new_str

    # not used log_xxxx
    def log_lane_vehicle_position(self):
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + self._list_to_str(
                    inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt" % dic_lane_map[lane]),
                                "a"))

    def log_first_vehicle(self):
        _veh_id = "flow_0_"
        _veh_id_2 = "flow_2_"
        _veh_id_3 = "flow_4_"
        _veh_id_4 = "flow_6_"

        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

    def end_sumo(self):
        print("[RUN] anon process END")



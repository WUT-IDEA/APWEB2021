import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import time
from multiprocessing import Process, Pool
from copy import deepcopy
import engine
# import cityflow as engine

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
                'output_edges': output_edges  # should be a dict, with key as an input edge, value as output edges
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
        '''
        edge dict has key as edge id, value could be the dict of input edges and output edges
        '''
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
                    end_lane = end_road + "_" + str(lane_link["endLaneIndex"])
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

    def hasEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return True
        else:
            return False

    def getEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return edge_id
        else:
            return None

    def getOutgoing(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return self.net_edge_dict[edge_id]['output_edges']
        else:
            return []


class Intersection:
    DIC_PHASE_MAP = {0: 1,1: 2,2: 3,3: 4,-1: 0}
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        self.eng = eng
        self.fast_compute = dic_traffic_env_conf['FAST_COMPUTE']
        self.controlled_model = dic_traffic_env_conf['MODEL_NAME']
        self.path_to_log = path_to_log
        self.dic_traffic_env_conf = dic_traffic_env_conf

        # =====  intersection settings =====
        self.list_approachs = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"W": 2, "E": 0, "S": 3, "N": 1}
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        # 源代码这里的NS应该是反了的 现在修正过来了
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})

        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach])
            for approach in self.list_approachs}
        # 这两个没有使用？
        self.dic_entering_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}
        self.dic_exiting_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)
        """
        ANON_PHASE_REPRE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }
        """
        self.list_phases = dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]

        # generate all lanes
        self.list_entering_lanes = []
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                         range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # 在这里拼接一些路口固定的转向信息  W E N S
        self.turn_map = ["N", "E", "S", "S", "W", "N", "E", "S", "W", "W", "N", "E"]
        self.tm_to_laneIndex = {"WL":0, "WT":1, "WR":2, "EL":3, "ET":4, "ER":5,
                                "NL":6, "NT":7, "NR":8, "SL":9, "ST":10, "SR":11}
        self.tm_to_phaseIndex = {"WL":0, "WT":1, "EL":2, "ET":3,
                                 "NL":4, "NT":5, "SL":6, "ST":7}
        self.phase_to_index = {
            1: "WT_ET",
            2: "NT_ST",
            3: "WL_EL",
            4: "NL_SL",
        }
        self.approach_lane_fix_num = {  # 描述的是这个方向的进车道和出车道在laneId-to-index中的固定索引位置
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11],
        }
        self.exiting_approach_source_lane = {  # 描述的是这个出道的车来自哪些lanes 索引
            "W": [4, 8, 9],
            "E": [1, 6, 11],
            "N": [0, 5, 10],
            "S": [2, 3, 7],
        }
        # 周边四个方向 邻居的id
        self.neighbors_direction_to_id = {
            "W": "intersection_{0}_{1}".format(inter_id[0] - 1, inter_id[1]),
            "E": "intersection_{0}_{1}".format(inter_id[0] + 1, inter_id[1]),
            "S": "intersection_{0}_{1}".format(inter_id[0], inter_id[1] - 1),
            "N": "intersection_{0}_{1}".format(inter_id[0], inter_id[1] + 1)
        }

        self.adjacency_row = light_id_dict['adjacency_row']
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']

        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            self.neighbor_lanes_ENWS = light_id_dict['entering_lane_ENWS']
            def _get_top_k_lane(lane_id_list, top_k_input):
                top_k_lane_indexes = []
                for i in range(top_k_input):
                    lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                    top_k_lane_indexes.append(lane_id)
                return top_k_lane_indexes
            self._adjacency_row_lanes = {}
            # _adjacency_row_lanes is the lane id, not index
            for lane_id in self.list_entering_lanes:
                if lane_id in light_id_dict['adjacency_matrix_lane']:
                    self._adjacency_row_lanes[lane_id] = light_id_dict['adjacency_matrix_lane'][lane_id]
                else:
                    self._adjacency_row_lanes[lane_id] = \
                        [_get_top_k_lane([], self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]),
                        _get_top_k_lane([], self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"])]
            # order is the entering lane order, each element is list of two lists
            self.adjacency_row_lane_id_local = {}
            for index, lane_id in enumerate(self.list_entering_lanes):
                self.adjacency_row_lane_id_local[lane_id] = index

        # previous & current
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}

        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

    def update_cmp_pressure_xz(self, neighbor_objs):
        list_enter_lanes_veh = []
        list_exit_lanes_veh = []
        for lane in self.list_entering_lanes:
            list_enter_lanes_veh.append(self.dic_lane_waiting_vehicle_count_current_step[lane])
        for lane in self.list_exiting_lanes:
            list_exit_lanes_veh.append(self.dic_lane_waiting_vehicle_count_current_step[lane])

        list_exit_lanes_veh_final = [0] * len(self.list_exiting_lanes)

        for approach in self.list_approachs:  # W E N S
            for index in self.approach_lane_fix_num[approach]:  # 0 1 2 WENS 是从进道角度开始处理
                one_neighbor_id = self.neighbors_direction_to_id[self.turn_map[index]]
                temp = sum([list_exit_lanes_veh[i] for i in self.approach_lane_fix_num[self.turn_map[index]]])
                if one_neighbor_id in neighbor_objs.keys():
                    one_neighbor = neighbor_objs[one_neighbor_id]
                    if one_neighbor.current_phase_index in [-1, -2]:  # 处理全红或全黄相位
                        list_exit_lanes_veh_final[index] = temp/3
                        continue
                    open_phase_lane = []
                    for one in self.phase_to_index[one_neighbor.current_phase_index].split('_'):
                        open_phase_lane.append(self.tm_to_laneIndex[one])
                    for i in self.approach_lane_fix_num[self.turn_map[index]]:
                        if i in open_phase_lane:
                            temp -= list_exit_lanes_veh[i]
                    temp /= 3
                list_exit_lanes_veh_final[index] = temp
        self.dic_feature["cmp_pressure_xz"] = sum(list_enter_lanes_veh) - sum(list_exit_lanes_veh)

    def new_pressure_phase(self):
        """
            考虑了当前路口开启相位的情况，只有开启的相位才会执行减去出道上的车辆数值，没有开启的相位的压力直接就是进道上的车辆数
        """
        prefixs = [0] * len(self.list_exiting_lanes)

        if self.current_phase_index not in [-1, -2]:
            for i in self.phase_to_index[self.current_phase_index].split('_'):
                prefixs[self.tm_to_laneIndex[i]] = 1

        list_enter_lanes_veh = []
        list_exit_lanes_veh = []
        for lane in self.list_entering_lanes:
            list_enter_lanes_veh.append(self.dic_lane_waiting_vehicle_count_current_step[lane])

        for i, lane in enumerate(self.list_exiting_lanes):
            list_exit_lanes_veh.append(self.dic_lane_waiting_vehicle_count_current_step[lane] * prefixs[i])

        return sum(list_enter_lanes_veh) - sum(list_exit_lanes_veh)


    def build_adjacency_row_lane(self, lane_id_to_global_index_dict):
        self.adjacency_row_lanes = []  # order is the entering lane order, each element is list of two lists
        for entering_lane_id in self.list_entering_lanes:
            _top_k_entering_lane, _top_k_leaving_lane = self._adjacency_row_lanes[entering_lane_id]
            top_k_entering_lane = []
            top_k_leaving_lane = []
            for lane_id in _top_k_entering_lane:
                top_k_entering_lane.append(lane_id_to_global_index_dict[lane_id] if lane_id is not None else -1)
            for lane_id in _top_k_leaving_lane:
                top_k_leaving_lane.append(lane_id_to_global_index_dict[lane_id]
                                          if (lane_id is not None) and (
                            lane_id in lane_id_to_global_index_dict.keys())  # TODO leaving lanes of system will also have -1
                                          else -1)
            self.adjacency_row_lanes.append([top_k_entering_lane, top_k_leaving_lane])

    # set
    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                self.all_yellow_flag = False
                ################
                # self._update_lane_veh_change_rate()
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(
                        self.list_phases)  # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_phase_to_set_index = self.DIC_PHASE_MAP[action]  # if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:  # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                # traci.trafficlights.setRedYellowGreenState(
                #    self.node_light, self.all_yellow_phase_str)
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

    def update_current_measurements_map(self, simulator_state):
        # need change, debug in seeing format
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

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
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] \
                = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = \
            list(set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = \
            list(set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane)
        # update feature
        self._update_feature_map(simulator_state)

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

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    @staticmethod
    def _add_suffix_to_dict_key(target_dict, suffix):
        keys = list(target_dict.keys())
        for key in keys:
            target_dict[key + "_" + suffix] = target_dict.pop(key)
        return target_dict

    def _update_feature_map(self, simulator_state):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        # dic_feature["vehicle_position_img"] = self._get_lane_vehicle_position(self.list_entering_lanes)
        # dic_feature["vehicle_speed_img"] = self._get_lane_vehicle_speed(self.list_entering_lanes)
        # dic_feature["vehicle_acceleration_img"] = None
        # dic_feature["vehicle_waiting_time_img"] =
        #   self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["pressure"] = [self._get_pressure()]
        # dic_feature["cmp_pressure"] = self._cal_new_pressure(net_edge_dict, neighbors)
        dic_feature["cmp_pressure_xz"] = 0
        dic_feature["cmp_pressure_xz_new"] = self.new_pressure_phase()

        dic_feature["coming_vehicle"] = self._get_coming_vehicles(simulator_state)
        dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(simulator_state)

        dic_feature["lane_num_vehicle_been_stopped_thres01"] = self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_queue_length"] = self._get_lane_queue_length(self.list_entering_lanes)
        # dic_feature["lane_num_vehicle_left"] = None
        # dic_feature["lane_sum_duration_vehicle_left"] = None
        # dic_feature["lane_sum_waiting_time"] = self._get_lane_sum_waiting_time(self.list_entering_lanes)
        # dic_feature["terminal"] = None
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()  # TODO this feature should be a dict? or list of lists
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            dic_feature["adjacency_matrix_lane"] = self._get_adjacency_row_lane()  # row: entering_lane # columns: [inputlanes, outputlanes]
            dic_feature['connectivity'] = self._get_connectivity(self.neighbor_lanes_ENWS)
        self.dic_feature = dic_feature

    def _get_adjacency_row(self):
        return self.adjacency_row

    def _get_adjacency_row_lane(self):
        return self.adjacency_row_lanes

    def lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter

    def _get_coming_vehicles(self, simulator_state):
        # TODO f vehicle position   eng.get_vehicle_distance()  ||  eng.get_lane_vehicles()
        coming_distribution = []
        # dimension = num_lane*3*num_list_entering_lanes
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        # TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()
        for lane in self.list_entering_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            coming_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return coming_distribution

    def _get_leaving_vehicles(self, simulator_state):
        leaving_distribution = []
        # dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        # TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()
        for lane in self.list_exiting_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            leaving_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return leaving_distribution

    def _get_pressure(self):
        # TODO eng.get_vehicle_distance(), another way to calculate pressure & queue length
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            all_enter_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]
        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            all_leaving_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]
        p = all_enter_car_queue - all_leaving_car_queue
        if p < 0:
            p = -p
        return p

        # return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
        #    [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    def _get_lane_num_vehicle_downstream(self, simulator_state):
        '''
        vehicle number for each lane
        '''
        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        return [len(lane_vid_mapping_dict[lane]) for lane in self.list_exiting_lanes]

    def _get_connectivity(self, dic_of_list_lanes):
        '''
        vehicle number for each lane
        '''
        result = []
        for i in range(len(dic_of_list_lanes['lane_ids'])):
            num_of_vehicles_on_road = sum(
                [len(self.dic_lane_vehicle_current_step[lane]) for lane in dic_of_list_lanes['lane_ids'][i]])
            result.append(num_of_vehicles_on_road)

        lane_length = [0] + dic_of_list_lanes['lane_length']
        if np.sum(result) == 0:
            result = [1] + result
        else:
            result = [np.sum(result)] + result
        connectivity = list(np.array(result * np.exp(-np.array(lane_length) / (self.length_lane * 4))))
        # print(connectivity)
        # sys.exit()
        return connectivity

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        raise NotImplementedError

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''
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

    # def _get_position_grid_along_lane(self, vec):
    #     pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
    #     return min(pos // self.length_grid, self.num_grid)

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
        # customize your own state
        # print(list_state_features)
        # print(self.dic_feature)
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in
                     list_state_features}
        return dic_state

    def get_reward(self, dic_reward_info):
        # customize your own reward
        dic_reward = dict()
        # dic_reward["flickering"] = None
        # dic_reward["sum_lane_queue_length"] = None
        # dic_reward["sum_lane_wait_time"] = None
        # dic_reward["sum_lane_num_vehicle_left"] = None
        # dic_reward["sum_duration_vehicle_left"] = None
        # dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = \
            np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        dic_reward['pressure'] = np.absolute(np.sum(self.dic_feature["pressure"]))
        dic_reward['cmp_pressure_xz'] = np.absolute(self.dic_feature['cmp_pressure_xz'])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class AnonENV:
    list_intersection_id = ["intersection_1_1"]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.feature_name_for_neighbor = self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("MIN_ACTION_TIME should include YELLOW_TIME")

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        self.roadnet = RoadNet('{0}'.format(file))

    def reset(self):
        self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
                                 self.dic_traffic_env_conf["THREADNUM"],
                                 self.dic_traffic_env_conf["SAVEREPLAY"],
                                 self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                                 False)
        self.load_roadnet(self.dic_traffic_env_conf["ROADNET_FILE"])
        self.load_flow(self.dic_traffic_env_conf["TRAFFIC_FILE"])

        print("=========================RESET=====================")
        # cityflow_config = {
        #     "interval": self.dic_traffic_env_conf["INTERVAL"],
        #     "seed": 0,
        #     "laneChange": False,
        #     "dir": self.path_to_work_directory + "/",
        #     "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
        #     "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
        #     "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
        #     "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
        #     "roadnetLogFile": "frontend/web/roadnetLogFile.json",
        #     "replayLogFile": "frontend/web/replayLogFile.txt"
        # }
        # with open(os.path.join(self.path_to_work_directory, "cityflow.pipeline"), "w") as json_file:
        #     json.dump(cityflow_config, json_file)
        # self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.pipeline"), thread_num=1)

        # get adjacency
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            self.traffic_light_node_dict = self._adjacency_extraction_lane()
        else:
            self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i + 1, j + 1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict[
                                                   "intersection_{0}_{1}".format(i + 1, j + 1)], self.path_to_log)
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]
        # set index for intersections and global index for lanes
        self.id_to_index = {}
        count_inter = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i + 1, j + 1)] = count_inter
                count_inter += 1
        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)):  # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        # build adjacency_matrix_lane in index from _adjacency_matrix_lane
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            for inter in self.list_intersection:
                inter.build_adjacency_row_lane(self.lane_id_to_index)

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
                                  "get_vehicle_distance": self.eng.get_vehicle_distance()
                                  }
        # 更新每一个路口的feature
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)

        # update neighbor's info 前面更新完每一个路口的feature后 将邻居的新信息注入到当前路口
        # 所以 理论上是可以在更新邻居的信息的时候计算压力的 也就是单独更新压力
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = self.update_neighbor_feature(neighbor_inters, deepcopy(inter.dic_feature))

        # 前面的信息都更新完毕后再计算压力
        for inter in self.list_intersection:
            neighbor_objs = {}
            for one_neighbor_id in inter.neighbor_ENWS:
                if one_neighbor_id is not None:
                    neighbor_objs[one_neighbor_id] = self.list_intersection[self.id_to_index[one_neighbor_id]]
            inter.update_cmp_pressure_xz(neighbor_objs)

        state, done = self.get_state()
        return state

    def update_neighbor_feature(self, neighbors_obj, old_feature):
        none_dic_feature = deepcopy(old_feature)
        for key in none_dic_feature.keys():
            if none_dic_feature[key] is not None:
                if "cur_phase" in key:
                    none_dic_feature[key] = [1] * len(none_dic_feature[key])
                else:
                    none_dic_feature[key] = [0] * len(none_dic_feature[key])
            else:
                none_dic_feature[key] = None
        for i in range(len(neighbors_obj)):
            neighbor = neighbors_obj[i]
            example_dic_feature = {}
            if neighbor is None:
                example_dic_feature["cur_phase_{0}".format(i)] = none_dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = none_dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = none_dic_feature["lane_num_vehicle"]
            else:
                example_dic_feature["cur_phase_{0}".format(i)] = neighbor.dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = neighbor.dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = neighbor.dic_feature["lane_num_vehicle"]
            old_feature.update(example_dic_feature)
        return old_feature

    def step(self, action):
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

            self._inner_step(action_in_sec)

            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            # average_reward_action = (average_reward_action*i + reward[0])/(i+1)
            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals multi_intersection decided by action {inter_id: phase}
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
                                  "get_vehicle_distance": self.eng.get_vehicle_distance()
                                  }
        # get new measurements
        # for inter in self.list_intersection:
        #     inter.update_current_measurements_map(self.system_states)
        # 添加的代码
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)

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
                inter.dic_feature = self.update_neighbor_feature(neighbor_inters, deepcopy(inter.dic_feature))

        # 前面的信息都更新完毕后再计算压力
        for inter in self.list_intersection:
            neighbor_objs = {}
            for one_neighbor_id in inter.neighbor_ENWS:
                if one_neighbor_id is not None:
                    neighbor_objs[one_neighbor_id] = self.list_intersection[self.id_to_index[one_neighbor_id]]
            inter.update_cmp_pressure_xz(neighbor_objs)

        # self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        # self.log_phase()

    def load_roadnet(self, roadnetFile=None):
        print("Start load roadnet")
        # start_time = time.time()
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        # print("successfully load roadnet:{0}, time: {1}".format(roadnetFile, time.time() - start_time))

    def load_flow(self, flowFile=None):
        print("Start load flowFile")
        # start_time = time.time()
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        # print("successfully load flowFile: {0}, time: {1}".format(flowFile, time.time() - start_time))

    def _check_episode_done(self, list_state):
        # ======== to implement ========
        return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in
                      self.list_intersection]
        done = self._check_episode_done(list_state)
        # print(list_state)
        return list_state, done

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0", "1", "2", "3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in
                       self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

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
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")
        for t in process_list:
            t.join()
        print("end join")

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

        self.eng.print_log(os.path.join(self.path_to_log, self.dic_traffic_env_conf["ROADNET_FILE"]),
                           os.path.join(self.path_to_log, "replay_1_1.txt"))

        # print("log files:", os.path.join("data", "frontend", "roadnet_1_1_test.json"))

    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()

    def log_hidden_state(self, hidden_states):
        path_to_log_file = os.path.join(self.path_to_log, "hidden_states.pkl")
        with open(path_to_log_file, "wb") as f:
            pickle.dump(hidden_states, f)

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str

        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(
                    inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt" % dic_lane_map[lane]),
                                "a"))

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str

        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(
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

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None, }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys():  # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road + "_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']:  # use geo-distance
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
                else:  # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']:  ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)

                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

        return traffic_light_node_dict

    def _adjacency_extraction_lane(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,
                                                            "total_lane_num": None, 'adjacency_matrix_lane': None,
                                                            "lane_id_to_index": None,
                                                            "lane_ids_in_intersction": []
                                                            }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            top_k_lane = self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]
            total_inter_num = len(traffic_light_node_dict.keys())

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()

            # set inter id to index dict
            inter_id_to_index = {}
            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            # set the neighbor_ENWS nodes and entring_lane_ENWS for intersections
            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys():  # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key
                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road + "_{0}".format(k))
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])

            lane_id_dict = self.roadnet.net_lane_dict
            total_lane_num = len(lane_id_dict.keys())

            # output an adjacentcy matrix for all the intersections
            # each row stands for a lane id,
            # each column is a list with two elements: first is the lane's entering_lane_LSR, second is the lane's leaving_lane_LSR
            def _get_top_k_lane(lane_id_list, top_k_input):
                top_k_lane_indexes = []
                for i in range(top_k_input):
                    lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                    top_k_lane_indexes.append(lane_id)
                return top_k_lane_indexes

            adjacency_matrix_lane = {}
            for i in lane_id_dict.keys():  # Todo lane_ids should be in an order
                adjacency_matrix_lane[i] = [_get_top_k_lane(lane_id_dict[i]['input_lanes'], top_k_lane),
                                            _get_top_k_lane(lane_id_dict[i]['output_lanes'], top_k_lane)]

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']:  # use geo-distance
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
                else:  # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']:  ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)

                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num
                traffic_light_node_dict[i]['total_lane_num'] = total_lane_num
                traffic_light_node_dict[i]['adjacency_matrix_lane'] = adjacency_matrix_lane

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a - b) ** 2))

    def end_sumo(self):
        print("anon process end")
        pass





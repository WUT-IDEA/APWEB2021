

def prepare_traffice_files(data_type, road_net):
    traffic_file_list = []
    if data_type == 'syn': # syn flow data
        if road_net == '1_6':
            # traffic_file_list = ["anon_1_6_700_0.6_synthetic.json",
            #                      "anon_1_6_300_0.3_synthetic.json",
            #                      "anon_1_6_300_0.6_synthetic.json",
            #                      "anon_1_6_700_0.3_synthetic.json"]
            traffic_file_list = ["anon_1_6_700_0.6_synthetic.json"]
            road_path = 'syn/road_lsr/1_6/'
            lane = 3
        if road_net == '3_3':
            traffic_file_list = ["anon_3_3_700_0.3_synthetic.json",
                                 "anon_3_3_700_0.6_synthetic.json"]
            road_path = 'syn/road_lsr/3_3/'
            lane = 3
        if road_net == '1_10':
            traffic_file_list = ["anon_1_10_700_0.3_synthetic.json",
                                 "anon_1_10_700_0.6_synthetic.json"]
            road_path = 'syn/road_lsr/1_10/'
            lane = 3
        if road_net == '1_20':
            traffic_file_list = ["anon_1_20_700_0.6_synthetic.json",
                                 "anon_1_20_700_0.3_synthetic.json"]
            road_path = 'syn/road_lsr/1_20/'
            lane = 3
        if road_net == '4_4':
            traffic_file_list = ["anon_4_4_750_0.6_synthetic.json",
                                 "anon_4_4_700_0.6_synthetic.json",
                                 "anon_4_4_700_0.3_synthetic.json",
                                 "anon_4_4_750_0.3_synthetic.json"]
            road_path = 'syn/road_lsr/4_4/'
            lane = 3
        if road_net == '6_6':
            traffic_file_list = ["anon_6_6_300_0.3_bi.json",
                                 "anon_6_6_300_0.3_uni.json"]
            road_path = 'syn/road_lsr/6_6/'
            lane = 3
    elif data_type == 'real':  # real flow data
        if road_net == "16_1":
            traffic_file_list = ["anon_16_1_300_newyork_real_3.json",
                                 "anon_16_1_300_newyork_real_2.json",
                                 "anon_16_1_300_newyork_real_1.json",
                                 "anon_16_1_300_newyork_real_4.json"]
            road_path = 'real/16_1/'
            lane = 3
        if road_net == "3_4":
            traffic_file_list = ["anon_3_4_jinan_real.json",
                                 "anon_3_4_jinan_real_2000.json",
                                 "anon_3_4_jinan_real_2500.json"]
            road_path = 'real/3_3/'
            lane = 3
        if road_net == "4_4":
            traffic_file_list = ["anon_4_4_hangzhou_real.json",
                                 "anon_4_4_hangzhou_real_5734.json",
                                 "anon_4_4_hangzhou_real_5816.json"]
            road_path = 'real/4_4/'
            lane = 3
        if road_net == "28_7":
            traffic_file_list = ["anon_28_7_newyork_real_double.json",
                                 "anon_28_7_newyork_real_triple.json"]
            road_path = 'real/28_7/'
            lane = 3
        if road_net == "16_3":
            traffic_file_list = ["anon_16_3_newyork_real.json"]
            road_path = 'real/16_3/'
            lane = 3
    else:
        print("[RUN] not found your road_net args")
        exit(-1)
    return traffic_file_list, road_path, lane


def get_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end])
    elif "synthetic" in traffic_file:
        traffic_file_list = traffic_file.split("-")
        volume_list = []
        for i in range(2, 6):
            volume_list.append(int(traffic_file_list[i][2:]))

        vol = min(max(volume_list[0:2]), max(volume_list[2:]))

        return int(vol/100)*100
    elif "anon" in traffic_file:
        return int(traffic_file.split('_')[3].split('.')[0])

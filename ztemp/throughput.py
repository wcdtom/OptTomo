import numpy as np
import networkx as nx
import pulp
import network_builder.network_topology as tt
import random

LARGE_VALUE = 10


def is_flow_zero(flow_s_t, node, node_neighbor, path_len_dict):
    slack = 4
    flow_s = flow_s_t[0]
    flow_t = flow_s_t[1]

    start_from_s_dict = path_len_dict[flow_s]
    len_flow_s_node = start_from_s_dict[node]
    start_from_node_neighbor_dict = path_len_dict[node_neighbor]
    len_node_neighbor_flow_t = start_from_node_neighbor_dict[flow_t]

    len_flow_s_flow_t = start_from_s_dict[flow_t]

    if (len_flow_s_node + 1 + len_node_neighbor_flow_t) >= (len_flow_s_flow_t + slack):
        return True
    else:
        return False


def is_neighbor(a, b, path_length_dict):
    a_neighbors = path_length_dict[a]
    flag = False
    for item in a_neighbors:
        if item == b:
            flag = True
    return flag


def reciprocal_random_graph(topology, capacity):
    if capacity is None:
        capacity = nx.get_edge_attributes(topology, 'weight')
        edge_list = []
        for _, v in capacity.items():
            edge_list.append(v)
        if capacity == {}:
            print("please give a fixed capacity or give capacity in topology graph")
            return
    else:
        edge_list = [capacity for _ in range(len(list(nx.edges(topology))))]
    random_graph = nx.Graph()
    nodes = list(nx.nodes(topology))
    random_graph.add_nodes_from(nodes)
    total_degree = len(edge_list) * 2
    first_degree = total_degree // len(nodes)
    last_batch = total_degree - first_degree * len(nodes)
    node_degree = []
    p = 0
    for node in nodes:
        if p < last_batch:
            node_degree.append([node, first_degree + 1])
        else:
            node_degree.append([node, first_degree])
        p += 1
    for edge_capacity in edge_list:
        remove_node = []
        for nd in node_degree:
            if nd[1] == 0:
                remove_node.append(nd)
        for rn in remove_node:
            node_degree.remove(rn)
        flag = 1
        while flag == 1:
            if len(node_degree) >= 2:
                tmp_list = [x for x in range(len(node_degree))]
                idx_pair = random.sample(tmp_list, 2)
                if node_degree[idx_pair[0]][1] == 0 or node_degree[idx_pair[1]][1] == 0:
                    flag = 1
                else:
                    node_degree[idx_pair[0]][1] -= 1
                    node_degree[idx_pair[1]][1] -= 1
                    random_graph.add_edge(node_degree[idx_pair[0]][0],
                                          node_degree[idx_pair[1]][0],
                                          weight=edge_capacity)
                    flag = 0
            else:
                flag = 0

    return random_graph


# probability [0, 1]
def build_lp_pulp(topology, traffic_matrix, probability, equal_capacity):
    neighbor_dict = {}
    path_length_dict = dict(nx.all_pairs_shortest_path_length(topology))
    nodes = list(nx.nodes(topology))
    for node in nodes:
        neighbors_list, _ = tt.get_k_neighbors(topology=topology,
                                               path_length_dict=path_length_dict,
                                               node_id=node,
                                               K=1)
        node_index = str(int(node))
        for neighbor_node in neighbors_list:
            neighbor_dict.setdefault(node_index, []).append(neighbor_node)

    num_edge = 0
    for node in nodes:
        node_index = str(int(node))
        node_neighbor_list = neighbor_dict[node_index]
        num_edge += len(node_neighbor_list)

    num_flow = 0
    curf_id = 0
    commodity_id = 0
    flow_dict = {}
    for node_i in nodes:
        for node_j in nodes:
            if traffic_matrix[node_i, node_j] > 0:
                flow_index = str(int(num_flow))
                flow_dict.setdefault(flow_index, []).append(node_i)
                flow_dict.setdefault(flow_index, []).append(node_j)
                num_flow += 1
                curf_id += 1
            if traffic_matrix[node_i, node_j] != 0:
                commodity_id += 1

    a_list = [_ for _ in range(curf_id)]
    random.shuffle(a_list)
    choose_count = round(curf_id * probability)

    model = pulp.LpProblem(name="throughput", sense=pulp.LpMaximize)
    k = pulp.LpVariable(name="k", lowBound=0)
    objective_function = k
    model += objective_function

    # add c0
    fid = 0
    for node_s in nodes:
        for node_t in nodes:
            if traffic_matrix[node_s, node_t] > 0:
                c0_var = []
                constraint_name = 'c0_' + str(fid)
                node_s_index = str(int(node_s))
                temp_neighbor_list = neighbor_dict[node_s_index]
                for s_neighbor in temp_neighbor_list:
                    if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                        node=node_s,
                                        node_neighbor=s_neighbor,
                                        path_len_dict=path_length_dict):
                        var_name = "f_" + str(fid) + "_" + str(node_s) + "_" + str(s_neighbor)
                        c0_var.append(var_name)
                if c0_var == []:
                    pass
                else:
                    c0 = {i: pulp.LpVariable(name=c0_var[i], lowBound=0) for i in range(len(c0_var))}
                    if a_list.index(fid) < choose_count and probability < 100:
                        last = exec(str(LARGE_VALUE*traffic_matrix[node_s, node_t]) + " * k")
                        model += (pulp.lpSum(c0.values()) >= last, constraint_name)
                    else:
                        last = exec(str(traffic_matrix[node_s, node_t]) + " * k")
                        model += (pulp.lpSum(c0.values()) >= last, constraint_name)
                fid += 1
    print('c0 constrains are done')

    # add c1
    capacity = nx.get_edge_attributes(topology, 'weight')

    for node_m in nodes:
        temp_neighbor_list = neighbor_dict[str(int(node_m))]
        for m_neighbor in temp_neighbor_list:
            c1_var = []
            constraint_name = 'c1_' + str(node_m) + '_' + str(m_neighbor)
            for i in range(commodity_id):
                if not is_flow_zero(flow_s_t=flow_dict[str(int(i))],
                                    node=node_m,
                                    node_neighbor=m_neighbor,
                                    path_len_dict=path_length_dict):
                    var_name = "f_" + str(i) + "_" + str(node_m) + "_" + str(m_neighbor)
                    c1_var.append(var_name)
            if c1_var == []:
                pass
            else:
                try:
                    tmp_capacity = capacity[(node_m, m_neighbor)]
                except KeyError:
                    try:
                        tmp_capacity = capacity[(m_neighbor, node_m)]
                    except KeyError:
                        tmp_capacity = None
                    else:
                        tmp_capacity = capacity[(m_neighbor, node_m)]
                else:
                    tmp_capacity = capacity[(node_m, m_neighbor)]
                if tmp_capacity is not None:
                    c1 = {j: pulp.LpVariable(name=c1_var[j], lowBound=0) for j in range(len(c1_var))}
                    model += (pulp.lpSum(c1.values()) <= exec(str(tmp_capacity)), constraint_name)
    print('c1 constrains are done')

    # add c2
    fid = 0
    for node_a in nodes:
        for node_b in nodes:
            if traffic_matrix[node_a, node_b] > 0:
                for node_c in nodes:
                    if node_c == node_a:
                        c21_var = []
                        constraint_name = "c2_" + str(fid) + "_" +str(node_c) + "_1"
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=node_c,
                                                node_neighbor=c_neighbor,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(node_c) + "_" + str(c_neighbor)
                                c21_var.append(var_name)
                        if c21_var == []:
                            pass
                        else:
                            c21 = {k1: pulp.LpVariable(name=c21_var[k1], lowBound=0) for k1 in range(len(c21_var))}
                            model += (pulp.lpSum(c21.values()) <=
                                      exec(str(traffic_matrix[node_a, node_b] * LARGE_VALUE)),
                                      constraint_name)

                        c22_var = []
                        constraint_name = "c2_" + str(fid) + "_" + str(node_c) + "_2"
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=c_neighbor,
                                                node_neighbor=node_c,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(c_neighbor) + "_" + str(node_c)
                                c22_var.append(var_name)
                        if c22_var == []:
                            pass
                        else:
                            c22 = {k2: pulp.LpVariable(name=c22_var[k2], lowBound=0) for k2 in range(len(c22_var))}
                            model += (pulp.lpSum(c22.values()) == 0, constraint_name)
                    elif node_c == node_b:
                        pass
                    else:
                        c23_var = []
                        constraint_name = "c2_" + str(fid) + "_" + str(node_c) + "_3"
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=node_c,
                                                node_neighbor=c_neighbor,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(node_c) + "_" + str(c_neighbor)
                                c23_var.append(var_name)
                        if c23_var == []:
                            pass
                        else:
                            c23 = {k3: pulp.LpVariable(name=c23_var[k3], lowBound=0) for k3 in range(len(c23_var))}

                        c24_var = []
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=c_neighbor,
                                                node_neighbor=node_c,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(c_neighbor) + "_" + str(node_c)
                                c24_var.append(var_name)
                        if c24_var == []:
                            pass
                        else:
                            c24 = {k4: pulp.LpVariable(name=c24_var[k4], lowBound=0) for k4 in range(len(c24_var))}
                            model += (pulp.lpSum(c23.values()) == pulp.lpSum(c24.values()), constraint_name)
                fid += 1
    print('c2 constrains are done')
    return model


def build_lp_glpk(topology, traffic_matrix, probability, equal_capacity, glpk_file):
    global_var = []
    neighbor_dict = {}
    path_length_dict = dict(nx.all_pairs_shortest_path_length(topology))
    nodes = list(nx.nodes(topology))
    for node in nodes:
        neighbors_list, _ = tt.get_k_neighbors(topology=topology,
                                               path_length_dict=path_length_dict,
                                               node_id=node,
                                               K=1)
        node_index = str(int(node))
        for neighbor_node in neighbors_list:
            neighbor_dict.setdefault(node_index, []).append(neighbor_node)

    num_edge = 0
    for node in nodes:
        node_index = str(int(node))
        node_neighbor_list = neighbor_dict[node_index]
        num_edge += len(node_neighbor_list)

    num_flow = 0
    curf_id = 0
    commodity_id = 0
    flow_dict = {}
    for node_i in nodes:
        for node_j in nodes:
            if traffic_matrix[node_i, node_j] > 0:
                flow_index = str(int(num_flow))
                flow_dict.setdefault(flow_index, []).append(node_i)
                flow_dict.setdefault(flow_index, []).append(node_j)
                num_flow += 1
                curf_id += 1
            if traffic_matrix[node_i, node_j] != 0:
                commodity_id += 1

    a_list = list(range(curf_id))
    random.shuffle(a_list)
    choose_count = round(curf_id * probability)

    # maximize z: 3 * x1 + x2 + 2 * x3;
    objective_str = 'maximize K: k;'

    # add c0
    fid = 0
    c0 = []
    for node_s in nodes:
        for node_t in nodes:
            if traffic_matrix[node_s, node_t] > 0:
                constraint_str = ""
                constraint_name = 's.t. c0_' + str(fid) + ': 0 '
                node_s_index = str(int(node_s))
                temp_neighbor_list = neighbor_dict[node_s_index]
                for s_neighbor in temp_neighbor_list:
                    if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                        node=node_s,
                                        node_neighbor=s_neighbor,
                                        path_len_dict=path_length_dict):
                        var_name = "f_" + str(fid) + "_" + str(node_s) + "_" + str(s_neighbor)
                        global_var.append(var_name)
                        constraint_str += "- " + var_name + " "
                if constraint_str == "":
                    pass
                else:
                    if a_list.index(fid) < choose_count and probability < 100:
                        constraint_str += "+ " + str(10*traffic_matrix[node_s, node_t]) + " * k <= 0;"
                    else:
                        constraint_str += "+ " + str(traffic_matrix[node_s, node_t]) + " * k <= 0;"
                    c0.append(constraint_name + constraint_str)
                fid += 1
    print('c0 constrains are done')

    # add c1
    c1 = []
    capacity = nx.get_edge_attributes(topology, 'weight')
    for node_m in nodes:
        temp_neighbor_list = neighbor_dict[str(int(node_m))]
        for m_neighbor in temp_neighbor_list:
            constraint_str = ""
            constraint_name = 's.t. c1_' + str(node_m) + '_' + str(m_neighbor) + ': '
            for i in range(commodity_id):
                if not is_flow_zero(flow_s_t=flow_dict[str(int(i))],
                                    node=node_m,
                                    node_neighbor=m_neighbor,
                                    path_len_dict=path_length_dict):
                    var_name = "f_" + str(i) + "_" + str(node_m) + "_" + str(m_neighbor)
                    global_var.append(var_name)
                    constraint_str += var_name + " + "
            if constraint_str == "":
                pass
            else:
                try:
                    tmp_capacity = capacity[(node_m, m_neighbor)]
                except KeyError:
                    try:
                        tmp_capacity = capacity[(m_neighbor, node_m)]
                    except KeyError:
                        tmp_capacity = None
                    else:
                        tmp_capacity = capacity[(m_neighbor, node_m)]
                else:
                    tmp_capacity = capacity[(node_m, m_neighbor)]
                if tmp_capacity is not None:
                    constraint_str += "0 <= " + str(tmp_capacity)
                    c1.append(constraint_name + constraint_str)
    print('c1 constrains are done')

    # add c2
    fid = 0
    c2 = []
    for node_a in nodes:
        for node_b in nodes:
            if traffic_matrix[node_a, node_b] > 0:
                for node_c in nodes:
                    if node_c == node_a:
                        constraint_str = ""
                        constraint_name = "s.t. c2_" + str(fid) + "_" +str(node_c) + "_1: "
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=node_c,
                                                node_neighbor=c_neighbor,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(node_c) + "_" + str(c_neighbor)
                                global_var.append(var_name)
                                constraint_str += var_name + " + "
                        if constraint_str == "":
                            pass
                        else:
                            constraint_str += "0 <= " + str(traffic_matrix[node_a, node_b] * LARGE_VALUE) + ';'
                            c2.append(constraint_name + constraint_str)

                        constraint_str = ""
                        constraint_name = "s.t. c2_" + str(fid) + "_" + str(node_c) + "_2: "
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=c_neighbor,
                                                node_neighbor=node_c,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(c_neighbor) + "_" + str(node_c)
                                global_var.append(var_name)
                                constraint_str += var_name + " + "
                        constraint_str += "0 == 0;"
                        if constraint_str == "0 == 0;":
                            pass
                        else:
                            c2.append(constraint_name + constraint_str)
                    elif node_c == node_b:
                        pass
                    else:
                        constraint_str = ""
                        constraint_name = "s.t. c2_" + str(fid) + "_" + str(node_c) + "_3: "
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=node_c,
                                                node_neighbor=c_neighbor,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(node_c) + "_" + str(c_neighbor)
                                global_var.append(var_name)
                                constraint_str += var_name + " + "
                        constraint_str += "0"
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=c_neighbor,
                                                node_neighbor=node_c,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(c_neighbor) + "_" + str(node_c)
                                global_var.append(var_name)
                                constraint_str += " - " + var_name
                        constraint_str += " == 0;"
                        if constraint_str == "0 == 0;":
                            pass
                        else:
                            c2.append(constraint_name + constraint_str)
                fid += 1
    print('c2 constrains are done')

    global_var = list(set(global_var))

    with open(glpk_file, 'w') as f:
        f.write('/* Vars */' + '\n')
        f.write('var k >= 0;' + '\n')
        for var in global_var:
            var_str = 'var ' + var + ' >= 0;'
            f.write(var_str + '\n')
        f.write('\n')
        f.write('/* Objective function */' + '\n')
        f.write(objective_str + '\n')
        f.write('\n')
        f.write('/* Constraints 0 */' + '\n')
        for cons0 in c0:
            f.write(cons0 + '\n')
        f.write('\n')
        f.write('/* Constraints 1 */' + '\n')
        for cons1 in c1:
            f.write(cons1 + '\n')
        f.write('\n')
        f.write('/* Constraints 2 */' + '\n')
        for cons2 in c2:
            f.write(cons2 + '\n')
        f.write('end;')
    print('GLPK model file is done')


def build_lp_gurobi(topology, traffic_matrix, probability, equal_capacity, gurobi_file):
    global_var = []
    neighbor_dict = {}
    path_length_dict = dict(nx.all_pairs_shortest_path_length(topology))
    nodes = list(nx.nodes(topology))
    for node in nodes:
        neighbors_list, _ = tt.get_k_neighbors(topology=topology,
                                               path_length_dict=path_length_dict,
                                               node_id=node,
                                               K=1)
        node_index = str(int(node))
        for neighbor_node in neighbors_list:
            neighbor_dict.setdefault(node_index, []).append(neighbor_node)

    num_edge = 0
    for node in nodes:
        node_index = str(int(node))
        node_neighbor_list = neighbor_dict[node_index]
        num_edge += len(node_neighbor_list)

    num_flow = 0
    curf_id = 0
    commodity_id = 0
    flow_dict = {}
    for node_i in nodes:
        for node_j in nodes:
            if traffic_matrix[node_i, node_j] > 0:
                flow_index = str(int(num_flow))
                flow_dict.setdefault(flow_index, []).append(node_i)
                flow_dict.setdefault(flow_index, []).append(node_j)
                num_flow += 1
                curf_id += 1
            if traffic_matrix[node_i, node_j] != 0:
                commodity_id += 1

    a_list = list(range(curf_id))
    random.shuffle(a_list)
    choose_count = round(curf_id * probability)

    # maximize z: 3 * x1 + x2 + 2 * x3;
    objective_str = 'k'

    # add c0
    fid = 0
    c0 = []
    for node_s in nodes:
        for node_t in nodes:
            if traffic_matrix[node_s, node_t] > 0:
                constraint_str = ""
                constraint_name = 'c0_' + str(fid) + ': 0 '
                node_s_index = str(int(node_s))
                temp_neighbor_list = neighbor_dict[node_s_index]
                for s_neighbor in temp_neighbor_list:
                    if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                        node=node_s,
                                        node_neighbor=s_neighbor,
                                        path_len_dict=path_length_dict):
                        var_name = "f_" + str(fid) + "_" + str(node_s) + "_" + str(s_neighbor)
                        global_var.append(var_name)
                        constraint_str += "- " + var_name + " "
                if constraint_str == "":
                    pass
                else:
                    if a_list.index(fid) < choose_count and probability < 100:
                        constraint_str += "+ " + str(10 * traffic_matrix[node_s, node_t]) + " k <= 0"
                    else:
                        constraint_str += "+ " + str(traffic_matrix[node_s, node_t]) + " k <= 0"
                    c0.append(constraint_name + constraint_str)
                fid += 1
    print('c0 constrains are done')

    # add c1
    c1 = []
    capacity = nx.get_edge_attributes(topology, 'weight')
    for node_m in nodes:
        temp_neighbor_list = neighbor_dict[str(int(node_m))]
        for m_neighbor in temp_neighbor_list:
            constraint_str = ""
            constraint_name = 'c1_' + str(node_m) + '_' + str(m_neighbor) + ': '
            for i in range(commodity_id):
                if not is_flow_zero(flow_s_t=flow_dict[str(int(i))],
                                    node=node_m,
                                    node_neighbor=m_neighbor,
                                    path_len_dict=path_length_dict):
                    var_name = "f_" + str(i) + "_" + str(node_m) + "_" + str(m_neighbor)
                    global_var.append(var_name)
                    constraint_str += var_name + " + "
            if constraint_str == "":
                pass
            else:
                if capacity == {}:
                    tmp_capacity = equal_capacity
                else:
                    try:
                        tmp_capacity = capacity[(node_m, m_neighbor)]
                    except KeyError:
                        try:
                            tmp_capacity = capacity[(m_neighbor, node_m)]
                        except KeyError:
                            tmp_capacity = None
                        else:
                            tmp_capacity = capacity[(m_neighbor, node_m)]
                    else:
                        tmp_capacity = capacity[(node_m, m_neighbor)]
                if tmp_capacity is not None:
                    constraint_str += "0 <= " + str(tmp_capacity)
                    c1.append(constraint_name + constraint_str)
    print('c1 constrains are done')

    # add c2
    fid = 0
    c2 = []
    for node_a in nodes:
        for node_b in nodes:
            if traffic_matrix[node_a, node_b] > 0:
                for node_c in nodes:
                    if node_c == node_a:
                        constraint_str = ""
                        constraint_name = "c2_" + str(fid) + "_" +str(node_c) + "_1: "
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=node_c,
                                                node_neighbor=c_neighbor,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(node_c) + "_" + str(c_neighbor)
                                global_var.append(var_name)
                                constraint_str += var_name + " + "
                        if constraint_str == "":
                            pass
                        else:
                            constraint_str += "0 <= " + str(traffic_matrix[node_a, node_b] * LARGE_VALUE)
                            c2.append(constraint_name + constraint_str)

                        constraint_str = ""
                        constraint_name = "c2_" + str(fid) + "_" + str(node_c) + "_2: "
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=c_neighbor,
                                                node_neighbor=node_c,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(c_neighbor) + "_" + str(node_c)
                                global_var.append(var_name)
                                constraint_str += var_name + " + "
                        constraint_str += "0 = 0"
                        if constraint_str == "0 = 0":
                            pass
                        else:
                            c2.append(constraint_name + constraint_str)
                    elif node_c == node_b:
                        pass
                    else:
                        constraint_str = ""
                        constraint_name = "c2_" + str(fid) + "_" + str(node_c) + "_3: "
                        temp_neighbor_list = neighbor_dict[str(int(node_c))]
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=node_c,
                                                node_neighbor=c_neighbor,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(node_c) + "_" + str(c_neighbor)
                                global_var.append(var_name)
                                constraint_str += var_name + " + "
                        constraint_str += "0"
                        for c_neighbor in temp_neighbor_list:
                            if not is_flow_zero(flow_s_t=flow_dict[str(int(fid))],
                                                node=c_neighbor,
                                                node_neighbor=node_c,
                                                path_len_dict=path_length_dict):
                                var_name = "f_" + str(fid) + "_" + str(c_neighbor) + "_" + str(node_c)
                                global_var.append(var_name)
                                constraint_str += " - " + var_name
                        constraint_str += " = 0"
                        if constraint_str == "0 = 0":
                            pass
                        else:
                            c2.append(constraint_name + constraint_str)
                fid += 1
    print('c2 constrains are done')

    global_var = list(set(global_var))

    with open(gurobi_file, 'w') as f:
        f.write('Maximize' + '\n')
        f.write(' ' + objective_str + '\n')
        f.write('Subject To' + '\n')
        for cons0 in c0:
            f.write(' ' + cons0 + '\n')
        for cons1 in c1:
            f.write(' ' + cons1 + '\n')
        for cons2 in c2:
            f.write(' ' + cons2 + '\n')
        f.write('Bounds' + '\n')
        for var in global_var:
            var_str = var + ' >= 0;'
            f.write(' ' + var_str + '\n')
        f.write('End')
    print('Gurobi model file is done')


def build_lp_cplex(topology, traffic_matrix, probability, equal_capacity):
    pass

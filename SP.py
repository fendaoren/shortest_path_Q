import networkx as nx
import numpy as np
from gurobipy import *
import copy
import matplotlib.pyplot as plt
import re
from set import generate_graph


class Data:
    customerNum = 0
    nodeNum     = 0
    vehicleNum  = 0
    capacity    = 0
    cor_X       = []
    cor_Y       = []
    demand      = []
    serviceTime = []
    readyTime   = []
    dueTime     = []
    disMatrix   = [[]]


# function to read data from .txt files
def readData(data, path, nodeNum):
    data.customerNum = nodeNum
    data.nodeNum = nodeNum
    f = open(path, 'r')
    lines = f.readlines()
    count = 0
    # read the info
    for line in lines:
        count = count + 1
        if (count == 5):
            line = line[:-1].strip()
            str = re.split(r" +", line)
            data.vehicleNum = int(str[0])
            data.capacity = float(str[1])
        elif (count >= 10 and count <= 10 + nodeNum - 1):
            line = line[:-1]
            str = re.split(r" +", line)
            data.cor_X.append(float(str[2]))
            data.cor_Y.append(float(str[3]))
            data.demand.append(float(str[4]))
            data.readyTime.append(float(str[5]))
            data.dueTime.append(float(str[6]))
            data.serviceTime.append(float(str[7]))

    # compute the distance matrix
    data.disMatrix = [([0] * data.nodeNum) for p in range(data.nodeNum)]  # 初始化距离矩阵的维度,防止浅拷贝
    # data.disMatrix = [[0] * nodeNum] * nodeNum]; 这个是浅拷贝，容易重复
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            temp = (data.cor_X[i] - data.cor_X[j]) ** 2 + (data.cor_Y[i] - data.cor_Y[j]) ** 2
            data.disMatrix[i][j] = math.sqrt(temp)
            temp = 0

    return data

data = Data()


BigM = 10000000 ##定义一个极大值


node_num = 100
seed = 8
Q = 3 #流量限制
pos, arcs, costs = generate_graph(seed,node_num)
task = [(79, 35), (37, 35), (64, 34), (35, 80), (52, 31), (46, 31), (31, 64), (95, 58), (31, 29), (76, 67), (25, 47), (63, 99), (13, 14), (49, 10), (83, 21), (99, 72), (54, 5), (17, 2), (78, 24), (87, 33)]

_task = {}
for i in range(len(task)):
    _task[i] = list(task[i])
print(_task)



# points = [(0, 0)]
# points += [(pos[i][0], pos[i][1]) for i in pos]


# Dictionary of Manhattan distance between each pair of points
dist = {
    arcs[i]: costs[i]
    for i in range(len(arcs))
    if arcs[i][0] != arcs[i][1]
}

# Create graph
# source = "A"
# target = "G"

G = nx.DiGraph()
for k, v in dist.items():
    i = k[0]
    j = k[1]
    G.add_edge(i, j, dist=v)
# Draw the graph with the node labels and edge costs
nx.draw(G, pos, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'cost'))
# Show the graph
plt.show()

A=np.array(nx.adjacency_matrix(G).todense())

index_ = {}
index_i = 0
for i in pos:
    index_[index_i] = i
    index_i += 1
    data.cor_X.append(pos[i][0])
    data.cor_Y.append(pos[i][1])
print(index_)


data.nodeNum = len(pos)
# compute the distance matrix
data.disMatrix = [([0] * data.nodeNum) for p in range(data.nodeNum)]  # 初始化距离矩阵的维度,防止浅拷贝
# data.disMatrix = [[0] * nodeNum] * nodeNum]; 这个是浅拷贝，容易重复

for i in range(0, data.nodeNum):

    for j in range(0, data.nodeNum):
        if A[i,j] ==1:
            temp = (data.cor_X[i] - data.cor_X[j]) ** 2 + (data.cor_Y[i] - data.cor_Y[j]) ** 2
            data.disMatrix[i][j] = math.sqrt(temp)
            temp = 0
        if A[i,j] ==0:
            data.disMatrix[i][j]= BigM



def solve_model(model,node_list,vehNum,S_T):
    nodeNum = len(node_list)
    x = {}
    s = {}
    node_allow_ = copy.deepcopy(node_list)

    node_dict = {}
    start_end = {}
    node_other = copy.deepcopy(node_list)
    print(node_other)
    for k in range(vehNum, vehNum + data.vehicleNum):
        st = k - vehNum
        node_allow = copy.deepcopy(node_list)
        node_allow.remove(S_T[st][0])

        node_end = copy.deepcopy(node_allow)
        node_allow.remove(S_T[st][1])

        if S_T[st][0] in node_other:
            node_other.remove(S_T[st][0])
        if S_T[st][1] in node_other:
            node_other.remove(S_T[st][1])

        node_start = copy.deepcopy(node_list)
        node_start.remove(S_T[st][1])
        node_dict[k] = [node_allow,node_start,node_end]
        start_end[k] = [S_T[st][0],S_T[st][1]]



    ##定义字典用来存放决策变量
    for k in range(vehNum, vehNum + data.vehicleNum):
        for i in node_list:
            name = 's_' + str(i) + '_' + str(k)
            s[i, k] = model.addVar(0
                                   , nodeNum
                                   , vtype=GRB.INTEGER
                                   , name=name)  ##定义访问时间为连续变量
            for j in node_list:

                name = 'x_' + str(i) + '_' + str(j) + '_' + str(k)
                x[i, j, k] = model.addVar(0
                                          , 1
                                          , vtype=GRB.BINARY
                                          , name=name)  ##定义是否服务为0-1变量

    ##首先定义一个线性表达式
    obj = LinExpr(0)
    for k in range(vehNum, vehNum + data.vehicleNum):
        for i in node_list:
            for j in node_list:
                # if(i != j):##将目标函数系数与决策变量相乘，并进行连加
                obj.addTerms(data.disMatrix[i][j], x[i, j, k])


    ##将表示目标函数的线性表达式加入模型，并定义为求解最小化问题
    model.setObjective(obj, GRB.MINIMIZE)


    # for k in range(vehNum, vehNum + data.vehicleNum):
    #     for i in node_dict[k][0]:
    #         lhs = LinExpr(0)
    #         for j in node_dict[k][2]:
    #             lhs.addTerms(1, x[i, j, k])
    #         model.addConstr(lhs <= 1, name='customer_visit_departure' + str(k))  # a city might be the departure


    for k in range(vehNum, vehNum + data.vehicleNum):
        lhs = LinExpr(0)
        for j in node_dict[k][2]:
            lhs.addTerms(1, x[start_end[k][0], j, k])
        model.addConstr(lhs == 1, name='start_departure' + str(k))  # a start must be the departure


    for k in range(vehNum, vehNum + data.vehicleNum):
        lhs = LinExpr(0)
        for i in node_list:
            lhs.addTerms(1, x[i, start_end[k][0], k])
        model.addConstr(lhs == 0, name='start_in_no' + str(k))


    # for k in range(vehNum, vehNum + data.vehicleNum):
    #     for j in node_dict[k][0]:
    #         lhs = LinExpr(0)
    #         for i in node_dict[k][1]:
    #             lhs.addTerms(1, x[i, j, k])
    #         model.addConstr(lhs <= 1, name='customer_visit_in' + str(j))  # every city might be the destination


    for k in range(vehNum, vehNum + data.vehicleNum):
        lhs = LinExpr(0)
        for i in node_dict[k][1]:
            lhs.addTerms(1, x[i, start_end[k][1], k])
        model.addConstr(lhs == 1, name='end_in' + str(j))  # every city must be the destination


    for k in range(vehNum, vehNum + data.vehicleNum):
        lhs = LinExpr(0)
        for j in node_list:
            lhs.addTerms(1, x[start_end[k][1], j, k])
        model.addConstr(lhs == 0, name='end_departure_no' + str(k))


    ###边数限制###

    for j in node_list:
        lhs = LinExpr(0)
        for k in range(vehNum, vehNum + data.vehicleNum):
            for i in node_list:
                if i != j:
                    lhs.addTerms(1, x[i, j, k])
        model.addConstr(lhs <= Q, name='edge_constraint' + str(k))

    ##避免结果里出现非法路径##

    for j in node_other:
        lhs = LinExpr(0)
        for k in range(vehNum, vehNum + data.vehicleNum):
            for i in node_list:
                if i != j:
                    lhs.addTerms(data.disMatrix[i][j], x[i, j, k])
                    lhs.addTerms(data.disMatrix[j][i], x[j, i, k])

        model.addConstr(lhs <= BigM, name='edge_constraint' + str(k))

    for k in range(vehNum, vehNum + data.vehicleNum):
        for u in node_dict[k][0]:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for j in node_dict[k][2]:
                expr1.addTerms(1, x[u, j, k])
            for l in node_dict[k][1]:
                expr2.addTerms(1, x[l, u, k])
            model.addConstr(expr2 >= expr1, name='flow_conservation_' + str(u))  # eliminate double edges between cities
            expr1.clear()
            expr2.clear()

    for k in range(vehNum, vehNum + data.vehicleNum):
        for i in node_dict[k][1]:
            for j in node_dict[k][2]:
                if (i != j):
                    model.addConstr(s[i, k] - s[j, k] + nodeNum * x[i, j, k] <= nodeNum - 1,
                                    name='sub_cicle_eliminate')

    model.optimize()
    print("\n\n-----optimal value-----")
    print(model.ObjVal)

    for key in x.keys():
        if (x[key].x > 0):
            print(x[key].VarName + ' = ', x[key].x)

    model.write('VRPTW.lp')
    return x
##################
k = 0
model = Model('SP')
# node_list = [0, 1, 2, 3, 4, 5, 6,7]
node_list = []
for i in range(node_num):
    node_list.append(i)


# S_T = {0:[0,4],1:[5,1],2:[4,2],3:[7,1],4:[8,5],5:[9,29]}
S_T = _task
#设置起点和终点

# S_T = {0:[0,6],1:[8,10],2:[7,2],3:[12,1],4:[16,5],5:[8,15]}
data.vehicleNum = 6  ##设置车辆数
x = solve_model(model,node_list,k,S_T)
key_list = []
for key in x.keys():
    if(x[key].x == 1):
        key_list.append([int(a) for a in re.findall(r"-?\d+\d*", x[key].VarName)])
        print(x[key].VarName + ' = ', x[key].x)


city_location_Xlist = data.cor_X
city_location_Ylist = data.cor_Y


def draw(x_list, y_list, k_list):
    print(k_list)
    node_class = {}

    for each in k_list:
        if each[-1] not in node_class:
            node_class[each[-1]] = [each[:-1]]
        else:
            node_class[each[-1]].append(each[:-1])
    node_class_ = copy.deepcopy(node_class)
    print(node_class_)
    route = {}
    for v in node_class:

        while len(node_class[v]):
            if v not in route:
                route[v] = [S_T[v][0]]
                for e in node_class[v]:
                    if e[0] == [S_T[v][0]]:
                        node_class[v].remove(e)

            else:
                for j in range(len(node_class[v])):
                    if node_class[v][j][0] == route[v][-1]:
                        route[v].append(node_class[v][j][1])
                        node_class[v].remove(node_class[v][j])
                        break
        



    # x = {}
    # y = {}

    # for v in route:
    #     x[v] = []
    #     y[v] = []
    #     for e in route[v]:
    #         x[v].append(x_list[e])
    #         y[v].append(y_list[e])

    # ax = plt.gca()
    #
    # for v in route:
    #     ax.scatter(x[v], y[v], s=50, alpha=0.8)
    #     ax.plot(x[v], y[v], linewidth=2.5)
    #
    # for k in range(len(x_list)):
    #      plt.text(city_location_Xlist[k], city_location_Ylist[k] + 0.3, str(k), ha='center', va='bottom', fontsize=10.5)
    #
    # # plt.text(-3, 70, 1, size=15, alpha=1)
    # ax.scatter(x_list[0], y_list[0], c='k')
    # plt.text(20, 20, model.ObjVal, size=10, alpha=1)

    # plt.show()

    return node_class_,route

node_class,route = draw(city_location_Xlist, city_location_Ylist, key_list)

route_set = {}
for d in route:
    route_set[d] = []
    for i in route[d]:
        route_set[d].append(index_[i])


print(route_set)
edge_set = []
for d in node_class:
    for edge in node_class[d]:
        edge_ = (index_[edge[0]], index_[edge[1]])
        edge_set.append(edge_)

whole_dis = 0
illegale = 0
for r in route_set:
    distance = 0
    for i in range(len(route_set[r])-1):
        subdis = data.disMatrix[route_set[r][i]][route_set[r][i+1]]
        distance +=subdis
    if distance >= BigM:
        print("路径不合法")
        print(route_set[r], "路径长度为：", "无穷大")
        illegale += 1

    else:
        print("路径合法")
        print(route_set[r],"路径长度为：", distance)
        whole_dis += distance

print("非法路径数量为",illegale)
print("合法路径总长度为",whole_dis)





G_ = nx.DiGraph()
G_.add_nodes_from(pos.keys())
for (k, v) in edge_set:

    G_.add_edge(k, v)
# Draw the graph with the node labels and edge costs
nx.draw(G_, pos, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(G_, pos, edge_labels=nx.get_edge_attributes(G, 'cost'))
# Show the graph
plt.show()



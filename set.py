# 生成一个50个点构成的连通图但非全连通图
# pos是点的坐标集合，arcs是边集，costs是边的权重集合

# 导入random库，用于生成随机数
import random

# 设置随机数的种子，保证每次运行结果一致


# 定义一个函数，用于生成随机的坐标
def random_pos():

    # 生成一个(0,100)之间的随机整数
    x = random.randint(0, 100)
    # 生成一个(0,100)之间的随机整数
    y = random.randint(0, 100)
    # 返回一个坐标元组
    return (x, y)

# 定义一个函数，用于计算两个坐标之间的距离
def distance(pos1, pos2):
    # 计算两个坐标的横坐标差
    dx = pos1[0] - pos2[0]
    # 计算两个坐标的纵坐标差
    dy = pos1[1] - pos2[1]
    # 计算两个坐标的欧几里得距离
    d = (dx ** 2 + dy ** 2) ** 0.5
    # 返回距离值
    return d

# 定义一个函数，用于生成一个连通图
def generate_graph(seed,n):
    random.seed(seed)
    # 初始化点的坐标集合
    pos = {}
    # 初始化边集
    arcs = []
    # 初始化边的权重集合
    costs = []
    # 循环生成n个点
    for i in range(n):
        # 生成一个随机的坐标
        p = random_pos()
        # 生成一个点的标签，从A到Z，然后从AA到AZ，以此类推
        label = i
        # 将点的标签和坐标加入到点的坐标集合中
        pos[label] = p
        # 如果不是第一个点，那么就随机选择一个之前的点，与当前点连一条边
        if i > 0:
            # 随机选择一个之前的点的索引
            j = random.randint(0, i - 1)
            # 获取之前的点的标签
            prev_label = j
            # 计算当前点和之前的点的距离，作为边的权重
            cost = distance(p, pos[prev_label])
            # 将当前点和之前的点的边加入到边集中，注意要加两次，因为是无向图
            arcs.append((label, prev_label))
            arcs.append((prev_label, label))
            # 将边的权重加入到边的权重集合中，注意也要加两次
            costs.append(cost)
            costs.append(cost)
    # 返回点的坐标集合，边集，和边的权重集合
    return pos, arcs, costs

# # 调用生成连通图的函数，传入参数50，表示生成50个点
# pos, arcs, costs = generate_graph(50)
#
# # 打印点的坐标集合
# print("pos =", pos)
#
# # 打印边集
# print("arcs =", arcs)
#
# # 打印边的权重集合
# print("costs =", costs)

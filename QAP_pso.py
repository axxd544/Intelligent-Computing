import time
import random
from math import *
import numpy as np

# 初始化矩阵维度，距离矩阵，流量矩阵，最大迭代次数，初始种群规模
n = 0
dist_matrix = 0
flow_matrix = 0
generation_max = 0
population = 0
# 初始化惯性权重，个体认知因子，社会认知因子
w = 1
c1 = 0.75
c2 = 0.75
# 初始化全局最有解，全局最优成本
best_group = float('inf')
best_cost = float('inf')
cal_evaluation = []
particle_dict = {}


# 读数据
def read_data(file_dir):
    global n, dist_matrix, flow_matrix
    file = open(file_dir, 'r')
    n = int(file.readline())
    distances = []
    flows = []
    lines = [line for line in file.readlines() if line.strip()]
    for i in range(n):
        distances.append(list(map(int, lines[i].split())))
    for i in range(n):
        flows.append(list(map(int, lines[i + n].split())))
    dist_matrix = np.array(distances)
    flow_matrix = np.array(flows)
    # generation_max = n * 200
    # print(n, dist_matrix, flow_matrix, generation_max)


# 目标函数
def cost(group):
    return sum(np.sum(flow_matrix * dist_matrix[group[:, None], group], 1))


# 更新粒子的速度，a为当前位置，b为最佳位置
def update_v(a, b):
    ans = []
    v = np.eye(len(a))
    for i in range(len(a)):
        a_idx = np.argwhere(a == i)
        b_idx = np.argwhere(b == i)
        tmp = np.eye(len(a))
        tmp[[a_idx, b_idx], :] = tmp[[b_idx, a_idx], :]
        if np.any(tmp != np.eye(len(a))):
            ans.append(tmp)
        v = np.dot(v, tmp)
        a = np.dot(a, tmp)
    return ans


# 移动粒子，并更新全局最优解和成本
def move():
    global best_cost, best_group
    total_evaluations = 0
    for i in range(population):
        v_0 = particle_dict[i]["velocity"]
        v_self = update_v(particle_dict[i]["group"], particle_dict[i]["best"])
        v_group = update_v(particle_dict[i]["group"], best_group)

        r1 = np.random.random(size=[len(v_self), 1])
        r2 = np.random.random(size=[len(v_group), 1])

        v_1 = np.eye(n)
        for j in range(len(v_self)):
            if r1[j] <= c1:
                v_1 = np.dot(v_self[j], v_1)
                total_evaluations += 1
        v_2 = np.eye(n)
        for j in range(len(v_group)):
            if r2[j] <= c2:
                v_2 = np.dot(v_group[j], v_2)
                total_evaluations += 1

        v_0 = v_0 * w
        v = np.dot(v_0, np.dot(v_1, v_2))
        particle_dict[i]["velocity"] = v

        particle_dict[i]["group"] = np.dot(v, particle_dict[i]["group"])
        particle_dict[i]["group"] = np.array([int(j) for j in particle_dict[i]["group"]])

        tmp_cost = cost(particle_dict[i]["group"])
        if tmp_cost < particle_dict[i]["best_cost"]:
            particle_dict[i]["best_cost"] = tmp_cost
            particle_dict[i]["best"] = particle_dict[i]["group"]
            total_evaluations += 1
        if tmp_cost < best_cost:
            best_group = particle_dict[i]["group"]
            best_cost = tmp_cost
            total_evaluations += 1

    return total_evaluations


def pso_run():
    global best_cost, best_group, particle_dict, population, generation_max
    # 初始化种群数量和迭代次数
    population = random.randint(2, 10000)
    generation_max = floor(10000 * n / population)
    print("种群数量：", population)
    print("最大迭代层数:", generation_max)
    evaluation_cnt = 0
    best_group = float('inf')
    best_cost = float('inf')
    particle_dict = {}
    for i in range(population):
        # 随机生成初始的分配方案
        group = np.random.permutation(n)
        particle_dict[i] = {"group": group, "cost": cost(group), "velocity": np.eye(n, dtype=int),
                            "best": group, "best_cost": cost(group)}
        if particle_dict[i]["cost"] < best_cost:
            best_group = particle_dict[i]["group"]
            best_cost = particle_dict[i]["cost"]
            evaluation_cnt += 1

    for i in range(generation_max):
        evaluation_cnt += move()
        if i == generation_max-1:
            print("本轮共迭代{}次，最优位置为{}，最优值为{}".format(i+1, best_group, best_cost))
    print("总评价次数：", evaluation_cnt)
    cal_evaluation.append(evaluation_cnt)

    return best_group, best_cost


if __name__ == '__main__':
    # 最终结果
    cal_group = []
    cal_cost = 2147483647
    cal_time = []
    data_num = 32
    read_data(f'.\data\QAP{data_num}.dat')
    for i in range(10):
        print("----------------第", i, "轮调用------------------")
        time_start = time.time()
        res_group, res_cost = pso_run()
        time_end = time.time()
        print("运行时间：", time_end - time_start, "秒\n")
        if res_cost < cal_cost:
            cal_cost = res_cost
            cal_group = res_group
        cal_time.append(time_end - time_start)

    print("-----------------10轮调用过程中----------------")
    print("平均评价次数：", sum(cal_evaluation) / len(cal_evaluation))
    print("平均运行时间：", sum(cal_time) / len(cal_time))
    print("10轮之后最优解：", cal_cost)
    print("建造方案为：", cal_group)


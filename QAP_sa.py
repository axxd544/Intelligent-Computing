import time
import numpy as np

# 初始化矩阵维度，距离矩阵，流量矩阵，最大迭代层数，退货速率，初始温度
n = 0
dist_matrix = 0
flow_matrix = 0
generation_max = 0
cooling_rate = 0.95
temp = 1000.0
cal_evaluation = []


# 读数据
def read_data(file_dir):
    global n, dist_matrix, flow_matrix, generation_max
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
    generation_max = n * 1000   # 设置最大迭代层数
    # print(n, dist_matrix, flow_matrix, generation_max)


# 目标函数
def cost(group):
    return sum(np.sum(flow_matrix * dist_matrix[group[:, None], group], 1))


# 计算接受概率
def acceptance_probability(old_cost, new_cost, T):
    if new_cost < old_cost:
        return 1.0
    else:
        return np.exp((old_cost - new_cost) / T)


def sa_run():
    global temp
    temp = 1000.0
    it = 0
    evaluation_count = 0
    solution = np.random.permutation(n) # 随机生成初始的分配方案
    while temp > 0.05:
        for i in range(generation_max):
            new_solution = np.copy(solution)
            idx1, idx2 = np.random.randint(0, n, 2)
            new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
            ap = acceptance_probability(cost(solution), cost(new_solution), temp)
            evaluation_count += 1
            if ap >= np.random.rand():  # 根据计算出的概率决定是否接受新解
                solution = new_solution
        temp *= cooling_rate    # 降温
        it += 1
        # if it == generation_max-1:
    print("共迭代{}次，最优位置为{}，最优值为{}".format(it, solution, cost(solution)))

    print("总评价次数：", evaluation_count)
    cal_evaluation.append(evaluation_count)
    return solution, cost(solution)

if __name__ == '__main__':
    cal_group = []
    cal_time = []
    cal_cost = 2147483647
    data_num = 32
    read_data(f'.\data\QAP{data_num}.dat')

    for i in range(10):
        print("------------第", i, "轮调用--------------")
        time_start = time.perf_counter()
        res_group, res_cost = sa_run()
        time_end = time.perf_counter()
        print("运行时间：", time_end - time_start, "秒")
        if res_cost < cal_cost:
            cal_cost = res_cost
            cal_group = res_group
            cal_time.append(time_end - time_start)

    print("-----------------10轮调用过程中----------------")
    print("平均评价次数：", sum(cal_evaluation) / len(cal_evaluation))
    print("平均运行时间：", sum(cal_time) / len(cal_time))
    print("10轮之后最优解：", cal_cost)
    print("建造方案为：", cal_group)


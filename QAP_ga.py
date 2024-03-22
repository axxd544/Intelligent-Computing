import numpy as np
import random
import time
from math import *

res_parent = []
res_count = []
res_time = []
res_cost = 2147483647
res_formula = []

Pm = 0.02       # 突变概率
Pc = 0.5        # 交叉概率
np.random.seed(544)

# 染色体交叉，部分映射交叉
def cros_over(parent_1, parent_2, cros_idx1, cros_idx2):
    if cros_idx1 > cros_idx2:
        cros_idx1, cros_idx2 = cros_idx2, cros_idx1

    n = chromosome_len  # 染色体长度，即工厂的数量

    index = list(range(cros_idx2 + 1, n)) + list(range(0, cros_idx1))
    child_1, child_2 = parent_1.copy(), parent_2.copy()

    child_1["chromosm"][index] = [p for p in parent_2["chromosm"] if
                                  p not in parent_1["chromosm"][cros_idx1:cros_idx2 + 1]]
    child_2["chromosm"][index] = [p for p in parent_1["chromosm"] if
                                  p not in parent_2["chromosm"][cros_idx1:cros_idx2 + 1]]

    child_1["cost"], child_2["cost"] = cost(child_1["chromosm"]), cost(child_2["chromosm"])

    return child_1, child_2

# 染色体变异，两个点位互换
def mutatation_swap(individual, idx_1, idx_2):
    if idx_1 > idx_2:
        idx_1, idx_2 = idx_2, idx_1

    individual["cost"] = mutation_cost(individual["chromosm"], individual["cost"], idx_1, idx_2)
    individual["chromosm"][idx_1], individual["chromosm"][idx_2] = individual["chromosm"][idx_2], \
                                                                   individual["chromosm"][idx_1]

def ga_run_main(res_formula):
    global res_cost
    size_population = random.randint(2, 10000)  # 初始的种群大小
    g_max = ceil(10000 * chromosome_len / size_population)  # 最大迭代层数
    print("种群数量：", size_population)
    print("最大迭代层数:", g_max)
    # 初始种群大小 * 最大迭代层数 = 最大的候选解空间中解的个数
    length = chromosome_len
    size_pop = size_population

    solutions = np.zeros(g_max, dtype=np.int64)
    total_evaluations = 0   # 记录比较 评价的次数

    num_crosses = ceil(size_pop / 2.0 * Pc)
    num_mutations = ceil(size_pop * length * Pm)
    # print (num_crosses)
    # print (num_mutations)

    len_chrom_str = str(length) + 'int'
    # print (len_chrom_str)
    datType = np.dtype([('chromosm', len_chrom_str), ('cost', np.int64)])

    # 生成初始种群
    parents = np.zeros(size_pop, dtype=datType)

    for indiv_soln in parents:
        indiv_soln["chromosm"] = np.random.permutation(length)
        indiv_soln["cost"] = cost(indiv_soln["chromosm"])
        total_evaluations += 1

    parents.sort(order="cost", kind='mergesort')
    # 打印出初始的种群
    # print(parents)

    # 进行迭代
    for g in range(g_max):
        contestant_Idx = np.empty(size_pop, dtype=np.int32)  # 进行比较的样本

        for index in range(size_pop):
            contestant_Idx[index] = np.random.randint(low=0, high=np.random.randint(1, size_pop))

        contestant_pairs = zip(contestant_Idx[0:2 * num_crosses:2], contestant_Idx[1:2 * num_crosses:2])
        # print(contestant_pairs)
        # 对这些染色体进行交叉操作
        children = np.zeros(size_pop, dtype=datType)
        cross_points = np.random.randint(length, size=2 * num_crosses)

        for pairs, index1, index2 in zip(contestant_pairs, range(0, 2 * num_crosses, 2), range(1, 2 * num_crosses, 2)):
            children[index1], children[index2] = cros_over(parents[pairs[0]], parents[pairs[1]], cross_points[index1],
                                                           cross_points[index2])
            # print (index1)
        children[2 * num_crosses:] = parents[contestant_Idx[2 * num_crosses:]].copy()

        # 变异
        mutant_children_Idx = np.random.randint(size_pop, size=num_mutations)
        mutant_allele = np.random.randint(length, size=2 * num_mutations)

        for index, gen_Index in zip(mutant_children_Idx, range(num_mutations)):
            mutatation_swap(children[index], mutant_allele[2 * gen_Index], mutant_allele[2 * gen_Index + 1])

        # 得到新一代
        children.sort(order="cost", kind='mergesort')

        # 1. 出现更优解
        if children["cost"][0] > parents["cost"][0]:
            children[-1] = parents[0]
        # 2. 未出现更优解
        parents = children  # 直接用这一代作为新的父亲

        parents.sort(order="cost", kind='mergesort')
        res_parent.append(parents)
        if g == g_max-1:
            print("本轮共迭代", g+1, "次，最优解为：", int(parents[0]["cost"]))   # 最后一次迭代后得到的最优解
            print("最优解对应的建造方案为：", parents[0][0])   # 该最优解对应的向量
        solutions[g] = parents[0]["cost"]

        if parents[0]["cost"] < res_cost:
            res_cost = parents[0]["cost"]
            res_formula.append(parents[0])

        total_evaluations += 2 * num_crosses
        total_evaluations += num_mutations

    print("总评价次数：", total_evaluations)
    res_count.append(total_evaluations)

    return parents[0]["chromosm"], parents[0]["cost"]


def mutation_cost(current_chromosom, current_cost, idx_1, idx_2):
    new_chromosom = np.copy(current_chromosom)
    new_chromosom[idx_1], new_chromosom[idx_2] = new_chromosom[idx_2], new_chromosom[idx_1]

    mutationCost = current_cost - sum(
        np.sum(flow_matrix[[idx_1, idx_2], :] * dist_matrix[current_chromosom[[idx_1, idx_2], None], current_chromosom],
               1))
    mutationCost += sum(
        np.sum(flow_matrix[[idx_1, idx_2], :] * dist_matrix[new_chromosom[[idx_1, idx_2], None], new_chromosom], 1))

    idx = list(range(chromosome_len))
    del (idx[idx_1])
    del (idx[idx_2 - 1])

    mutationCost -= sum(sum(flow_matrix[idx][:, [idx_1, idx_2]] * dist_matrix[
        current_chromosom[idx, None], current_chromosom[[idx_1, idx_2]]]))
    mutationCost += sum(
        sum(flow_matrix[idx][:, [idx_1, idx_2]] * dist_matrix[new_chromosom[idx, None], new_chromosom[[idx_1, idx_2]]]))

    return mutationCost


def cost(chromosome):
    return sum(np.sum(flow_matrix * dist_matrix[chromosome[:, None], chromosome], 1))


if __name__ == "__main__":
    # 读取矩阵维度
    chromosome_len = int(input())
    # 读掉中间的空行
    space1 = input()
    # 读取距离矩阵数据
    dist_lines = []
    for _ in range(chromosome_len):
        dist_lines.append(list(map(int, input().split())))
    # 读掉中间的空行
    space2 = input()
    # 读取流量矩阵数据
    flow_lines = []
    for _ in range(chromosome_len):
        flow_lines.append(list(map(int, input().split())))
    # 将数据转换为NumPy数组
    dist_matrix = np.array(dist_lines)
    flow_matrix = np.array(flow_lines)

    # 打印结果
    """print("Chromosome Length:", chromosome_len)
    print("Distance Matrix:")
    print(dist_matrix)
    print("Flow Matrix:")
    print(flow_matrix)"""

    # 调用ga函数
    for j in range(10):
        print("----------------第", j, "轮调用------------------")
        start_time = time.time()
        ga_run_main(res_formula)
        end_time = time.time()
        print("运行时间；", end_time - start_time, "秒\n")
        res_time.append((end_time - start_time))

    print("--------------10轮调用过程中---------------\n")
    print("平均评价次数：", sum(res_count) / len(res_count))
    print("平均运行时间：", sum(res_time) / len(res_time))
    print("10轮调用后最优解：", res_cost)
    print("建造方案为：", res_formula[-1][0])

import time

import numpy as np
from deap import base, creator, tools, algorithms

n = 100  # 变量个数
cal_min = 1 << 32
cal_solution = 0
cal_time = []

# 定义问题
# 创建一个最小化目标
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# 创建一个个体类，包括一个适应度函数
creator.create("Individual", list, fitness=creator.FitnessMin)


# 定义函数
def objective_function(individual):
    sum1 = np.sum((np.array(individual) - 1) ** 2)
    sum2 = np.sum(np.array(individual[1:]) * np.array(individual[:-1]))
    return sum1 - sum2,


# 定义遗传算法相关参数
# 创建遗传算法工具箱
toolbox = base.Toolbox()
# 定义变量的取值范围
toolbox.register("attr_float", np.random.uniform, -n ** 2, n ** 2)  # 定义变量的取值范围
# 定义个体的生成方式
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n)
# 定义种群的生成方式
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# 定义交叉操作
toolbox.register("mate", tools.cxTwoPoint)
# 定义变异操作
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
# 定义选择操作
toolbox.register("select", tools.selTournament, tournsize=3)
# 定义评价函数
toolbox.register("evaluate", objective_function)


def main():
    global cal_min
    global cal_solution
    population_size = 50    # 种群大小
    generations = 200 * n    # 迭代次数
    mutation_rate = 0.1     # 变异率

    pop = toolbox.population(n=population_size)     # 生成初始种群
    hof = tools.HallOfFame(1)   # 记录一个最优个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)    # 创建统计对象
    stats.register("min", np.min)

    # 运行遗传算法
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=mutation_rate, ngen=generations,
                                   stats=stats, halloffame=hof, verbose=False)

    # 获取最优个体
    best_solution = hof[0]
    # 获取最优个体的适应度值
    best_fitness = objective_function(best_solution)[0]

    print("最优解:", best_solution)
    print("最优值:", best_fitness)
    if best_fitness < cal_min:
        cal_min = best_fitness
        cal_solution = best_solution


if __name__ == "__main__":
    for i in range(10):
        print("-------------第", i, "次运行-----------------")
        start_time = time.time()
        main()
        end_time = time.time()
        print("运行时间:", end_time - start_time, "秒")
        cal_time.append(end_time - start_time)

    print("-------------运行10次后---------------")
    print("最优解:", cal_solution)
    print("最优值:", cal_min)
    print("平均运行时间:", sum(cal_time) / len(cal_time))


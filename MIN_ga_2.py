import time

import numpy as np
from deap import base, creator, tools, algorithms

n = 100
cal_min = 1 << 32
cal_solution = 0
cal_time = []
# 定义问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# 定义函数
def objective_function(individual):
    sum1 = np.sum(np.array(individual) ** 2 / 4000)
    sum2 = np.prod(np.cos(np.array(individual) / np.sqrt(np.arange(1, n + 1))))
    return sum1 - sum2 + 1,


# 定义遗传算法相关参数
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -600, 600)  # 定义变量的取值范围
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 使用混合交叉算子
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.1)  # 使用高斯变异算子
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective_function)


def main():
    global cal_min
    global cal_solution
    population_size = 50
    generations = 200 * n
    mutation_rate = 0.1

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size,
                              cxpb=0.5, mutpb=mutation_rate, ngen=generations,
                              stats=None, halloffame=hof, verbose=False)

    best_solution = hof[0]
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


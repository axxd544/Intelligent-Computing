import numpy as np


# 定义函数
def objective_function(x):
    n = len(x)
    sum1 = np.sum((x - 1) ** 2)
    sum2 = np.sum(x[1:] * x[:-1])
    return sum1 - sum2


# 遗传算法
def genetic_algorithm(objective_function, bounds, population_size=100, generations=1000, mutation_rate=0.01):
    # 初始化种群
    population = np.random.uniform(bounds[0], bounds[1], size=(population_size, len(bounds)))

    for gen in range(generations):
        # 计算适应度
        fitness = np.array([objective_function(ind) for ind in population])
        print(fitness)

        # 选择父代
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices[:population_size]]

        # 交叉操作
        offspring = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(population_size, size=2, replace=False)
            crossover_point = np.random.randint(1, len(bounds))
            child = np.concatenate((population[parent1][:crossover_point], population[parent2][crossover_point:]))
            offspring.append(child)
        offspring = np.array(offspring)

        # 变异操作
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(len(bounds))
                offspring[i][mutation_point] += np.random.uniform(-0.5, 0.5)

        population = offspring

    # 返回最优解和最优值
    best_index = np.argmin(np.array([objective_function(ind) for ind in population]))
    best_solution = population[best_index]
    best_fitness = objective_function(best_solution)

    return best_solution, best_fitness


if __name__ == "__main__":
    n = 2  # 定义变量的数量
    bounds = [-n**2, n**2]  # 定义变量的搜索范围
    best_solution, best_fitness = genetic_algorithm(objective_function, bounds, generations=1000)
    print("最优解:", best_solution)
    print("最优值:", best_fitness)

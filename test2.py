import numpy as np
from deap import base, creator, tools, algorithms

n = 2

# 定义问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# 定义函数
def objective_function(individual):
    n = len(individual)
    sum1 = np.sum((np.array(individual) - 1) ** 2)
    sum2 = np.sum(np.array(individual[1:]) * np.array(individual[:-1]))
    return sum1 - sum2,


# 定义遗传算法相关参数
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -n ** 2, n ** 2)  # 定义变量的取值范围
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective_function)


def main():
    population_size = 50
    generations = 1000
    mutation_rate = 0.1

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=mutation_rate, ngen=generations,
                                   stats=stats, halloffame=hof, verbose=True)

    best_solution = hof[0]
    best_fitness = objective_function(best_solution)[0]

    print("最优解:", best_solution)
    print("最优值:", best_fitness)


if __name__ == "__main__":
    main()

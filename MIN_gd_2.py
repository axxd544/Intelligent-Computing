import time

import numpy as np


cal_result = 0
cal_min = 1 << 32
cal_time = []


# 定义函数
def objective_function(x):
    # n = len(x)
    sum1 = np.sum(np.array(x) ** 2 / 4000)
    sum2 = np.prod(np.cos(np.array(x) / np.sqrt(np.arange(1, n + 1))))
    return sum1 - sum2 + 1


# 定义梯度函数
def gradient(x):
    # n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        sum_cos = np.sum(np.cos(x[:i + 1] / np.sqrt(np.arange(1, i + 2))))
        grad[i] = x[i] / 2000 - np.sin(x[i] / np.sqrt(i + 1)) / np.sqrt(i + 1) * sum_cos
    return grad


# 梯度下降法
def gradient_descent(objective_function, gradient, initial_point, learning_rate=0.01, max_iterations=10000,
                     tolerance=1e-7):
    x = initial_point
    for _ in range(max_iterations):
        grad = gradient(x)
        x_new = x - learning_rate * grad
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x, objective_function(x)


if __name__ == "__main__":
    n = 100  # 定义变量的数量

    learning_rate = 0.01  # 学习率
    max_iterations = 10000 * n  # 最大迭代次数
    tolerance = 1e-7  # 允许的误差范围

    for i in range(10):
        initial_point = np.random.uniform(-600, 600, size=n)  # 随机生成初始点
        print("-----------------第", i, "次运行------------------")
        start_time = time.time()
        result, min_value = gradient_descent(objective_function, gradient,
                                             initial_point, learning_rate, max_iterations, tolerance)
        end_time = time.time()
        cal_time.append(end_time - start_time)
        print("最小值点:", result)
        print("最小值:", min_value)
        print("运行时间:", end_time - start_time, "秒")
        if min_value < cal_min:
            cal_min = min_value
            cal_result = result

    print("-----------------运行10次后------------------")
    print("最优解:", cal_result)
    print("最优值:", cal_min)
    print("平均运行时间:", sum(cal_time) / len(cal_time))

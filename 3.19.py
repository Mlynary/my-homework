import numpy as np
import matplotlib.pylab as plt

# 定义数据
y = [0.1, 0.05, 0.6, 0.0, 0.05,
    0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 定义均方误差函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 计算均方误差
print(np.argmax(y))
print(mean_squared_error(np.array(y), np.array(t)))

# 第二组数据
y2 = [0.1, 0.05, 0.1, 0.0, 0.05,
    0.1, 0.0, 0.6, 0.0, 0.0]
print(np.argmax(y2))
print(mean_squared_error(np.array(y2), np.array(t)))

# 绘制图形
x = np.arange(0., 1., 0.001)
plt.plot(x,-np.log(x))
plt.ylim(0., 5.3)
plt.show()

# 定义交叉熵误差函数
def cross_entropy_error(y, t):
    delta = 1e-7 # 注意！1e-7，这里是数字10的负7次方
    return -np.sum(t * np.log(y + delta))

# 计算交叉熵误差
print(cross_entropy_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(y2), np.array(t)))

# 批量处理的交叉熵误差函数
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

# 测试批量处理的交叉熵误差函数
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
print(cross_entropy_error(y, t))

# 打印维度信息
print(y.ndim)
t = t.reshape(1, t.size)
print(t)
y = y.reshape(1, y.size)
print(y)
batch_size = y.shape[0]
print(batch_size)

# 计算批量处理的交叉熵误差
delta = 1e-7
print(-np.sum(t * np.log(y + delta)) / batch_size)
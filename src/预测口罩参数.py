import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sigmoid(x):
    # 第一层到第二层的激活函数
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # 第一层到第二层的激活函数的求导函数
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # 使用方差作为损失函数
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    def __init__(self):
        # 第一层到第二层的函数
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()

        self.w21 = np.random.normal()
        self.w22 = np.random.normal()

        # 第二层到第三层的函数
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        # 截距项，Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # 前向传播学习
        h1 = sigmoid(self.w11 * x[0] + self.w12 * x[1] + self.b1)
        h2 = sigmoid(self.w21 * x[0] + self.w22 * x[1] + self.b1)
        o1 = self.w1 * h1 + self.w2 * h2 + self.b3
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.01  # 学习率
        epochs = 1000  # 训练的次数
        # 画图数据
        self.loss = np.zeros(100)
        self.sum = 0;
        # 开始训练
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 计算h1
                h1 = sigmoid(self.w11 * x[0] + self.w12 * x[1] + self.b1)
                # 计算h2
                h2 = sigmoid(self.w21 * x[0] + self.w22 * x[1] + self.b2)
                # 计算输出节点
                y_pred = self.w1 * h1 + self.w2 * h2 + self.b3
                # 反向传播计算导数
                d_L_d_ypred = -2 * (y_true - y_pred)
                d_ypred_d_w1 = h1
                d_ypred_d_w2 = h2
                d_ypred_d_b3 = 1
                d_ypred_d_h1 = self.w1
                d_ypred_d_h2 = self.w2
                sum_1 = self.w11 * x[0] + self.w12 * x[1] + self.b1
                d_h1_d_w11 = x[0] * deriv_sigmoid(sum_1)
                d_h1_d_w12 = x[1] * deriv_sigmoid(sum_1)
                d_h1_d_b1 = deriv_sigmoid(sum_1)
                sum_2 = self.w21 * x[0] + self.w22 * x[1] + self.b2
                d_h1_d_w21 = x[0] * deriv_sigmoid(sum_2)
                d_h1_d_w22 = x[1] * deriv_sigmoid(sum_2)
                d_h1_d_b2 = deriv_sigmoid(sum_2)

                # 梯度下降法
                self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w11
                self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w12

                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                self.w21 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w21
                self.w22 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w22

                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_b2
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_w1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_w2
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("模型训练次数：%d   损失率: %.3f" % (epoch, loss))
                self.loss[self.sum] = loss
                self.sum = self.sum + 1


# 禁用科学计数法
pd.set_option('float_format', lambda x: '%.3f' % x)
# np.set_printoptions(suppress=True, threshold=np.nan)
# 得到的DataFrame分别为接受距离，热风速度，厚度，孔隙率，压缩回弹性率
data = pd.read_excel(r'C:\Users\青山七海\PycharmProjects\MathModelProject\File\C题数据.xlsx', sheet_name="data3", header=0,
                     usecols="A,B,C,D,E")
# DataFrame转化为array
DataArray = data.values
A = DataArray[:, 2:5]
B = DataArray[:, 0:2]
# C = DataArray[:, 1:2]
B = np.array(B)  # 转化为array,自变量
A = np.array(A)  # 转化为array，因变量
# C = np.array(A)
# 处理数据
data = np.array(B)
data_mean = np.sum(data, axis=0) / np.size(data, 0)
data = (data - data_mean) / np.max(data)
all_y_trues = np.array(A)
all_y_trues_mean = np.sum(all_y_trues) / np.size(all_y_trues)  # 平均值
all_y_trues = (all_y_trues - all_y_trues_mean) / np.max(all_y_trues)  # 所有y的标准化数值
# 训练数据
network = OurNeuralNetwork()
network.train(data, all_y_trues)
# 标题显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 测试数据
testData = np.array([23, 850])

print(network.feedforward(testData))

# 损失函数曲线图
plt.plot(np.arange(100), network.loss)
plt.show()
# 真实值与预测值对比
y_preds = np.apply_along_axis(network.feedforward, 1, data)

plt.plot(np.arange(82), all_y_trues, "r^")
plt.plot(np.arange(82), y_preds, "bs")
plt.title("红色为真实值，蓝色为预测值")
plt.show()

import torch
import matplotlib
matplotlib.use('TkAgg')
from torch import optim
import matplotlib.pyplot as plt


# 1. 定义函数, 演示: 等间隔学习率衰减
def dm01(lr):
    # 1. 定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数.
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集.  y_true, x, w
    # 真实值.
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象, 动量法 -> 加速模型的收敛, 减少震荡.
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数


    # 5. 创建两个列表, 分别表示: 训练轮数, 每轮训练用的学习率
    # epoch_list = [0, 1, 2, 3.... 50, 51, 52...100, 101, 101... 150, 151...199]
    i_list = []
    iteration_list = []
    # 6. 循环遍历训练轮数, 进行具体的训练.
    b=0
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    for epoch in range(20):     # epoch: 0 ~ 199
        # 7. 获取当前轮数 和 学习率, 并保存到列表中.
        # 8. 循环遍历, 每轮每批次进行训练.
        for batch in range(iteration):
            b=b+1
            i_list.append(b)
            # 9. 先计算预测值, 然后基于损失函数计算损失.
            # 10. 计算损失, 最小二乘法.
            loss = (w) ** 2
            iteration_list.append(loss.item())
            # 11. 梯度清零 + 反向传播 + 优化器更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 14. 可视化.
    # x轴: 训练的轮数, y轴: 每轮训练用的损失值
    plt.plot(i_list, iteration_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss num')
    plt.show()
dm01(0.6)
dm01(0.3)
dm01(0.1)
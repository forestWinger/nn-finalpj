# 神经网络期末报告
## 马骎 21307110024
## cifar-10训练
采用了自定义的resnet18模型进行训练
小模型采用16-32-64-128-256逐层增加的卷积层进行特征提取，使用adam进行训练
![alt text](image-3.png)
较大的模型采用32-64-128-256-512数量的卷积核进行训练，采用sgd，并设置了依据训练损失的学习率下降的scheduler，具体是patience为3，若超过则将学习率降为0.5倍。

![alt text](image.png)
![alt text](image-1.png)

可以看到第二个设置的模型效果更好。

实验中比较了不同weight decay和激活函数的设置，发现Relu和0.0005的配置最好。

可视化滤波器
![alt text](image-4.png)

可视化特征图
![alt text](image-5.png)

原图
![alt text](image-6.png)

loss landscape可视化
![alt text](image-7.png)
可以看到较为明显的最优点，说明训练顺利。

## VGG-BN实验
| 特性               | 使用 BatchNorm   | 不使用 BatchNorm  |
| ---------------- | -------------- | -------------- |
| Loss 曲线平滑程度      | 明显更平滑，震荡小      | 波动大，不稳定        |
| min/max 差距（区间宽度） | 更窄，说明 loss 更稳定 | 更宽，说明不同 lr 差异大 |
| 收敛速度             | 较快，训练更稳定       | 易震荡或卡顿         |
| 泛化能力（验证集表现）      | 通常更好           | 容易过拟合或欠拟合      |

结论：BN 提高了训练稳定性与泛化能力，Loss Landscape 更平缓，便于优化器收敛。
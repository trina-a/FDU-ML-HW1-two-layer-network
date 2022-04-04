# FDU-ML-HW1-two-layer-network

## 1、代码说明
该两层神经网络的实现有三个py文件:NeuralNetwork.py、Train.py、test.py。
NeuralNetwork.py中定义了神经网络的类NeuralNetwork，Train.py为训练集数据的导入、网络的训练及保存、以及部分可视化程序，test.py为测试集数据的导入以及预测。

**（1）NeuralNetwork类主要函数说明**
```
loss():计算损失函数，定义了神经网络的前向传导和反向传播过程
```
```
updata_lr():学习率更新函数
```

```
train()：训练网络的函数
参数：X,y：训练集的X及Y，y为$1\times10$的向量
      y_train：训练集的y，y取值1~9
      inilr:初始的学习率
      mini_batch_size：mini batch的大小，默认设置为64
      epoch：epoch数量，默认设置为10
```
```
predict()：输入X，预测Y
参数：X_test：输入X
```

```
get_lr():输出训练过程中的学习率列表
```
```
get_hiddensize()：输出隐藏层的节点个数
```
```
get_reg()：输出正则化强度
```
训练模型调用train()函数
参数查找学习率，隐藏层大小，正则化强度分别调用get_lr()、get_hiddensize()、get_reg()函数
运用模型进行预测调用predict()函数

## 2、训练和测试步骤
训练集和测试集已经处理为csv格式，分别为mnist_train.csv和mnist_test.csv
**(1）训练模型**
运行Train.py,其中导入数据，设定参数，训练模型，可视化训练的loss和accuracy曲线，参数查找学习率，隐藏层大小，正则化强度，保存模型代码

**（2）读取已训练的模型并在测试集测试模型**
运行Test.py

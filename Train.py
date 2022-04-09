import pandas as pd
import pickle
import matplotlib.pyplot as plt
from NeuralNetwork import *
import matplotlib


# 导入训练集，处理训练集数据格式
data_train = pd.read_csv('mnist_train.csv', header=None)
data_train.columns = ['x' + str(v) for v in range(785)]
dummies = pd.get_dummies(data_train['x0'])
data_train = dummies.join(data_train)

X_train = np.array(data_train.iloc[:, 11:])
Y_train = np.array(data_train.iloc[:, 0:10])
Y_train_label = np.array(data_train.iloc[:, 10])

# 设定参数
inilr=0.0001
mini_batch_size=64
epoch=10

#训练模型
net=NeuralNetwork(input_size=784, hidden_size=1000, output_size=10)
loss_list,acc_list=net.train(X_train, Y_train, Y_train_label, inilr=inilr, mini_batch_size=mini_batch_size,epoch=epoch)

# #可视化训练的loss和accuracy曲线
#
# matplotlib.use('TKAgg')#加上这行代码即可，agg是一个没有图形显示界面的终端，常用的有图形界面显示的终端有TkAgg等
# plt.subplot(211)
# plt.plot(loss_list,label = 'train_loss')
# plt.title('train loss')
# plt.ylabel('loss')
# plt.xlabel('iteration')
# plt.legend()
#
# plt.subplot(212)
# plt.plot(acc_list,label = 'train_acc',color = 'red')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(loc = 'lower right')
#
# plt.savefig('train_loss_acc.jpg')
# plt.show(block=True)

#参数查找学习率，隐藏层大小，正则化强度
#学习率列表
lr_list=net.get_lr()
print(F"learning rate: {lr_list}")
#隐藏层节点数
hiddensize=net.get_hiddensize()
print(F"the number of node in hidden layer: {hiddensize}")
#正则化强度
reg=net.get_reg()
print(F"Regularization: {reg}")

#保存模型
output_hal = open("model.pkl", 'wb')
str = pickle.dumps(net)
output_hal.write(str)
output_hal.close()
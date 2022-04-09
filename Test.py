import pandas as pd
import pickle
from NeuralNetwork import *

#读取模型
with open("1.pkl", 'rb') as file:
    net = pickle.loads(file.read())

# 导入测试集，处理测试集数据格式
data_test = pd.read_csv('mnist_test.csv', header=None)
data_test.columns = ['x' + str(v) for v in range(785)]
dummy2 = pd.get_dummies(data_test['x0'])
data_test = dummy2.join(data_test)

X_test = np.array(data_test.iloc[:, 11:])
Y_test_label = np.array(data_test.iloc[:, 10])

#计算测试集上的accuracy
accurate = np.mean(Y_test_label == net.predict(X_test))
print(F"Test accuracy: {accurate}")
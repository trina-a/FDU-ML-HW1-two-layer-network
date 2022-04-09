import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def Relu(x):
    return np.where(x>0,x,0)

def dRelu(x):
    return x>=0

class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = 0.01 * np.random.randn(input_size, hidden_size)  # D*H
        self.b1 = np.zeros(hidden_size)  # H
        self.W2 = 0.01 * np.random.randn(hidden_size, output_size)  # H*C
        self.b2 = np.zeros(output_size)  # C
        self.lr=[]
        self.reg=0.01


    def loss(self, X, y):
        num_train, num_feature = X.shape
        # forward
        a1 = X  # input layer:N*D
        a2 = Relu(a1.dot(self.W1) + self.b1)  # hidden layer:N*H
        a3 = sigmoid(a2.dot(self.W2) + self.b2)  # output layer:N*C

        loss = - np.sum(y * np.log(a3) + (1 - y) * np.log((1 - a3))) / num_train
        loss += 0.5 * self.reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2)) / num_train

        # backward
        error3 = a3 - y  # N*C
        dW2 = a2.T.dot(error3) + self.reg * self.W2  # (H*N)*(N*C)=H*C
        db2 = np.sum(error3, axis=0)

        error2 = error3.dot(self.W2.T) * dRelu(a2)  # N*H
        dW1 = a1.T.dot(error2) + self.reg * self.W1  # (D*N)*(N*H) =D*H
        db1 = np.sum(error2, axis=0)

        dW1 /= num_train
        dW2 /= num_train
        db1 /= num_train
        db2 /= num_train

        return loss, dW1, dW2, db1, db2

    def update_lr(self,epoch_num,inilr):
        #学习率按训练轮数增长指数差值递减

        return 0.95**epoch_num*inilr

    def mini_batches(self,X, Y, y_train,batch_size):
        # 取数据集大小
        m = X.shape[0]
        # 随机生成一个索引顺序
        permutation = list(np.random.permutation(m))
        # 把X,Y打乱成相同顺序
        X_shuffle = X[permutation,:]
        Y_shuffle = Y[permutation,:]
        y_train_shuffle=y_train[permutation]
        # 建立一个空列表，存储迷你批
        mini_batches = []
        i=0
        while i* batch_size<=m:
            mini_batches.append([X_shuffle[i*batch_size:(i+1)*batch_size,:],Y_shuffle[i*batch_size:(i+1)*batch_size,:],
                                 y_train_shuffle[i*batch_size:(i+1)*batch_size]])
            i+=1
        return mini_batches


    def train(self, X, y, y_train, inilr=0.01, mini_batch_size=64,epoch=10):
        '''
        :param X: 训练集的X
        :param y:y为1*10的向量
        :param y_train:训练集的y，y取值1~9
        :param inilr:初始的学习率
        :param mini_batch_size:mini batch的大小，默认设置为64
        :param epoch:epoch数量，默认设置为10
        :return:
        '''
        loss_list = []
        acc_list=[]

        # 每个epoch都会重新划分一次batch
        for k in range(epoch):
            mini_batch = self.mini_batches(X, y, y_train,mini_batch_size)
            #每个epoch更新一下learning rate
            learn_rate=self.update_lr(k,inilr)
            self.lr.append(learn_rate)

            for i in range(len(mini_batch)):
                X_batch=mini_batch[i][0]
                y_batch = mini_batch[i][1]
                y_train_batch=mini_batch[i][2]

                loss, dW1, dW2, db1, db2 = self.loss(X_batch, y_batch)
                loss_list.append(loss)


                # update the weight
                self.W1 += -learn_rate * dW1
                self.W2 += -learn_rate * dW2
                self.b1 += -learn_rate * db1
                self.b2 += -learn_rate * db2

            #每个epoch记录一下train loss,acc,validation loss,输出一次 loss,accuracy,vol_loss,vol_accuracy
            train_acc = np.mean(y_train_batch== self.predict(X_batch))
            acc_list.append(train_acc)
            #val_acc = np.mean(y_val == self.predict(X_val))
            print(F"Epoch: {k+1}/{epoch}, loss: {loss},accuracy: {train_acc}, ")

        return loss_list,acc_list

    def predict(self, X_test):
        a2 = Relu(X_test.dot(self.W1) + self.b1)
        a3 = sigmoid(a2.dot(self.W2) + self.b2)
        y_pred = np.argmax(a3, axis=1)
        return y_pred

    def get_lr(self):
        return self.lr

    def get_hiddensize(self):
        return self.W1.shape[1]

    def get_reg(self):
        return self.reg
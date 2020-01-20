#encoding: utf-8

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


max_index = -2
min_max_scaler_X = None
X = None
y = None
T_var = None
gsess = None
init = None
saver = None
logits = None

# 倒入数据
def trainModel(datafilepath):
    global min_max_scaler_X
    global logits
    global saver
    global init
    global gsess
    global X
    global y
    global T_var
    global max_index
    data = pd.read_excel(datafilepath)

    feature_cols = ['性别','年龄','体重','输血前血小板计数(x10^9/L)','输血前血红蛋白测定(g/L)','输血后血红蛋白测定(g/L)','保存时间(天)']
    X_data = data[feature_cols]
    y_data = data['输血实红量']
    # 选择特征
    # 进行 n_splits=10 折实验
    rmse = []
    mae = []
    batch_loss = []
    i = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    for train_index, test_index in kf.split(data):  # 索引
        i = i + 1
        print(i)
        # 划分测试集 训练集合
        X_train_ori, X_test_ori = X_data.values[train_index], X_data.values[test_index]
        y_train_ori, y_test_ori = y_data.values[train_index], y_data.values[test_index]

        # 数据归一化
        min_max_scaler_X = preprocessing.MinMaxScaler()
        min_max_scaler_y = preprocessing.MinMaxScaler()

        X_train = min_max_scaler_X.fit_transform(X_train_ori)
        X_train_T = X_train[:, max_index].reshape(-1, 1)
        X_train_rest = np.delete(X_train, max_index, 1)

        X_test = min_max_scaler_X.transform(X_test_ori)
        X_test_T = X_test[:, max_index].reshape(-1, 1)
        X_test_rest = np.delete(X_test, max_index, 1)

        y_train = y_train_ori.reshape(-1, 1)
        y_test = y_test_ori.reshape(-1, 1)

        n_inputs = 7 # 输入层

        # 隐藏层
        n_hidden1 = 512
        n_hidden2 = 256
        n_hidden3 = 128

        n_outputs = 1 # 输出层
        # activation = tf.nn.relu

        X = tf.placeholder(tf.float32, shape=(None, n_inputs - 1), name="X")
        T_var = tf.placeholder(tf.float32, shape=(None, 1), name='T')
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        def neuron_layer(X, n_neurons, name, activation=None):
            '''定义隐藏层'''
            with tf.name_scope(name):
                n_inputs = int(X.get_shape()[1])
                stddev = 2 / np.sqrt(n_inputs)
                init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
                W = tf.Variable(init, name="kernel")
                b = tf.Variable(tf.zeros([n_neurons]), name="bias")
                Z = tf.matmul(X, W) + b
                if activation is not None:
                    return activation(Z)
                else:
                    return Z

        with tf.name_scope("dnn"):
            hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                                   activation=tf.nn.relu)
            hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                                   activation=tf.nn.relu)
            hidden3 = neuron_layer(hidden2, n_hidden3, name="hidden3",
                                   activation=tf.nn.sigmoid)
            # 构造异构网络 最后一层加入选定的特征
            hidden3_new = tf.concat([hidden3, T_var], 1)

            logits = neuron_layer(hidden3_new, n_outputs, name="outputs")

        with tf.name_scope("loss"):
            # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            # loss = tf.reduce_mean(xentropy, name="loss")
            # 使用MAE作为损失函数
            loss = tf.losses.mean_squared_error(labels=y, predictions=logits)
            # loss = tf.reduce_mean(tf.square(y - logits))

        learning_rate = 0.01  # 学习效率

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        n_epochs = 100 # 训练次数
        losses = []
        saver = tf.train.Saver()
        if gsess != None:
            gsess.close()

        sess = tf.Session()
        gsess = sess
        init.run(session=gsess)
        for epoch in range(n_epochs):
            losses.append(loss.eval(session=gsess, feed_dict={X: X_train_rest, y: y_train, T_var: X_train_T}))
            sess.run(training_op, feed_dict={X: X_train_rest, y: y_train, T_var: X_train_T})
            if epoch % 1 == 0:
                #                 print(epoch, "Batch loss:", loss.eval(feed_dict={X: X_train_rest, y: y_train.reshape(-1,), T_var: X_train_T.reshape(-1,1)}))
                batchloss = loss.eval(session=gsess, feed_dict={X: X_train_rest, y: y_train.reshape(-1, ), T_var: X_train_T.reshape(-1, 1)})
                batch_loss.append(batchloss)

                y_test_pre_minmax = logits.eval(session=gsess, feed_dict={X: X_test_rest, T_var: X_test_T.reshape(-1, 1)})
                y_test_pre = y_test_pre_minmax

                rmse_test = np.sqrt(mean_squared_error(y_test_pre, y_test_ori))
                #                 print(epoch, "original test rmse:", rmse_test)

               
                # 寻找最优模型
                min_rmse_test = 1000
                if rmse_test < min_rmse_test:
                    min_rmse_test = rmse_test

                mae_test = mean_absolute_error(y_test_pre, y_test_ori)
                #                 print(epoch, "original test mae:", mae_test)
                min_mae_test = 1000
                if mae_test < min_mae_test:
                    min_mae_test = mae_test
        saver.save(sess, './%d_model/model.ckpt'%i)
        mae1 = min_mae_test
        rmse1 = min_rmse_test

           
        rmse.append(rmse1)
        mae.append(mae1)
        print('---------------------------------------------------------')
        return np.mean(rmse), np.mean(mae)


def predict(dataarr):
    global min_max_scaler_X
    global logits
    global saver
    global init
    global gsess
    global X
    global y
    global T_var
    global max_index
    inputargs = np.array(dataarr).reshape(1,-1)
    inputargs_minmax = min_max_scaler_X.transform(inputargs)
    inputargs_minmax_T = inputargs_minmax[:,max_index].reshape(-1,1)
    inputargs_minmax_rest = np.delete(inputargs_minmax, max_index, 1)
    y_test_pre_minmax = logits.eval(session=gsess, feed_dict={X: inputargs_minmax_rest, T_var: inputargs_minmax_T.reshape(-1,1)})
    return y_test_pre_minmax


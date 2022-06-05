# -*- coding: utf-8 -*-
# 改进鲸鱼优化lstm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from scipy.io import savemat
import scipy.special as sps

# In[]

'''
进行适应度计算
'''


def fitness(pop, P, T, Pt, Tt):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    alpha = pop[0]
    num_epochs = int(pop[1])
    hidden_nodes0 = int(pop[2])
    hidden_nodes = int(pop[3])
    input_features = P.shape[1]
    output_class = T.shape[1]
    batch_size = 16  # batchsize
    # placeholder
    X = tf.placeholder("float", [None, input_features])
    Y = tf.placeholder("float", [None, output_class])

    # 定义一个隐层的神经网络
    def RNN(x, hidden_nodes0, hidden_nodes, input_features, output_class):
        x = tf.reshape(x, [-1, 1, input_features])
        # 定义输出层权重
        weights = {'out': tf.Variable(tf.random_normal([hidden_nodes, output_class]))}
        biases = {'out': tf.Variable(tf.random_normal([output_class]))}
        lstm_cell0 = tf.nn.rnn_cell.LSTMCell(hidden_nodes0)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_nodes)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell0, lstm_cell])
        # 初始化
        init_state = lstm_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, initial_state=init_state)

        output_sequence = tf.matmul(tf.reshape(outputs, [-1, hidden_nodes]), weights['out']) + biases['out']
        return tf.reshape(output_sequence, [-1, output_class])

    logits = RNN(X, hidden_nodes0, hidden_nodes, input_features, output_class)
    loss = tf.losses.mean_squared_error(predictions=logits, labels=Y)
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        alpha,
        global_step,
        num_epochs, 0.99,
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-10).minimize(loss,
                                                                                            global_step=global_step)
    init = tf.global_variables_initializer()

    # 训练
    with tf.Session() as sess:
        sess.run(init)
        N = P.shape[0]
        for epoch in range(num_epochs):
            total_batch = int(math.ceil(N / batch_size))
            indices = np.arange(N)
            np.random.shuffle(indices)
            avg_loss = 0
            # 迭代训练，顺便计算训练集loss
            for i in range(total_batch):
                rand_index = indices[batch_size * i:batch_size * (i + 1)]
                x = P[rand_index]
                y = T[rand_index]
                _, cost = sess.run([optimizer, loss],
                                   feed_dict={X: x, Y: y})
                avg_loss += cost / total_batch
        # 计算测试集的预测值

        test_pred = sess.run(logits, feed_dict={X: Pt})
        test_pred = test_pred.reshape(-1, output_class)

    F2 = np.mean(np.square((test_pred - Tt)))
    return F2


'''
约束迭代结果
'''


def boundary(pop, Lb, Ub):
    # 防止粒子跳出范围
    # 除学习率之外  其他的都是整数
    pop = [pop[i] if i == 0 else int(pop[i]) for i in range(len(Lb))]
    pop[0] = int(pop[0])
    pop[2] = int(pop[2])
    pop[3] = int(pop[3])  # 迭代数和节点数都应为整数
    for i in range(len(Lb)):
        if pop[i] > Ub[i] or pop[i] < Lb[i]:
            if i == 0:
                pop[i] = (Ub[i] - Lb[i]) * np.random.rand() + Lb[i]
            else:
                pop[i] = np.random.randint(Lb[i], Ub[i])
    return pop


'''guass映射'''


def GuassMap(dim):
    r = np.linspace(0, 0.0001, dim);
    x = np.zeros([dim])

    x0 = 0.1  # 初始点
    x[0] = np.exp(-5 * x0 ** 2) - r[0];
    for j in range(1, dim):
        x[j] = np.exp(-5 * x[j - 1] ** 2) - r[j]
    return x


'''Levy飞行'''


def Levy(d):
    beta = 1.5
    sigma = (sps.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                sps.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta);
    # Mantegna's algorithm for Levy random numbers
    u = np.random.normal(d) * sigma
    v = np.random.normal(d)
    step = u / (np.abs(v) ** (1 / beta))
    L = 0.01 * step  # Final Levy steps
    return L


def iwoa(p_train, t_trian, p_test, t_test):
    '''
        noclus = 维度
        max_iterations = 迭代次数
        noposs 种群数
    '''
    noclus = 4
    max_iterations = 10
    noposs = 10
    Lb = [0.001, 10, 1, 1]  # 下界
    Ub = [0.01, 100, 200, 200]  # 上界,依次为学习率，训练次数，两个层的节点数
    poss_sols = np.zeros((noposs, noclus))  # whale positions
    gbest = np.zeros((noclus,))  # globally best whale postitions
    fitnessi = np.zeros((noposs,))
    b = 2.0
    # 种群初始化,除学习率之外  其他的都是整数
    '''改进点1：加入高斯混沌映射初始化'''
    for i in range(noposs):
        MapValue = GuassMap(noclus)
        for j in range(noclus):
            if j == 0:
                poss_sols[i][j] = (Ub[j] - Lb[j]) * MapValue[j] + Lb[j]
            else:
                poss_sols[i][j] = int((Ub[j] - Lb[j]) * MapValue[j] + Lb[j])
    global_fitness = np.inf
    for i in range(noposs):
        fitnessi[i] = fitness(poss_sols[i, :], p_train, t_trian, p_test, t_test)
        if fitnessi[i] < global_fitness:
            global_fitness = fitnessi[i]
            gbest = poss_sols[i].copy()
    # 开始迭代
    trace, trace_pop = [], []
    for it in range(max_iterations):
        '''改进点4非线性权重'''
        w = 1 - (1 - 0) * np.arcsin(it / max_iterations * 2 / np.pi)  # 改进粒子群算法优化LSTM神经网络的铁路客运量预测_李万

        for i in range(noposs):
            a = 2.0 - (2.0 * it) / (1.0 * max_iterations)
            r = np.random.random_sample()
            A = 2.0 * a * r - a
            C = 2.0 * r
            l = 2.0 * np.random.random_sample() - 1.0
            p = np.random.random_sample()

            for j in range(noclus):
                x = poss_sols[i][j]
                if p < 0.5:
                    if abs(A) < 1:
                        _x = gbest[j].copy()
                    else:
                        rand = np.random.randint(noposs)
                        _x = poss_sols[rand][j]
                    D = abs(C * _x - x)
                    updatedx = _x - A * D
                else:
                    _x = gbest[j].copy()
                    D = abs(_x - x)
                    updatedx = w * D * math.exp(b * l) * math.cos(2.0 * math.acos(-1.0) * l) + _x
                # if updatedx < ground[0] or updatedx > ground[1]:
                #    updatedx = (ground[1]-ground[0])*np.random.rand()+ground[0]
                #   randomcount += 1

                poss_sols[i][j] = updatedx
            '''改进点2：levy飞行进行更新'''
            L = Levy(noclus)
            dS = L * (poss_sols[i, :] - gbest)
            poss_sols[i, :] = poss_sols[i, :] + dS
            poss_sols[i, :] = boundary(poss_sols[i, :], Lb, Ub)

            fitnessi[i] = fitness(poss_sols[i], p_train, t_trian, p_test, t_test)
            if fitnessi[i] < global_fitness:
                global_fitness = fitnessi[i]
                gbest = poss_sols[i].copy()
        '''改进点3后期加精英反向学习机制'''
        if it > int(2 / 3 * max_iterations):
            for i in range(noposs):
                poss_ij = np.random.rand((noclus)) * (poss_sols.min(0) + poss_sols.max(0)) - poss_sols[i, :]
                poss_ij = boundary(poss_ij, Lb, Ub)
                fiti = fitness(poss_sols[i], p_train, t_trian, p_test, t_test)
                if fiti < fitnessi[i]:
                    fitnessi[i] = fiti
                    poss_sols[i, :] = poss_ij.copy()
                if fitnessi[i] < global_fitness:
                    global_fitness = fitnessi[i]
                    gbest = poss_sols[i].copy()

        trace.append(global_fitness)
        print("iteration", it + 1, "=", global_fitness, [gbest[i] if i == 0 else int(gbest[i]) for i in range(len(Lb))])
        trace_pop.append(gbest)
    return gbest, trace, trace_pop


# In[] 加载数据
xlsfile = pd.read_excel('').iloc[0:, 1:]

data = np.array(xlsfile)

in_ = np.hstack((data[1:, 0:5], data[0:-1, :]))
out_ = data[1:, 5:]

n = range(in_.shape[0])
m = 95
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
train_label = out_[n[0:m],]
test_label = out_[n[m:],]
# 归一化


ss_X = MinMaxScaler(feature_range=(0, 1)).fit(train_data)
ss_Y = MinMaxScaler(feature_range=(0, 1)).fit(train_label)
train_data = ss_X.transform(train_data)
train_label = ss_Y.transform(train_label)

test_data = ss_X.transform(test_data)
test_label = ss_Y.transform(test_label)

in_num = train_data.shape[1]
out_num = train_label.shape[1]
# In[]
pop, trace, result = iwoa(train_data, train_label, test_data, test_label)
savemat('result/iwoa_lstm_result.mat', {'pop': pop, 'trace': trace, 'result': result})
# In[]
result = np.array(result)
plt.figure()
plt.plot(trace)
plt.title('fitness curve')
plt.xlabel('iteration')
plt.ylabel('fitness value')
plt.savefig('iwoa_lstm图片保存/fitness curve.png')
plt.show()

plt.figure()
plt.plot(result[:, 0])
plt.title('learning rate optim')
plt.xlabel('iteration')
plt.ylabel('learning rate value')
plt.savefig('iwoa_lstm图片保存/lr curve.png')
plt.show()
plt.figure()
plt.plot(result[:, 1])
plt.title('itration optim')
plt.xlabel('iteration')
plt.ylabel('itration value')
plt.savefig('iwoa_lstm图片保存/itration curve.png')
plt.show()

plt.figure()
plt.plot(result[:, 2])
plt.title('first hidden nodes optim')
plt.xlabel('iteration')
plt.ylabel('first hidden nodes value')
plt.savefig('iwoa_lstm图片保存/first hidden-node curve.png')
plt.show()

plt.figure()
plt.plot(result[:, 3])
plt.title('second hidden nodes optim')
plt.xlabel('iteration')
plt.ylabel('second hidden nodes value')
plt.savefig('iwoa_lstm图片保存/second hidden-node curve.png')
plt.show()
# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import pickle

from sklearn.feature_extraction import DictVectorizer

# 读取迁移学习网络训练的数据
with open('/data.pk', 'r') as f:
    X = pickle.load(f)

# 700万条带标签的数据集23种攻击类型
dataframe = pd.read_csv('/root/kddcup.data.corrected.csv', header=None)

# 提取label
Y = dataframe.iloc[:, 41].values

del dataframe

# 对攻击类型编码
v = DictVectorizer(sparse=False)
D = [{'attack_cat': y} for y in Y]
labeled_Y = v.fit_transform(D)

del D


# 下面一部分代码用于从每种攻击类型提取80%的数据用于训练卷积神经网络，当然可以提取100%。

inds = []
category_list = np.unique(Y)
for category in category_list:
    ind = np.where(Y == category)[0]
    len_of_subset = int(len(ind) * 0.8)
    inds.extend(ind[0:len_of_subset])


X_train = X
Y_train = labeled_Y
X_test = np.delete(X, inds, axis=0)
Y_test = np.delete(labeled_Y, inds, axis=0)


class IdsNetwork:

    def weight_variable(self, shape):
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        return weights

    def bias_variable(self, shape):
        biases = tf.Variable(tf.constant(0.1, shape=shape))
        return biases

    def conv2d(self, x, w):                                  # 因为不设计图像问题，索引没有填充0值
        h_conv2d = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        return h_conv2d

    def max_pool_2x2(self, x):
        h_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
        return h_pool
    # 生成每次训练的batch
    def next_batch(self, feature_list, label_list, size):
        feature_batch = []
        label_batch = []
        # random.sample函数随机生成size大小的索引列表
        inds = random.sample(range(len(feature_list)), size)
        for i in inds:
            feature_batch.append(feature_list[i])
            label_batch.append(label_list[i])
        return feature_batch, label_batch

    def train(self, X_train, Y_train):
        xs = tf.placeholder(tf.float32, [None, 45])
        ys = tf.placeholder(tf.float32, [None, 23])
        keep_prob = tf.placeholder(tf.float32)
        x = tf.reshape(xs, [-1, 5, 9, 1])
        # 2 * 6 的卷积核，生成 4 * 4 张量
        self.W_conv1 = self.weight_variable([2, 6, 1, 32])
        self.b_conv1 = self.bias_variable([32])

        self.h_conv1 = tf.nn.sigmoid(self.conv2d(x, self.W_conv1) + self.b_conv1)

        # 经过2 * 2 的pooling层， 生成 3 * 3 的张量
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # 全连接层
        self.W_fc1 = self.weight_variable([3*3*32, 1024])
        self.b_fc1 = self.bias_variable([1024])
        # 将池化生成的张量变形
        self.h_pool2_flat = tf.reshape(self.h_pool1, [-1, 3*3*32])
        self.h_fc1 = tf.nn.sigmoid(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        # 过拟合处理
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)
        # 全连接层
        W_fc2 = self.weight_variable([1024, 23])
        b_fc2 = self.bias_variable([23])
        # softmax函数分类
        self.prediction = tf.nn.softmax(tf.matmul(self.h_fc1_drop, W_fc2) + b_fc2)
        self.cross_entropy = -tf.reduce_sum(ys * tf.log(self.prediction))
        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(ys, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # 随机题都下降优化
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for step in range(500):
                feature_train_batch, label_train_batch = self.next_batch(X_train, Y_train, 1000)
                sess.run(train_step, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob:0.5})
                print(step)

# 初始化模型
model = IdsNetwork()
# 训练模型
model.train(X_train, Y_train)

# 读取迁移网络的数据
with open('/total_data_X.pk', 'r') as f:
    X = pickle.load(f)
with open('/total_data_Y.pk', 'r') as f:
    Y = pickle.load(f)

# 对攻击类型编码
v = DictVectorizer(sparse=False)
D = [{'attack_cat': y} for y in Y]
labeled_Y = v.fit_transform(D)


x = tf.placeholder(tf.float32, [None, 45])
y = tf.placeholder(tf.float32, [None, 21])
keep_prob = tf.placeholder(tf.float32)

x1 = tf.reshape(x, [-1, 5, 9, 1])

# 'model.'调用上一个模型训练好的模型的参数
W_conv1 = model.W_conv1
b_conv1 = model.b_conv1

# 卷积层
h_conv1 = tf.nn.sigmoid(model.conv2d(x1, W_conv1) + b_conv1)

# 池化层
h_pool1 = model.max_pool_2x2(h_conv1)

# 全连接层
W_fc1 = model.W_fc1
b_fc1 = model.b_fc1
h_pool2_flat = tf.reshape(h_pool1, [-1, 3*3*32])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# W_fc2, b_fc2调用权重，偏置初始化函数
W_fc2 = model.weight_variable([1024, 21])
b_fc2 = model.bias_variable([21])
# 使用softmax函数分类
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 随机题都下降优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(300):
        feature_train_batch, label_train_batch = model.next_batch(X, labeled_Y, 100)
        sess.run(train_step, feed_dict={x: feature_train_batch, y: label_train_batch, keep_prob: 0.5})
        if step % 5 == 0:
            acc = sess.run(accuracy, feed_dict={x: X, y: labeled_Y, keep_prob: 0.5})
            print(acc)



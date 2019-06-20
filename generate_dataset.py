# coding=utf-8
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

'''generate_dataset.py用于生成训练迁移学习神经网络的数据'''
with open('/first_model.pk', 'r') as f:
    model = pickle.load(f)

# 读取全部的数据集
dataframe = pd.read_csv('/root/kddcup.data.corrected.csv', header=None)

protocol_list = np.unique(dataframe[1])
service_list = np.unique(dataframe[2])
flag_list = np.unique(dataframe[3])
attack_list = np.unique(dataframe[41])


def nominal_encode(x, category):
    for index in range(len(category)):
        if x == category[index]:
            return index


dataframe[1] = dataframe[1].apply(nominal_encode, args=(protocol_list, ))
dataframe[2] = dataframe[2].apply(nominal_encode, args=(service_list, ))
dataframe[3] = dataframe[3].apply(nominal_encode, args=(flag_list, ))

# 提取用于知识发现网络模型预测数据所需的X
X = dataframe.loc[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 31, 32]].values
X = StandardScaler().fit_transform(X)

dataset = np.delete(dataframe.values, [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 31, 32, 41], axis=1)

del dataframe

#
Y_predicted = model.predict(X)
# 将没有训练和模型生成的数据组合
dataset = np.hstack((dataset, Y_predicted))

# 存储
with open('/data.pk', 'w') as f:
    pickle.dump(dataset, f)

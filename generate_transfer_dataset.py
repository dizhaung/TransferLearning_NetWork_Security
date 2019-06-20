# coding=utf-8
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

# 读取知识发现模型
with open('/first_model.pk', 'r') as f:
    model = pickle.load(f)

# 读取包含未知攻击类型的数据集
total_dataframe = pd.read_csv('/root/corrected', header=None)
par_dataframe = pd.read_csv('/root/kddcup.data_10_percent_corrected.csv', header=None)

protocol_list = np.unique(total_dataframe[1])
service_list = np.unique(total_dataframe[2])
flag_list = np.unique(total_dataframe[3])
# 全部的攻击类型
total_attack_list = np.unique(total_dataframe[41])
# 部分的攻击类型
par_attack_list = np.unique(par_dataframe[41])

del par_dataframe


def nominal_encode(x, category):
    for index in range(len(category)):
        if x == category[index]:
            return index


total_dataframe[1] = total_dataframe[1].apply(nominal_encode, args=(protocol_list, ))
total_dataframe[2] = total_dataframe[2].apply(nominal_encode, args=(service_list, ))
total_dataframe[3] = total_dataframe[3].apply(nominal_encode, args=(flag_list, ))

# 提取未知攻击类型的数据
# ind为未知攻击类型的行索引列表
inds = []
Y = total_dataframe[41].values
for attack in total_attack_list:
    if attack in par_attack_list:
        pass
    else:
        inds.extend(np.where(attack == Y)[0])

dataset = np.delete(total_dataframe.values, inds, axis=0)

X = dataset[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 31, 32]]
X = StandardScaler().fit_transform(X)
Y_predicted = model.predict(X)

# 迁移网络训练数据
X = np.hstack((np.delete(dataset, [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 31, 32, 41], axis=1), Y_predicted))
Y = np.delete(Y, inds, axis=0)

# 保存数据
with open('/total_data_X.pk', 'w') as f:
    pickle.dump(X, f)
with open('/total_data_Y.pk', 'w') as f:
    pickle.dump(Y, f)

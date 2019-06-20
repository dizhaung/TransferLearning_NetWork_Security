#coding=utf-8
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout,Input
from keras.models import Model
from sklearn.feature_extraction import DictVectorizer

'''
keras_network.py用于训练知识发现的模型，将19种离散的特征最终通过softmax转化为分为23种概率特征。
'''


def baseline_model(num_of_features):
    def branch(x):
        x = Dense(int(np.floor(num_of_features * 5)), kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(0.75)(x)
        x = Dense(int(np.floor(num_of_features * 2)), kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(int(np.floor(num_of_features)), kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(0.1)(x)
        return x
    main_input = Input(shape=(num_of_features,), name='main_input')
    x = main_input
    x = branch(x)
    main_output = Dense(23, activation='softmax')(x)
    model = Model(input=main_input, output=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
    return model


# kddcup.data.corrected.csv为700万带标签具有23种攻击类型的数据
dataframe = pd.read_csv('/root/kddcup.data.corrected.csv', header=None)

# 统计协议类型，服务类型，状态类型，攻击类型的种类
protocol_list = np.unique(dataframe[1])
service_list = np.unique(dataframe[2])
flag_list = np.unique(dataframe[3])
attack_list = np.unique(dataframe[41])


# [攻击1，攻击2，攻击3，攻击4]  如果特征值为攻击3,则该值被替换为，攻击类型列表的索引
def nominal_encode(x, category):
    for index in range(len(category)):
        if x == category[index]:
            return index


# 对离散型的数据值应用上面的函数转换
dataframe[1] = dataframe[1].apply(nominal_encode, args=(protocol_list, ))
dataframe[2] = dataframe[2].apply(nominal_encode, args=(service_list, ))
dataframe[3] = dataframe[3].apply(nominal_encode, args=(flag_list, ))

# 从全部的数据种取出离散的数据
X = dataframe.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 31, 32]].values
Y = dataframe.iloc[:, 41].values

del dataframe

# 对数据进行标准化
X = StandardScaler().fit_transform(X)

# 通过DictVectorizer对Y进行编码
v = DictVectorizer(sparse=False)
D = [{'attack_cat': y} for y in Y]
labeled_Y = v.fit_transform(D)

# 构建模型
model = baseline_model(19)

# 训练模型
history = model.fit(X, labeled_Y, batch_size=1000, epochs=20)

# 存储模型
with open('/first_model.pk', 'w') as f:
    pickle.dump(model, f)
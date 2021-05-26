# 提供了对多维数组对象的支持，支持高级大量的维度数组与矩阵运算，也针对数组运算提供大量的数学函数库。
import numpy as np
# Pandas是一个强大的分析结构化数据的工具集；用于数据挖掘和数据分析，同时也提供数据清洗功能。
import pandas as pd
# os模块提供了多数操作系统的功能接口函数
import os
# 处理日期和时间
import arrow as ar
# 绘图
import matplotlib.pyplot as plt
# Seaborn是在matplotlib的基础上进行了更高级的API封装,从而使得作图更加容易,
import seaborn as sns
# 一个优化matplotlib函数操作的package, Matplotlib 中文支持组件
from pyplotz.pyplotz import PyplotZ

pltz = PyplotZ()
# 三种配色的调色板
from palettable.colorbrewer.sequential import Blues_9, BuGn_9, Greys_3, PuRd_5
# re模块是python独有的匹配字符串的模块,该模块中提供的很多功能是基于正则表达式实现的
import re
# 格式化日期和时间
import time
# 进度条库
from tqdm import tqdm
# 序列化库
import pickle
# 预处理库
from sklearn import preprocessing

# matplotlib支持中文
# plt.rcParams['font.sans-serif'] = ['Songti SC']  # 用来正常显示中文标签，字体根据自己电脑情况更改，如Windows可用SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 通过警告过滤器进行控制不发出警告消息
import warnings

warnings.filterwarnings('ignore')
# matplotlib中设置样式表
plt.style.use('fivethirtyeight')
# 直接在python console里面生成图像
# %matplotlib inline
# 目录
os.chdir('.')

# # round1
# train_path_people_1='./round1_train_cut_by_people.txt'
# train_path_type_1='./round1_train_cut_by_type.txt'
# test_path_a_1='./round1_ijcai_18_test_a_20180301.txt'
# test_path_b_1='./round1_ijcai_18_test_b_20180418.txt'
# # round2
# train_path_type_2='./round2_train_cut_by_type.txt'
# test_path_a_2='./round2_test_a.txt'
# test_path_b_2='./round2_test_b.txt'
# dataset_cut
train_path = '090.txt'
test_path = 'test_no_label.txt'
train = pd.read_table(train_path, delimiter=' ')
test = pd.read_table(test_path, delimiter=' ')

# 去除完全相同的行，inplace=True为原地更改
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

# train_ad_data 的长度
trainLen = len(train)
# 训练集是否交易的标签
trainlabel = train['is_trade']
# 测试集的样本ID
testInstanceID = test['instance_id']

# 使用字典方便存储
serialize_constant = {}
serialize_constant['trainLen'] = trainLen
serialize_constant['trainlabel'] = trainlabel
serialize_constant['testInstanceID'] = testInstanceID

# 使用pickle序列化
filename = './serialize_constant'

with open(filename, 'wb') as f:
    pickle.dump(serialize_constant, f)

# with open(filename, 'rb') as f:
#     serialize_constant = pickle.load(f)
#     trainLen = serialize_constant['trainLen']
#     trainlabel = serialize_constant['trainlabel']
#     testInstanceID = serialize_constant['testInstanceID']

# 将训练集和测试集合并在一起
key = list(test)  # test 的列名
mergeData = pd.concat([train, test], keys=key)
# 样本重新编号
mergeData = mergeData.reset_index(drop=True)


# 将timestamp转换成datetime【%Y-%m-%d %H:%M:%S】
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt  # str


# 时间，datetime64[ns]
mergeData['time'] = pd.to_datetime(mergeData.context_timestamp.apply(timestamp_datetime))

# 初赛是用18-24号来预测25号
# 复赛是用8/31-9/6全天和9/7上午来预测9/7下午的数据

mergeData['day'] = mergeData.time.dt.day
mergeData['hour'] = mergeData.time.dt.hour
mergeData['minute'] = mergeData.time.dt.minute

# ID从0重新编号
lbl = preprocessing.LabelEncoder()
for col in ['item_id', 'item_brand_id', 'item_city_id', 'shop_id', 'user_id']:
    mergeData[col] = lbl.fit_transform(mergeData[col])

# 保存为csv文件
mergeData.to_csv('./mergeData.csv', sep=' ')

# Pandas是一个强大的分析结构化数据的工具集；用于数据挖掘和数据分析，同时也提供数据清洗功能。
import pandas as pd
# os模块提供了多数操作系统的功能接口函数
import os
# 绘图
import matplotlib.pyplot as plt
# 一个优化matplotlib函数操作的package, Matplotlib 中文支持组件
from pyplotz.pyplotz import PyplotZ
pltz = PyplotZ()
# 三种配色的调色板
# re模块是python独有的匹配字符串的模块,该模块中提供的很多功能是基于正则表达式实现的
# 格式化日期和时间
import time
# 序列化库
import pickle
# 预处理库
from sklearn import preprocessing
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 通过警告过滤器进行控制不发出警告消息
import warnings
warnings.filterwarnings('ignore')
# matplotlib中设置样式表
plt.style.use('fivethirtyeight')
# 目录
os.chdir('.')

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
serialize_constant = {'trainLen': trainLen, 'trainlabel': trainlabel, 'testInstanceID': testInstanceID}

# 使用pickle序列化
filename = './serialize_constant'

with open(filename, 'wb') as f:
    pickle.dump(serialize_constant, f)

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

mergeData['day'] = mergeData.time.dt.day
mergeData['hour'] = mergeData.time.dt.hour
mergeData['minute'] = mergeData.time.dt.minute

# ID从0重新编号
lbl = preprocessing.LabelEncoder()
for col in ['item_id', 'item_brand_id', 'item_city_id', 'shop_id', 'user_id']:
    mergeData[col] = lbl.fit_transform(mergeData[col])

# 保存为csv文件
mergeData.to_csv('./mergeData.csv', sep=' ')

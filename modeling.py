import pandas as pd
import pickle
import xgboost as xgb
import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")

datapath = './featureData.csv'
data = pd.read_csv(datapath, sep=' ')

filename = './serialize_constant'

with open(filename, 'rb') as f:
    serialize_constant = pickle.load(f)
    trainLen = serialize_constant['trainLen']
    trainlabel = serialize_constant['trainlabel']
    testInstanceID = serialize_constant['testInstanceID']

data = data.iloc[0: trainLen, :]
target = trainlabel

# 可以选择的模型包括 'LGBMmodel' 'XGBModel'
# XGBModel
dtrain = xgb.DMatrix(data=data.values, label=target.values)
progress = dict()
# xgbparamSetting()
param = {
    'learning_rate': 0.01,
    'eta': 0.4,
    'max_depth': 3,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 1,
    # 'lambda' : 0.1,
    'nthread': 4,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
# XGBbestNumRounds
num_round = 811
bst = xgb.train(param, dtrain, num_round, evals_result=progress)
bst.save_model('./xgbModelFinal')

# LGBMmodel
dtrain = lgb.Dataset(data=data.values, label=target.values)
progress = dict()
# LGBMparamSetting()
param = {
    'learning_rate': 0.01,
    'num_leaves': 32,
    # 'eta' : 0.4,
    'subsample': 0.35,
    'colsample_bytree': 0.3,
    'nthread': 4,
    # 'lambda_l1' : 0.1,
    'objective': 'binary',
    'metric': 'binary_logloss'
}
# LGBMbestNumRounds
num_round = 2263
bst = lgb.train(param, dtrain, num_round, evals_result=progress)
bst.save_model('./lgbModelFinal')

datapath = './featureData.csv'
data = pd.read_csv(datapath, sep=' ')
print(data['instance_id'].nunique())

data = data.iloc[trainLen:, :]  # 训练集后应该是测试集

writefileName = './result.txt'

XGBmodel = xgb.Booster(model_file='./xgbModelFinal')
XGBpreds = XGBmodel.predict(xgb.DMatrix(data.values))
LGBMmodel = lgb.Booster(model_file='./lgbModelFinal')
LGBMpreds = LGBMmodel.predict(data.values)

preds = 0.49 * XGBpreds + 0.51 * LGBMpreds

sub = pd.DataFrame()
print(testInstanceID)
print(len(preds))
sub['instance_id'] = testInstanceID
sub['predicted_score'] = preds  # 已经解决实际预测数和测试集数量不一致问题：这是由于特征集加入时候导致的

sub.to_csv(writefileName, sep='\t', index=False, line_terminator='\r')

import pandas as pd

# 打印不省略
pd.set_option('display.max_columns', None)  # 列
pd.set_option('display.max_rows', None)  # 行
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import time

import warnings

warnings.filterwarnings("ignore")

datapath = './featureData.csv'
data = pd.read_csv(datapath, sep=' ')

# 解决模型不能处理非英文字符特征的问题
import re

data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

filename = './serialize_constant'

with open(filename, 'rb') as f:
    serialize_constant = pickle.load(f)
    trainLen = serialize_constant['trainLen']
    trainlabel = serialize_constant['trainlabel']
    testInstanceID = serialize_constant['testInstanceID']

data = data.iloc[0: trainLen, :]
target = trainlabel

data['target'] = np.array(target)

loglossList = []
avglogloss = 0
foldNum = 1

day24 = data[data['day'] == 24]

day18_23 = data[data['day'] < 24]
day19_23 = day18_23[data['day'] >= 18]

dataTrain = day19_23.drop(['target'], axis=1)
labelTrain = day19_23['target']

dataTest = day24.drop(['target'], axis=1)
labelTest = day24['target']

print('here')

dtrain = xgb.DMatrix(data=dataTrain.values, label=labelTrain.values)
dtest = xgb.DMatrix(data=dataTest.values, label=labelTest.values)

watchlist = [(dtrain, 'train'), (dtest, 'test')]
progress = dict()
# xgbparamSetting()
param = {
    'learning_rate': 0.05,
    'eta': 0.4,
    'max_depth': 3,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 1,
    # 'lambda' : 0.1,
    'nthread': 4,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
# treeNum
num_boost_round = 3000

bst = xgb.train(param, dtrain, num_boost_round, watchlist, early_stopping_rounds=150, evals_result=progress)

bst.save_model('./xgbModel' + str(foldNum))

xgb.plot_importance(bst)

logList = progress['test']['logloss']

tmplogloss = np.min(np.array(logList))
avglogloss = avglogloss + tmplogloss
loglossList.append(logList)

num = len(loglossList)
val = len(loglossList[0])
result = []
for j in range(val):
    cal = 0
    for i in range(num):
        tmplist = loglossList[i]
        cal = cal + tmplist[j]
        cal = cal / num
        result.append(cal)
print(result)

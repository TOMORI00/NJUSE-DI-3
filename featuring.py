# Pandas是一个强大的分析结构化数据的工具集；用于数据挖掘和数据分析，同时也提供数据清洗功能。
import pandas as pd
# 序列化库
import pickle
# 预处理库
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")



data = pd.read_csv('./mergeData.csv', sep=' ')
print(data.shape)
print(data['instance_id'].nunique())



# 添加一些有用特征

# item_category_list长度为2-3个;分割的id字符串，这里提取它的第2个种类
item_category_list_2 = pd.DataFrame([int(i.split(';')[1]) for i in data.item_category_list])

data['item_category_list_2'] = item_category_list_2
data.head()



'''
开始添加一些外部特征
'''

# 增加每个用户在每天、每天的每个时刻的点击广告商品的次数
user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
columns={0: 'user_query_day_hour'})
data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

# 增加每个广告商品被点击的频率
item_id_frequence = data.groupby([ 'item_id']).size().reset_index().rename(columns={0: 'item_id_frequence'})
item_id_frequence=item_id_frequence/(data.shape[0])
data = pd.merge(data, item_id_frequence, 'left', on=['item_id'])

# 增加每个用户每天每分钟点击广告商品的次数
num_user_minute = data.groupby(['user_id','day','minute']).size().reset_index().rename(columns = {0:'num_user_day_minute'})
data = pd.merge(data, num_user_minute,'left',on = ['user_id','day','minute'])

# 增加每天每个用户点击每个广告商品的次数
day_user_item_id = data.groupby(['day', 'user_id', 'item_id']).size().reset_index().rename(
columns={0: 'day_user_item_id'})
data = pd.merge(data, day_user_item_id, 'left', on=['day', 'user_id', 'item_id'])

# 增加每天每小时每分钟每个用户点击每个广告商品的次数
day_hour_minute_user_item_id = data.groupby(
['day', 'hour', 'minute', 'user_id', 'item_id']).size().reset_index().rename(
columns={0: 'day_hour_minute_user_item_id'})
data = pd.merge(data, day_hour_minute_user_item_id, 'left', on=['day', 'hour', 'minute', 'user_id', 'item_id'])

# 增加每天每小时每个用户点击每个广告商品的次数
number_day_hour_item_id = data.groupby(['day', 'hour', 'item_id']).size().reset_index().rename(
columns={0: 'number_day_hour_item_id'})
data = pd.merge(data, number_day_hour_item_id, 'left', on=['day', 'hour', 'item_id'])

# 增加每个广告商品被每个用户点击的次数
item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(columns={0: 'item_user_id'})
data = pd.merge(data, item_user_id, 'left', on=['item_id', 'user_id'])

data.head()



'''
开始添加一些新特征
'''

# 增加每个被点击广告商品类目在每个城市的次数
item_category_city_id = data.groupby(['item_category_list', 'item_city_id']).size().reset_index().rename(
columns={0: 'item_category_city_id'})
data = pd.merge(data, item_category_city_id, 'left', on=['item_category_list', 'item_city_id'])

# 增加每个被点击广告商品类目每种销量等级的次数，等级类数字越大程度越大
item_category_sales_level = data.groupby(
['item_category_list', 'item_sales_level']).size().reset_index().rename(
columns={0: 'item_category_sales_level'})
data = pd.merge(data, item_category_sales_level, 'left', on=['item_category_list', 'item_sales_level'])

# 增加每个被点击广告商品类目每种价格等级的次数
item_category_price_level = data.groupby(
['item_category_list', 'item_price_level']).size().reset_index().rename(
columns={0: 'item_category_price_level'})
data = pd.merge(data, item_category_price_level, 'left', on=['item_category_list', 'item_price_level'])

# 增加每个被点击广告商品每种销量等级的次数
item_ID_sales_level = data.groupby(['item_id', 'item_sales_level']).size().reset_index().rename(
columns={0: 'item_ID_sales_level'})
data = pd.merge(data, item_ID_sales_level, 'left', on=['item_id', 'item_sales_level'])

# 增加每个被点击广告商品每种收藏等级的次数
item_ID_collected_level = data.groupby(['item_id', 'item_collected_level']).size().reset_index().rename(
columns={0: 'item_ID_collected_level'})
data = pd.merge(data, item_ID_collected_level, 'left', on=['item_id', 'item_collected_level'])

data.head()

'''
开始添加一些危险特征
'''

# 增加每个用户出现次数
number_user_id = data.groupby(['user_id']).size().reset_index().rename(columns={0: 'number_user_id'})
data = pd.merge(data, number_user_id, 'left', on=['user_id'])

# 增加每个商品出现次数
number_shop_id = data.groupby(['shop_id']).size().reset_index().rename(columns={0: 'number_shop_id'})
data = pd.merge(data, number_shop_id, 'left', on=['shop_id'])

lbl = preprocessing.LabelEncoder()

# 把【预测的种类：属性列表】按照predict_category_property0..4提取成一列，属性从0重新编号，否则是空字符串
for i in range(5):
    data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

# 把【广告商品的类型列表】item_category_list1..2提取成一列，属性从0重新编号，否则是空字符串
for i in range(1, 3):
    data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

# 把【广告商品的属性列表】item_property_list0..9提取成一列，属性从0重新编号，否则是空字符串
for i in range(10):
    data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

# data['context_page0'] = data['context_page_id'].apply(
#   lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

data.head()



'''
对缺失值进行填充处理，都是填充众数
'''

# 性别填充为0女性
data['gender0'] = data['user_gender_id'].apply(lambda x: x + 1 if x == -1 else x)

# 年龄填充为1003，年龄范围是1000-1007
# print(data['user_age_level'].value_counts())
data['age0'] = data['user_age_level'].apply(lambda x: 1003 if x == -1  else x)

# 职业填充为2005，职业范围是2002-2005
# print(data['user_occupation_id'].value_counts())
data['occupation0'] = data['user_occupation_id'].apply(lambda x: 2005 if x == -1  else x)

# 星级填充为3006，星级范围是3000-3010
# print(data['user_star_level'].value_counts())
data['star0'] = data['user_star_level'].apply(lambda x: 3006 if x == -1 else x)

'''
开始添加一些新特征
'''

# 增加每个广告商品被每个用户点击的次数
number_item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(
    columns={0: 'number_item_user_id'})
data = pd.merge(data, number_item_user_id, 'left', on=['item_id', 'user_id'])

# 增加被点击广告商品的品牌的每种店铺评分出现次数
number_item_brand_positive_rate = data.groupby(
    ['item_brand_id', 'shop_review_positive_rate']).size().reset_index().rename(
    columns={0: 'number_item_brand_positive_rate'})
data = pd.merge(data, number_item_brand_positive_rate, 'left',
                on=['item_brand_id', 'shop_review_positive_rate'])

# 增加被点击广告商品的品牌的每种店铺星级出现次数
number_item_brand_shop_star = data.groupby(['item_brand_id', 'shop_star_level']).size().reset_index().rename(
    columns={0: 'number_item_brand_shop_star'})
data = pd.merge(data, number_item_brand_shop_star, 'left', on=['item_brand_id', 'shop_star_level'])

# 增加被点击广告商品的城市的每种被展示次数等级出现的次数
number_item_city_pv_level = data.groupby(['item_city_id', 'item_pv_level']).size().reset_index().rename(
    columns={0: 'number_item_city_pv_level'})
data = pd.merge(data, number_item_city_pv_level, 'left', on=['item_city_id', 'item_pv_level'])

# 增加被点击广告商品的城市的每个用户的点击次数
number_item_city_user_id = data.groupby(['item_city_id', 'user_id']).size().reset_index().rename(
    columns={0: 'number_item_city_user_id'})
data = pd.merge(data, number_item_city_user_id, 'left', on=['item_city_id', 'user_id'])

# 增加被点击广告商品在每个价格等级下的每个销量等级出现次数
number_item_price_sales_level = data.groupby(
    ['item_price_level', 'item_sales_level']).size().reset_index().rename(
    columns={0: 'number_item_price_sales_level'})
data = pd.merge(data, number_item_price_sales_level, 'left', on=['item_price_level', 'item_sales_level'])

# 增加被点击广告商品在每个预测类目属性下的每个销量等级出现次数
number_predict_category_sales_level = data.groupby(
    ['predict_category_property', 'item_sales_level']).size().reset_index().rename(
    columns={0: 'number_predict_category_sales_level'})
data = pd.merge(data, number_predict_category_sales_level, 'left',
                on=['predict_category_property', 'item_sales_level'])

# 增加被点击广告商品在每个收藏次数等级下的每个商铺出现次数
number_collected_shop_id = data.groupby(['item_collected_level', 'shop_id']).size().reset_index().rename(
    columns={0: 'number_collected_shop_id'})
data = pd.merge(data, number_collected_shop_id, 'left', on=['item_collected_level', 'shop_id'])

# 把【广告商品类目列表】按照每个类目提取成一列，否则是空格
for i in range(3):
    data['category_%d' % (i)] = data['item_category_list'].apply(
        lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")

# 把【广告商品属性列表】按照每个属性提取成一列，否则是空格
for i in range(3):
    data['property_%d' % (i)] = data['item_property_list'].apply(
        lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")

# 把【预测的类目：属性列表】按照每个类目取第一个属性提取成一列，否则是空格
for i in range(3):
    data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
        lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " ")

# 增加每个用户对应的点击广告商品、广告所属商铺、点击在第几天发生、广告所在页数的总数
# nunique返回不同值个数
for i in ['item_id', 'shop_id', 'day', 'context_page_id']:
    temp = data.groupby('user_id').nunique()[i].reset_index().rename(columns={i: 'number_' + i + '_query_user'})
    data = pd.merge(data, temp, 'left', on='user_id')

print(data.shape)
print(data['instance_id'].nunique())

data.head()



# 提取信息
basic_data = data[['instance_id']]
ad_information = data[
        ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level','item_property_list',
         'item_sales_level', 'item_collected_level', 'item_pv_level']]
user_information = data[
        ['user_id', 'user_age_level', 'user_star_level', 'user_occupation_id','user_gender_id']]
text_information = data[['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']]
shop_information = data[
        ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
         'shop_score_delivery', 'shop_score_description']]
external_information = data[
        ['time', 'day', 'hour', 'minute', 'user_query_day', 'user_query_day_hour', 'day_user_item_id', \
         'day_hour_minute_user_item_id',
         'number_day_hour_item_id', 'number_user_id', 'number_shop_id', \
         'item_category_list_2', 'item_user_id', 'item_category_city_id', 'item_category_sales_level', \
         'item_ID_sales_level', 'item_ID_collected_level', 'item_category_price_level', \
         'predict_category_property0', 'predict_category_property1', 'predict_category_property2', \
         'predict_category_property3', 'predict_category_property4', 'item_category_list1', \
         'item_category_list2', 'item_property_list0', 'item_property_list1', 'item_property_list2', \
         'item_property_list3', 'item_property_list4', 'item_property_list5', 'item_property_list6', \
         'item_property_list7', 'item_property_list8', 'item_property_list9', 'gender0', 'age0', \
         'occupation0', 'star0', 'number_item_brand_positive_rate', 'number_item_brand_shop_star', \
         'number_item_city_pv_level', 'number_item_city_user_id', 'number_item_price_sales_level', \
         'number_predict_category_sales_level', 'number_collected_shop_id'# ,'shop_score_delivery_round','number_item_id_query_user' ,'number_shop_id_query_user','number_day_query_user' ,'number_context_page_id_query_user'
         ]]

# 这些信息合并成一个结果
result = pd.concat(
    [basic_data, ad_information, user_information, text_information, shop_information, external_information],
    axis=1)

print(result.shape)
print(result['instance_id'].nunique())

result.head()

#
# # 以下代码作用包括：
# # 1、增加样本按某个属性分组后每天出现的时间先后顺序特征
# # 2、增加统计某个属性在前几天出现次数特征
#
# # 这段代码添加按照每天的每个用户点击广告商品的时间、每个用户点击每个广告商品的时间、每个用户
# # 点击广告所属每个店铺的时间进行排序的特征
# for d in range(18, 26):
#     # 取每一天的样本数据
#     df1 = result[result['day'] == d]
#
#     # df.rank(method='min')返回从小到大的排名的dataframe，若两个排名相同，取最小排名
#     # 排名是从1开始的
#     # 下面的代码按照groupby内容分组，在每组内对所有列进行排序
#     rnColumn_user = df1.groupby('user_id').rank(method='min')
#     rnColumn_user_item = df1.groupby(['user_id', 'item_id']).rank(method='min')
#     rnColumn_user_shop = df1.groupby(['user_id', 'shop_id']).rank(method='min')
#
#     # 用户出现、用户点击广告商品、用户点击广告所属商铺的时间先后
#     df1['user_id_order'] = rnColumn_user['context_timestamp']
#     df1['user_item_id_order'] = rnColumn_user_item['context_timestamp']
#     df1['user_shop_id_order'] = rnColumn_user_shop['context_timestamp']
#
#     # 准备合并的几列属性
#     df2 = df1[['user_id', 'instance_id', 'item_id', 'user_id_order', 'user_item_id_order', 'user_shop_id_order']]
#     if d == 18:
#         Df = df2
#     else:
#         Df = pd.concat([Df, df2])
#
# Df.drop_duplicates(inplace=True)
#
# result = pd.merge(result, Df, on=['user_id', 'instance_id', 'item_id'], how='left')
#
# print(result.shape)
# print(result['instance_id'].nunique())
# df1 = result.groupby(["instance_id"]).size()
# col = df1[df1 > 1].reset_index()[["instance_id"]]
# ttt = pd.merge(col, result, on=["instance_id"])
# print(ttt)
#
# # 添加训练集标签
# filename = './serialize_constant'
#
# with open(filename, 'rb') as f:
#     serialize_constant = pickle.load(f)
#     trainlabel = serialize_constant['trainlabel']
# result['is_trade'] = trainlabel
#
# print(result.shape)
# print(result['instance_id'].nunique())
#
# result.head()
#
# # 这段代码增加一些在第n天出现的某个属性，它在第n-1天出现的次数的特征
# for d in range(18, 26):
#     df1 = result[result['day'] == d - 1]  # 前一天
#     df2 = result[result['day'] == d]  # 当天
#
#     df_cvr = result[(result['day'] == d - 1) & (result['is_trade'] == 1)]  # 前一天且已经交易的
#
#     # 按照属性分组进行样本数量统计，然后转化为{column -> {index -> value}}的形式
#     user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
#     # item_trade_cnt = df1.groupby(['item_id','shop_id','is_trade']).count()['instance_id'].to_dict()
#     user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
#     item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
#     shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
#     item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
#     user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()
#
#     # 统计某个属性在第n天出现时候，它在第n-1天出现的次数
#     df2['item_cvr_cnt1'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
#     df2['user_cvr_cnt1'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))
#     df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
#     df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
#     df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
#     # tuple()变元组，axis=1对行进行操作
#     df2['user_item_cnt1'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)
#
#     # 取以下特征进行合并
#     df2 = df2[['user_item_cnt1', 'user_cnt1', \
#                # 'item_cnt1', 'shop_cnt1',\
#                'item_cvr_cnt1', 'user_cvr_cnt1', \
#                'item_id', 'user_id', 'instance_id']]
#     if d == 18:
#         Df2 = df2
#     else:
#         Df2 = pd.concat([df2, Df2])
#
# Df2.drop_duplicates(inplace=True)
#
# result = pd.merge(result, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')
#
# print(result.shape)
# print(result['instance_id'].nunique())
#
# result.head()
#
#
#
# # 这段代码增加一些在第n天出现的某个属性，它在第0..n-1天出现的次数的特征
# for d in range(18, 26):
#     df1 = result[result['day'] < d] # 前面所有天
#     df2 = result[result['day'] == d] # 当天
#
#     df_cvr = result[(result['day'] < d) & (result['is_trade'] == 1)] # 前面所有天且已经交易的
#
#     # 按照属性分组进行样本数量统计，然后转化为{column -> {index -> value}}的形式
#     user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
#     user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
#     item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
#     user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()
#
#     # 统计某个属性在第n天出现时候，它在第0..n-1天出现的次数
#     df2['item_cvr_cntx'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
#     df2['user_cvr_cntx'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))
#     df2['user_item_cntx'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1) # tuple()变元组，axis=1对行进行操作
#     df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
#
#     # 取以下特征进行合并
#     df2 = df2[['user_item_cntx', 'user_cntx',
#        'item_cvr_cntx', 'user_cvr_cntx', \
#        'item_id', 'user_id', 'instance_id']]
#
#     if d == 18:
#         Df2 = df2
#     else:
#         Df2 = pd.concat([df2, Df2])
#
# Df2.drop_duplicates(inplace=True)
#
# result = pd.merge(result, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')
#
# print(result.shape)
# print(result['instance_id'].nunique())
#
# result.head()



# 扔掉以下列
result = result.drop(
    ['item_category_list', 'item_property_list', 'predict_category_property', 'time']
    #,'instance_id']
    , axis=1)

# result.drop_duplicates(inplace=True)

# 保存
result.to_csv('./featureData.csv', sep=' ')
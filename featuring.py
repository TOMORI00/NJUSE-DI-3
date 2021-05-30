# Pandas是一个强大的分析结构化数据的工具集；用于数据挖掘和数据分析，同时也提供数据清洗功能。
import warnings

import pandas as pd
# 预处理库
from sklearn import preprocessing

warnings.filterwarnings("ignore")

data = pd.read_csv('./mergeData.csv', sep=' ')
print(data.shape)
print(data['instance_id'].nunique())

# 添加一些特征

# item_category_list长度为2-3个;分割的id字符串，这里提取它的第1个种类
item_category_list_1 = pd.DataFrame([int(i.split(';')[0]) for i in data.item_category_list])
data['item_category_list_1'] = item_category_list_1

# item_category_list长度为2-3个;分割的id字符串，这里提取它的第2个种类
item_category_list_2 = pd.DataFrame([int(i.split(';')[1]) for i in data.item_category_list])
data['item_category_list_2'] = item_category_list_2

'''
开始添加一些外部特征
'''

# 增加每天每个用户点击每个广告商品的次数
day_user_item_id = data.groupby(['day', 'user_id', 'item_id']).size().reset_index().rename(
    columns={0: 'day_user_item_id'})
data = pd.merge(data, day_user_item_id, 'left', on=['day', 'user_id', 'item_id'])

# 增加每天每小时每个用户点击每个广告商品的次数
number_day_hour_item_id = data.groupby(['day', 'hour', 'item_id']).size().reset_index().rename(
    columns={0: 'number_day_hour_item_id'})
data = pd.merge(data, number_day_hour_item_id, 'left', on=['day', 'hour', 'item_id'])

# 增加每个用户每天每分钟点击广告商品的次数
num_user_minute = data.groupby(['user_id', 'day', 'minute']).size().reset_index().rename(
    columns={0: 'num_user_day_minute'})
data = pd.merge(data, num_user_minute, 'left', on=['user_id', 'day', 'minute'])

# 增加每天每小时每分钟每个用户点击每个广告商品的次数
day_hour_minute_user_item_id = data.groupby(
    ['day', 'hour', 'minute', 'user_id', 'item_id']).size().reset_index().rename(
    columns={0: 'day_hour_minute_user_item_id'})
data = pd.merge(data, day_hour_minute_user_item_id, 'left', on=['day', 'hour', 'minute', 'user_id', 'item_id'])

# 增加每个用户在每天、每天的每个时刻的点击广告商品的次数
user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
    columns={0: 'user_query_day_hour'})
data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

# 增加每个广告商品被点击的频率
item_id_frequence = data.groupby(['item_id']).size().reset_index().rename(columns={0: 'item_id_frequence'})
item_id_frequence = item_id_frequence / (data.shape[0])
data = pd.merge(data, item_id_frequence, 'left', on=['item_id'])

# 增加每个广告商品被每个用户点击的次数
item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(columns={0: 'item_user_id'})
data = pd.merge(data, item_user_id, 'left', on=['item_id', 'user_id'])

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

'''
对缺失值进行填充处理，都是填充众数
'''

# 性别填充为0女性
data['gender0'] = data['user_gender_id'].apply(lambda x: x + 1 if x == -1 else x)

# 年龄填充为1003，年龄范围是1000-1007
# print(data['user_age_level'].value_counts())
data['age0'] = data['user_age_level'].apply(lambda x: 1003 if x == -1 else x)

# 职业填充为2005，职业范围是2002-2005
# print(data['user_occupation_id'].value_counts())
data['occupation0'] = data['user_occupation_id'].apply(lambda x: 2005 if x == -1 else x)

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

# 增加一些在第n天出现的某个属性，它在第0..n-1天出现的次数的特征
for d in range(18, 26):
    df1 = data[data['day'] < d]  # 前面所有天
    df2 = data[data['day'] == d]  # 当天
    df_cvr = data[(data['day'] < d) & (data['is_trade'] == 1)]  # 前面所有天且已经交易的
    # 按照属性分组进行样本数量统计，然后转化为{column -> {index -> value}}的形式
    user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
    user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
    item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
    user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()
    # 统计某个属性在第n天出现时候，它在第0..n-1天出现的次数
    df2['item_cvr_cntx'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
    df2['user_cvr_cntx'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))
    df2['user_item_cntx'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)
    df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
    # 取以下特征进行合并
    df2 = df2[['user_item_cntx', 'user_cntx',
               'item_cvr_cntx', 'user_cvr_cntx',
               'item_id', 'user_id', 'instance_id']]
    if d == 18:
        Df2 = df2
    else:
        Df2 = pd.concat([df2, Df2])
Df2.drop_duplicates(inplace=True)
data = pd.merge(data, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

# 增加额外特征
'''
一些额外特征有着很好的性能表现，比如广告商品是店铺某种类中最贵还是最便宜这个特征
'''
# 增加特征：被展示的广告商品中销量比例
data['sales_div_pv'] = data.item_sales_level / (1 + data.item_pv_level)
# na_action='ignore'表示如果x是NaN值就忽略掉
data['sales_div_pv'] = data.sales_div_pv.map(lambda x: int(10 * x), na_action='ignore')

# 增加特征：每天广告商品被点击的总次数
number_click_day = data.groupby(['day']).size().reset_index().rename(columns={0: 'number_click_day'})
data = pd.merge(data, number_click_day, 'left', on=['day'])

# 增加特征：每小时广告商品被点击的总次数
number_click_hour = data.groupby(['hour']).size().reset_index().rename(columns={0: 'number_click_hour'})
data = pd.merge(data, number_click_hour, 'left', on=['hour'])

# 增加特征：每个广告商品被用户点击，这些用户年龄的不同值个数，值越大说明广告覆盖的人群更广
# nunique返回不同值个数
temp = data.groupby('item_id')['user_age_level'].nunique().reset_index().rename(
    columns={'user_age_level': 'number_' + 'user_age_level' + '_query_item'})
data = pd.merge(data, temp, 'left', on=['item_id'])

# 增加特征：每个种类的每个广告商品被点击的次数
# 注意：item_category_list是按照树形的方式展开的，因此item_category_list_1是最粗的，所以没有进行分析
number_category_item = data.groupby(['item_category_list_2', 'item_id']).size().reset_index().rename(
    columns={0: 'number_category_item'})
data = pd.merge(data, number_category_item, 'left', on=['item_category_list_2', 'item_id'])

# 增加特征：每个种类的广告商品被点击的次数，以种类为单位
number_category2 = data.groupby(['item_category_list_2']).size().reset_index().rename(columns={0: 'number_category2'})
data = pd.merge(data, number_category2, 'left', on=['item_category_list_2'])

# 增加特征：每个广告商品被点击的次数在商品所属种类的比例
data['prob_item_id_category2'] = data['number_category_item'] / data['number_category2']

# 扔掉number_category2和number_category_item两个特征
data = data.drop(['number_category2', 'number_category_item'], axis=1)

# 增加特征：每个种类的每个广告商品平均价格等级
ave_price_category_item = data.groupby(['item_category_list_2', 'item_id']).mean()[
    'item_price_level'].reset_index().rename(columns={'item_price_level': 'ave_price_category_item'})
data = pd.merge(data, ave_price_category_item, 'left', on=['item_category_list_2', 'item_id'])

# 增加特征：每个种类的商品平均价格等级，以种类为单位
ave_price_category = data.groupby(['item_category_list_2']).mean()['item_price_level'].reset_index().rename(
    columns={'item_price_level': 'ave_price_category'})
data = pd.merge(data, ave_price_category, 'left', on=['item_category_list_2'])

# 增加特征：每个广告商品价格在商品所属种类的平均价格的比例
data['prob_item_price_to_ave_category2'] = data['item_price_level'] / data['ave_price_category']

# 增加特征：每个种类的每个广告商品的每种价格下的销量平均值
ave_sales_price_category_item = data.groupby(['item_category_list_2', 'item_id', 'item_price_level']).mean()[
    'item_sales_level'].reset_index().rename(columns={'item_sales_level': 'ave_sales_price_category_item'})
data = pd.merge(data, ave_sales_price_category_item, 'left', on=['item_category_list_2', 'item_id', 'item_price_level'])

# 增加特征：每个种类的广告商品的销量平均值，以种类为单位
ave_sales_level_category = data.groupby(['item_category_list_2']).mean()['item_sales_level'].reset_index().rename(
    columns={'item_sales_level': 'ave_sales_level_category'})
data = pd.merge(data, ave_sales_level_category, 'left', on=['item_category_list_2'])

# 每个广告商品销量在商品所属种类的平均销量的比例
data['prob_ave_category_sales_item_sales'] = data['item_sales_level'] / data['ave_sales_level_category']

# 增加特征：广告商品所属种类中商品价格最高值
max_price_category = data.groupby(['item_category_list_2'])['item_price_level'].max().reset_index().rename(
    columns={'item_price_level': 'max_price_category'})
data = pd.merge(data, max_price_category, 'left', on=['item_category_list_2'])

# 增加特征：广告商品价格占所属种类中商品价格最高值比例，并向下取整
data['is_max_price_category'] = data['item_price_level'] / data['max_price_category']
data['is_max_price_category'] = data['is_max_price_category'].map(lambda x: int(x), na_action='ignore')

# 增加特征：广告商品所属种类中商品价格最低值
min_price_category = data.groupby(['item_category_list_2'])['item_price_level'].min().reset_index().rename(
    columns={'item_price_level': 'min_price_category'})
data = pd.merge(data, min_price_category, 'left', on=['item_category_list_2'])

# 增加特征：广告商品所属种类中商品价格最低值与该商品价格的比例，并向下取整
data['is_min_price_category'] = data['min_price_category'] / data['item_price_level']
data['is_min_price_category'] = data['is_min_price_category'].map(lambda x: int(x), na_action='ignore')

# 扔掉max_price_category和min_price_category两个特征
data = data.drop(['max_price_category', 'min_price_category'], axis=1)

# 增加特征：广告商品所属种类中商品销量最高值
max_sales_category = data.groupby(['item_category_list_2'])['item_sales_level'].max().reset_index().rename(
    columns={'item_sales_level': 'max_sales_category'})
data = pd.merge(data, max_sales_category, 'left', on=['item_category_list_2'])

# 增加特征：广告商品销量占所属种类中商品销量最高值比例，并向下取整
data['is_max_sales_category'] = data['item_sales_level'] / data['max_sales_category']
data['is_max_sales_category'] = data['is_max_sales_category'].map(lambda x: int(x), na_action='ignore')

# 增加特征：广告商品所属种类中商品销量最低值
min_sales_category = data.groupby(['item_category_list_2'])['item_sales_level'].min().reset_index().rename(
    columns={'item_sales_level': 'min_sales_category'})
data = pd.merge(data, min_sales_category, 'left', on=['item_category_list_2'])

# 增加特征：广告商品所属种类中商品销量最低值与该商品销量的比例，并向下取整
data['is_min_sales_category'] = data['min_sales_category'] / data['item_sales_level']
data['is_min_sales_category'] = data['is_min_sales_category'].map(lambda x: int(x), na_action='ignore')

# 扔掉max_sales_category和min_sales_category两个特征
data = data.drop(['max_sales_category', 'min_sales_category'], axis=1)

print(data.shape)
print(data['instance_id'].nunique())

data.head()

print(data.shape)
print(data['instance_id'].nunique())

# 提取信息
basic_data = data[['instance_id']]
ad_information = data[
    ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_property_list',
     'item_sales_level', 'item_collected_level', 'item_pv_level']]
user_information = data[
    ['user_id', 'user_age_level', 'user_star_level', 'user_occupation_id', 'user_gender_id']]
text_information = data[['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']]
shop_information = data[
    ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
     'shop_score_delivery', 'shop_score_description']]
external_information = data[
    ['time', 'day', 'hour', 'minute', 'user_query_day', 'user_query_day_hour', 'day_user_item_id',
     'day_hour_minute_user_item_id',
     'number_day_hour_item_id', 'number_user_id', 'number_shop_id', 'item_category_list_1',
     'item_category_list_2', 'item_user_id', 'item_category_city_id', 'item_category_sales_level',
     'item_ID_sales_level', 'item_ID_collected_level', 'item_category_price_level',
     'predict_category_property0', 'predict_category_property1', 'predict_category_property2',
     'predict_category_property3', 'predict_category_property4', 'item_category_list1',
     'item_category_list2', 'item_property_list0', 'item_property_list1', 'item_property_list2',
     'item_property_list3', 'item_property_list4', 'item_property_list5', 'item_property_list6',
     'item_property_list7', 'item_property_list8', 'item_property_list9', 'gender0', 'age0',
     'occupation0', 'star0', 'number_item_brand_positive_rate', 'number_item_brand_shop_star',
     'number_item_city_pv_level', 'number_item_city_user_id', 'number_item_price_sales_level',
     'number_predict_category_sales_level', 'number_collected_shop_id',
     'user_item_cntx', 'user_cntx', 'item_cvr_cntx', 'user_cvr_cntx',
     'sales_div_pv', 'number_click_day', 'number_click_hour', 'user_age_level',
     'prob_item_id_category2',
     'ave_price_category_item', 'ave_price_category', 'prob_item_price_to_ave_category2',
     'ave_sales_price_category_item', 'ave_sales_level_category', 'prob_ave_category_sales_item_sales',
     'is_max_price_category', 'is_min_price_category'
     ]]

# 这些信息合并成一个结果
data = pd.concat(
    [basic_data, ad_information, user_information, text_information, shop_information, external_information],
    axis=1)
print(data.shape)
print(data['instance_id'].nunique())

# 扔掉以下列
data = data.drop(
    ['item_category_list', 'item_property_list', 'predict_category_property', 'time']
    , axis=1)

# data.drop_duplicates(inplace=True)

# 保存
data.to_csv('./featureData.csv', sep=' ')

print(data.shape)
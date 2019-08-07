import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data.csv', encoding='gbk')

# 查看数据总体情况
# data.describe()

# 查看各个列之间的相关性
# cor = data.corr()
# print(cor['status'][abs(cor['status'])<0.01])

# 第一步：删除无关特征
"""
    删去的特征包括：
    * 用户身份特征：交易单号、银行卡号、用户id、名字等
    * 日期相关：看了所给的特征中有相应的天数，所以日期特征可以删去
    * 相关性小的特征：利用corr函数查看任意两列间的相关性，去除了与status相关性小于0.01的特征
    * 取值单一的特征：取值只有2~3个的特征
"""
irrelevant_feature = ['Unnamed: 0', 'custid', 'trade_no', 'bank_card_no', 'source', 'id_name', 
                      'student_feature', 'is_high_user', 'take_amount_in_later_12_month_highest',
                     'trans_amount_increase_rate_lately', 'transd_mcc', 'trans_days_interval_filter',
                     'jewelry_consume_count_last_6_month', 'query_finance_count', 'latest_six_month_apply',
                     'loans_credibility_behavior', 'first_transaction_time', 'historical_trans_amount',
                     'historical_trans_day', 'latest_query_time', 'loans_latest_time',
                     'railway_consume_count_last_12_month']
data = data.drop(irrelevant_feature, axis=1)

# 第二步：类型转换
data = pd.get_dummies(data)

# 第三步：缺失值处理
# 删去了有效特征小于50的样本
data = data.dropna(thresh=50)
# 填充缺失值，使用该列均值填充
data = data.fillna(data.mean())

# 保存到csv文件
data.to_csv("new_data.csv",sep=',')
# print(data)

# 划分训练集和测试集
x = data.drop('status', axis=1)
y = data['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

# 保存
x_train.to_csv("x_train.csv", sep=',')
x_test.to_csv("x_test.csv", sep=',')
y_train.to_csv("y_train.csv", header=True, sep=',')
y_test.to_csv("y_test.csv", header=True, sep=',')
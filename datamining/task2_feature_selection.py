# """
#     任务2
#     * 特征衍生
#     * 特征挑选：分别用IV值和随机森林等进行特征选择
# """

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

x_train = pd.read_csv("x_train.csv")
x_test = pd.read_csv("x_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")
print(x_train.shape)


# 1. 特征衍生：需要从业务上理解各个特征的关系
# 暂时没有思路

# 2. 特征挑选

# 利用IV值进行挑选


# 利用RF对特征值进行挑选
clf = RandomForestClassifier(n_estimators=100, bootstrap = True, oob_score = True, criterion = 'gini')
clf.fit(x_train, y_train)

# 特征挑选前的准确率
acc = clf.score(x_test, y_test)
print('before feature selection:')
print('acc: %f' % acc)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
# col_list = x_train.columns.tolist()
# # 打印各个特征的重要性
# for f in range(x_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, col_list[indices[f]], importances[indices[f]]))

# # 去掉后10个特征
# del_col_list = [col_list for ix in indices[-10:]]

# 删去重要性小于threshold的特征
threshold = 0.005
del_col_list = list(x_train.columns[importances < threshold])
x_train_1 = x_train.drop(del_col_list, axis=1)
x_test_1 = x_test.drop(del_col_list, axis=1)

print(x_train_1.shape)

# # 保存
# x_train_1.to_csv("x_train_1.csv", index=False, sep=',')
# x_test_1.to_csv("x_test_1.csv", index=False, sep=',')


# 特征挑选后的准确率
clf1 = RandomForestClassifier(n_estimators=100, bootstrap = True, oob_score = True, criterion = 'gini')
clf1.fit (x_train_1, y_train)

acc = clf1.score(x_test_1, y_test)
print('after feature selection:')
print('acc: %f' % acc)
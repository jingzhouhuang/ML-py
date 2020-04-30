import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')
### 一般我们机器学习的流程

# 一、加载数据
#processs_datas/train_process_datas_x.csv
datas_x = pd.read_csv('./processs_datas/train_process_datas_x.csv')
# print(datas_x.head())
# print(datas_x.info(verbose=True,null_counts=True))
# processs_datas/train_datas_Y.csv
datas_y = pd.read_csv('./processs_datas/train_datas_Y.csv')
# print(datas_y.head())
# print(datas_y.info())
# print(datas_y['happiness'].value_counts())

### 标签里面存在异常值 -8
# ## 一般情况下对于标签存在异常的数据需要删除，注意对应的X也要删除
# ## 先合并X,Y,再删除
# datas = pd.concat([datas_x,datas_y],axis=1)
# # print(datas.head())
# # print(datas.info(verbose=True,null_counts=True))
# datas.replace(-8,np.nan,inplace=True)
# datas.dropna(axis=0,how='any',inplace=True)
# # print(datas.shape)
# # print(datas['happiness'].value_counts(dropna=True))

## 这里我们可以考虑将-8替换成3（特殊情况特殊对待）
datas_y.replace(-8,3,inplace=True)
# print(datas_y.shape)
# print(datas_y['happiness'].value_counts())

# 二、数据清洗与预处理  （这里在dataprocess里面完成了）
# 删除ID字段
datas_x.drop(['id'],axis=1,inplace=True)


# 三、获取数据的特征属性X和目标属性Y



# 四、数据分割
x_train,x_test,y_train,y_test = train_test_split(datas_x,datas_y,test_size=0.2,random_state=11)
# print(x_train.shape)
# print(x_test.shape)

# 五、特征工程  这里我们做个PCA降维 ## 使用管道流
# 六、模型构建
##我们这里使用网格交叉验证,随机森林分类（也可以使用回归：评估指标是mse）
'''
n_estimators=10, 决策树的棵数
criterion="gini", 
max_depth=None,  决策树的深度
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.,
max_features="auto",
max_leaf_nodes=None,
min_impurity_decrease=0.,
min_impurity_split=None,
bootstrap=True,
oob_score=False,
n_jobs=1,
random_state=None,
verbose=0,
warm_start=False,
class_weight=None 权重
'''
# pipe = Pipeline([('pca',PCA()),
#                  ('RF',RandomForestClassifier())])
#
# params = {
#     'pca__n_components':[0.6,0.7,0.8,0.9],
#     'RF__n_estimators':[100,200,500],
#     'RF__max_depth':[3,5,7,9],
#     'RF__class_weight':['balanced','balanced_subsample',None]
# }
# model = GridSearchCV(estimator=pipe,param_grid=params,cv=5)
# model.fit(datas_x,datas_y)
# print('最优模型参数：{}'.format(model.best_params_))
# print('最优模型的评分：{}'.format(model.best_score_))
# ## 保存一下最优的参数和评分
# with open('./model/pca_rf_params.txt','w',encoding='utf-8') as writer:
#     writer.write('最优模型参数：{}'.format(model.best_params_)+'\n'+'最优模型的评分：{}'.format(model.best_score_))

model = Pipeline([('pca',PCA(n_components=0.6)),
                 ('RF',RandomForestClassifier(n_estimators=100,max_depth=3))])

# 七、模型训练
model.fit(x_train,y_train)
# 八、模型评估
print('训练集准确率:',model.score(x_train,y_train))
print('测试集准确率:',model.score(x_test,y_test))
y_train_hat = model.predict(x_train)
y_test_hat = model.predict(x_test)
print('训练集MSE：',mean_squared_error(y_train,y_train_hat))
print('测试集MSE：',mean_squared_error(y_test,y_test_hat))


# 九、模型持久化
## 实际上对于最终我们保存的模型来说，我们需要把所有数据进行训练，再保存模型

model.fit(datas_x,datas_y)
print(model.score(datas_x,datas_y))
datas_y_hat = model.predict(datas_x)
print(mean_squared_error(datas_y,datas_y_hat))

joblib.dump(model,'./model/pca_rf.pkl')

## todo:使用其他的模型训练,也可以考虑使用回归的模型（建议LightGBM）

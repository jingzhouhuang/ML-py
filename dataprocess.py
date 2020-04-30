import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# 设置忽略警告
import warnings
warnings.filterwarnings('ignore')
import sys
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

### 设置不适用科学计数法  #为了直观的显示数字，不采用科学计数法
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

### 获取原始数据 完整版的数据
# datas/happiness_train_complete.csv
# datas/happiness_test_complete.csv
train_datas = pd.read_csv('./datas/happiness_train_complete.csv',sep=',',encoding='gb2312')
test_datas = pd.read_csv('./datas/happiness_test_complete.csv',sep=',',encoding='gb2312')
# print(train_datas.shape)
# print(test_datas.shape)
# print(train_datas.head())
# print(train_datas.info(verbose=True,null_counts=True))
train_datas_Y = train_datas[['happiness']]
# print(train_datas_Y['happiness'].value_counts())
# print(train_datas_Y[:5])
# print('---------------')
# train_datas_Y1 = train_datas.iloc[:,1]
# print(train_datas_Y1[:5])
train_datas_X = train_datas.drop(['happiness'],axis=1)
# print(train_datas_X.head())
# print(train_datas_X.shape)
all_datas = pd.concat(([train_datas_X,test_datas]),axis=0,ignore_index=True)
# print(all_datas.shape)
'''
数据处理的思路：
    0、先删除一些缺省比较多的记录，或者特征？（在训练阶段没有标签的数据我们也是不能要的）
    1、时间的数据需要进行时间数据的提取 time striptime()
    2、连续的数据进行分段处理===》转换为离散数据
    3、文本数据的提取：oneHOT，TFIDF，词向量  （文本向量化）
    4、对连续型的数据进行缺失值的填充 一般可以填充为0，均值，众数，中位数，经验数据
    5、对离散数据进行缺失值的填充 (缺失值单独作为一个类别，根据标签类别数量进行填充，经验数据) 哑编码
    对于我们处理好后的数据，还可以进行一个相关性的检验，对于一些相关系数绝对值比较大的特征只保留其中一个特征（可以考虑的，但是不是说效果就一定好，这样做好处是特征降维了，节约我们的训练成本）
    PCA，标准化，归一化特征工程这些工作是在我们把数据进行预处理后（数值型的特征）在进行的
'''
# print(all_datas.head())
# print('-------------')
# print(all_datas.tail())

### 在统计nan值前对数据中的异常值进行替换成nan值
## -1 = 不适用; -2 = 不知道; -3 = 拒绝回答; -8 = 无法回答;
all_datas.replace([-1,-2,-3,-8],np.nan,inplace=True)

# print(all_datas.info(verbose=True,null_counts=True))
###查看所有字段的缺失值数量
# print(all_datas.isnull().sum())
nan_num = all_datas.isnull().sum()
# 获取缺失值为0的字段名称 是不需要进行缺失值的填充 bool索引
# TODO：但是有可能要进行其他的处理
nan_num0_list = list(nan_num[nan_num==0].index)
# print(nan_num0_list)
# print(len(nan_num0_list))
# print(all_datas.head())
df_nan0 = all_datas[nan_num0_list]
# print(df_nan0.head())

# 删除survey_time
# todo：其实 可以对数据进行时间的提取 年份，时间段 time striptime()
all_datas.drop(['survey_time'],axis=1,inplace=True)

##todo: 根据survey_time提取的时间求出年龄  可以不做分段

### 这里我们做的是将连续的年份数据进行了离散化
birth = all_datas['birth']
# print(birth.value_counts())
# 画图展示一下'birth'这个数据的值得统计
# plt.bar(birth.value_counts().index,birth.value_counts().values,label='birth')
# plt.legend()
# plt.show()
def birth_split(x):
    if x<=1930:
        return 0
    elif x<=1940:
        return 1
    elif x<=1950:
        return 2
    elif x<=1960:
        return 3
    elif x<=1970:
        return 4
    elif x<=1980:
        return 5
    elif x<=1990:
        return 6
    else:
        return 7

    pass
## map
all_datas['birth_s']=all_datas['birth'].map(birth_split)
# print(all_datas['birth_s'].value_counts())
all_datas.drop(['birth'],axis=1,inplace=True)
# print(all_datas.info(verbose=True,null_counts=True))
# sys.exit()

# 对于一些缺失值比较严重的数据我们先进行删除工作 大于k（一个阈值）的进行删除
nan_numk_list = list(nan_num[nan_num>=9000].index)
# print(nan_numk_list)
## inplace=True 在原对象上进行修改，不需要在赋值的
# all_datas.drop(nan_numk_list,axis=1,inplace=True)
# inplace=False 需要重新赋值给一个对象
# print(all_datas.shape)
all_datas = all_datas.drop(nan_numk_list,axis=1,inplace=False)
# print(all_datas.shape)

### 对于缺失值不严重的数据的处理
# nan_num = all_datas.isnull().sum()
# nan_num02k_list = list(nan_num[nan_num>0].index)
nan_num02k_list = list(nan_num[(nan_num>0)&(nan_num<9000)].index)
# print(nan_num02k_list)
# nan_num02k_df = all_datas[nan_num02k_list]
# print(nan_num02k_df.head())

# 对于一些虽然缺失不严重，但是是无效特征的数据进行删除   基于业务场景理解
# print(all_datas.shape)

##缺失值在我们允许的阈值内的数据 进行缺失值的填充处理 ==》考虑离散值还是连续值
# 查看每个字段的所取值的个数 初步判断离散的数据和连续的数据
# 查看字段分别对离散的字段和连续的字段做填充处理

list01 = [] ## 用来存储我们离散的字段
list02 = [] ## 用来存储连续的字段
### 我认为如果你这个字段的值大于40个就是连续的
for column in nan_num02k_list:
    # print(all_datas[column].value_counts(dropna=False).shape[0])
    if all_datas[column].value_counts(dropna=False).shape[0]<=40:
        list01.append(column)
    else:
        list02.append(column)
print('list01:\n',list01)
print('---------------------------------')
print('list02:\n',list02)

## todo：年份的数据转化成年龄
## 我在这里只做了一个离散化
# 处理'edu_yr'，'s_birth'，f_birth，m_birth，marital_1st，marital_now
# print(all_datas['marital_1st'].value_counts(dropna=False))
## 年份的数据 参考birth的数据处理方式
## 'edu_yr'根据是否有edu_yr 是否已完成学业

# print(all_datas['edu_yr'].value_counts(dropna=False)) ### NaN        4435
all_datas['has_edu_yer'] = all_datas['edu_yr'].map(lambda x:1 if x>0 else 0)
# print(all_datas['has_edu_yer'].value_counts(dropna=False))
## todo:做一下毕业时的年纪 （是否有用，其实我也不知道）
all_datas.drop(['edu_yr'],axis=1,inplace=True)

# 's_birth'，f_birth，m_birth，marital_1st，marital_now
## 不能直接用birth_split(x)，1、NAN  2、年份不一样，要去看年份的一个分布情况
# plt.bar(all_datas['s_birth'].value_counts().index,all_datas['s_birth'].value_counts().values,label='s_birth')
# plt.legend()
# plt.show()
# print(all_datas['s_birth'].value_counts(dropna=False))  ## NaN        2367
def s_birth_split(x):
    # print(x)
    if x>0:
        if x<=1930:
            return 1
        elif x<=1940:
            return 2
        elif x<=1950:
            return 3
        elif x<=1960:
            return 4
        elif x<=1970:
            return 5
        elif x<=1980:
            return 6
        elif x<=1990:
            return 7
        else:
            return 8
    else:
        return 0

all_datas['s_birth_split'] = all_datas['s_birth'].map(s_birth_split)
# print(all_datas['s_birth_split'].value_counts(dropna=False))
all_datas.drop(['s_birth'],axis=1,inplace=True)
## todo:f_birth，m_birth，marital_1st，marital_now

### 查看f_birth
# print(all_datas.f_birth.value_counts(dropna=False))
# plt.bar(all_datas.f_birth.value_counts().index,all_datas.f_birth.value_counts().values,label='f_birth')
# plt.legend()
# plt.show()
### 将f_birth 字段nan值处理为0，其他的按照小于1900，1910,1920。。赋值
def f_birth_split(x):
    if x>0:
        if x<1900:
            return 1
        elif 1900<=x<1910:
            return 2
        elif x<1920:
            return 3
        elif x<1930:
            return 4
        elif x<1940:
            return 5
        elif x<1950:
            return 6
        elif x<1960:
            return 7
        else:
            return 8
    else:
        # print('x为空',x)
        # sys.exit()
        return 0
all_datas["f_birth_s"] = all_datas["f_birth"].map(f_birth_split)
# print(all_datas.f_birth_s.value_counts(dropna=False))
all_datas.drop(["f_birth"],axis=1,inplace=True)

### 查看处理m_birth
# print(all_datas.m_birth.value_counts(dropna=False))
# plt.bar(all_datas.m_birth.value_counts().index,all_datas.m_birth.value_counts().values,label='m_birth')
# plt.legend()
# plt.show()
### 将m_birth 字段nan值处理为0，其他的按照小于1900，1910,1920。。赋值
def m_birth_split(x):
    if x>0:
        if x<1900:
            return 1
        elif x<1910:
            return 2
        elif x<1920:
            return 3
        elif x<1930:
            return 4
        elif x<1940:
            return 5
        elif x<1950:
            return 6
        elif x<1960:
            return 7
        else:
            return 8
    else:
        return 0
all_datas["m_birth_s"] = all_datas["m_birth"].map(m_birth_split)
# print(all_datas.m_birth_s.value_counts(dropna=False))
all_datas.drop(["m_birth"],axis=1,inplace=True)

''''''
#### 对'marital_1st', 'marital_now'进行处理 结婚的时间
# # print(all_datas['marital_1st'].value_counts(dropna=False))
# plt.bar(all_datas.marital_1st.value_counts().index,all_datas.marital_1st.value_counts().values,label='marital_1st')
# plt.legend()
# plt.show()

def marital_1st_split(x):
    if x >0:
        if x<1950:
            return 1
        elif x<1960:
            return 2
        elif x<1970:
            return 3
        elif x<1980:
            return 4
        elif x<1990:
            return 5
        elif x<2000:
            return 6
        elif x<2010:
            return 7
        else:
            return 8
    else:
        return 0

all_datas["marital_1st_s"] = all_datas["marital_1st"].map(marital_1st_split)
# print(all_datas.marital_1st_s.value_counts(dropna=False))
all_datas.drop(["marital_1st"],axis=1,inplace=True)


# # print(all_datas['marital_now'].value_counts(dropna=False))
# plt.bar(all_datas.marital_now.value_counts().index,all_datas.marital_now.value_counts().values,label='marital_now')
# plt.legend()
# plt.show()

def marital_now_split(x):
    if x >0:
        if x<1950:
            return 1
        elif x<1960:
            return 2
        elif x<1970:
            return 3
        elif x<1980:
            return 4
        elif x<1990:
            return 5
        elif x<2000:
            return 6
        elif x<2010:
            return 7
        else:
            return 8
    else:
        return 0

all_datas["marital_now"] = all_datas["marital_now"].map(marital_now_split)
# print(all_datas.marital_1st_s.value_counts(dropna=False))
all_datas.drop(["marital_now"],axis=1,inplace=True)

## NOTE：以上所有的年份的数据处理，大家都可以根据调查时间的年份以及出生的年份来求出年龄相关的数据



## 'income'，'inc_exp'，'family_income'，s_income
# print(all_datas['income'].value_counts(dropna=False))
# 直接进行均值（众数、中位数）填充
## 缺失值的填充sklearn Imputer可以进行填充
## 使用基本的python语法
all_datas['income'] = all_datas['income'].fillna(np.mean(all_datas['income']))
# print('------------------------------------------------')
# print(all_datas['income'].value_counts(dropna=False))

# print('------------对inc_exp字段进行均值填充-----------------')
all_datas['inc_exp'] = all_datas.inc_exp.fillna(np.mean(all_datas.family_income))
# print(all_datas.inc_exp.value_counts(dropna=False))

# todo:'income'，'inc_exp' 做一个字段：收入是否达到期望收入
### 判断一下收入是否符合预期
def if_exp_inc(x):
    income, inc_exp = x[0],x[1]
    if income>=inc_exp:
        return 1
    else:
        return 0
all_datas['if_exp_inc'] = all_datas[['income','inc_exp']].apply(if_exp_inc,axis=1)
# print(all_datas['if_exp_inc'].value_counts(dropna=False))

''''''
## Note：实际上下面的都是进行均值填充，可以放在一起进行

## 对family_income , s_income字段进行处理
# print(all_datas.family_income.value_counts(dropna=False))
## 对该字段进行均值填充
# print('------------对family_income字段进行均值填充-----------------')
all_datas['family_income'] = all_datas.family_income.fillna(np.mean(all_datas.family_income))
# print(all_datas.family_income.value_counts(dropna=False))

# print(all_datas.s_income.value_counts(dropna=False))
## 对该字段进行均值填充
# print('------------对s_income字段进行均值填充-----------------')
all_datas['s_income'] = all_datas.s_income.fillna(np.mean(all_datas.s_income))
# print(all_datas.s_income.value_counts(dropna=False))

### 处理work_yr
# print(all_datas.work_yr.value_counts(dropna=False))
### 将nan值填充为均值
# print('------------对work_yr字段进行均值填充-----------------')
all_datas['work_yr'] = all_datas.work_yr.fillna(np.mean(all_datas.work_yr))
# print(all_datas.work_yr.value_counts(dropna=False))


## 'public_service_1', 'public_service_2', 'public_service_3', 'public_service_4', 'public_service_5', 'public_service_6', 'public_service_7', 'public_service_8', 'public_service_9'
# 直接进行均值（众数、中位数）填充
'''
对以下字段进行处理
['public_service_1', 'public_service_2', 'public_service_3', 'public_service_4', 'public_service_5', 'public_service_6', 'public_service_7', 'public_service_8', 'public_service_9']
'''
public_list = ['public_service_1', 'public_service_2', 'public_service_3', 'public_service_4', 'public_service_5', 'public_service_6', 'public_service_7', 'public_service_8', 'public_service_9']

### 将nan值填充为均值
# print('------------对public_字段进行均值填充-----------------')
for column_name in public_list:
    # print('------------对{}字段进行均值填充-----------------'.format(column_name))
    all_datas[column_name] = all_datas[column_name].fillna(np.mean(all_datas[column_name]))
    # print(all_datas.work_yr.value_counts(dropna=False))


### 以上是对一些连续数据的处理

### 接下来list01 离散的数据的填充 一般不会使用均值填充
# 简单处理 填充成新的一类   特征
all_datas.fillna(-1,inplace=True)

nan_num = all_datas.isnull().sum()
# print(nan_num)
# 获取缺失值为0的字段名称 是不需要进行缺失值的填充 bool索引
# TODO：但是有可能要进行其他的处理
nan_num_list = list(nan_num[nan_num>0].index)
print(nan_num_list)
# print(all_datas['id'].value_counts(dropna=False))
## 以上数据的填充就已经完成

### todo：考虑一些特征工程的操作
## onehot，连续数据标准化，归一化 （决策树相关的模型，特征工程的操作可以不做）
## 如果是一些跟距离的计算相关的一些算法 ===》特征工程 svm，线性回归，knn
## 最终看效果

## 处理好后的all_datas 进行一个存储为【训练数据，训练的标签】和测试数据
train_process_datas_x = all_datas.iloc[:train_datas_X.shape[0],:]
test_process_datas = all_datas.iloc[train_datas_X.shape[0]:,:]
print(train_process_datas_x.shape)
print(test_process_datas.shape)

### 保存为csv的数据
dir_path = './processs_datas'
import os
if os.path.exists(dir_path):
    pass
else:
    os.makedirs(dir_path)


train_process_datas_x.to_csv(dir_path+'/train_process_datas_x.csv',index=False)
test_process_datas.to_csv(dir_path+'/test_process_datas.csv',index=False)
train_datas_Y.to_csv(dir_path+'/train_datas_Y.csv',index=False)
print(train_datas_Y['happiness'].value_counts())
## 删除缺失的Y的时候，对应的x也要删除掉






import numpy as np
import pandas as pd

from sklearn.externals import joblib

##一、加载待预测数据,如果需要清洗就要清洗
# processs_datas/test_process_datas.csv
data_test = pd.read_csv('./processs_datas/test_process_datas.csv')
data_id = data_test['id']
data_test.drop(['id'],axis=1,inplace=True)

##二、加载训练好的模型
rf = joblib.load('./model/pca_rf.pkl')

##三、预测
test_y_hat = rf.predict(data_test)

submit = pd.concat([pd.DataFrame(data_id,columns=['id']),pd.DataFrame(test_y_hat,columns=['happiness'])],axis=1)
print(submit.head())
submit.to_csv('./predict/rf_submit.csv',index=False)
print(submit['happiness'].value_counts())

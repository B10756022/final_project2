#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices


import kerastuner as kt

import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


# In[ ]:


#減少記憶體空間
#Jane Street - Tensorflow Dense

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
train = reduce_mem_usage(train)
features = [c for c in train.columns if 'feature' in c]#抓出特徵

NAN_VALUE = -999#空值填入-999


# In[ ]:


train = train.astype({c: np.float32 for c in train.select_dtypes(include='float16').columns}) 
train = train.fillna(train.mean())
##dtype選擇特定條件之列:資料型別float16
##float16->float32
##fillna填0，mean()會自動忽略nan，inplace=True會取代原數據，就不用再丟回，ex:Train()，Train = Train.fillna(Train.mean())
f_mean = np.mean(train[features[1:]].values,axis=0)
#mean()會自動忽略nan，抓出有值的部分，索引1到最後
##不抓features_0
train = train.query('date > 85').reset_index(drop = True)
##篩選出date>85
##reset_index()可以讓index重置成原本的樣子
train = train[train.weight != 0]
#0無意義
n_folds = 5
#折數
seed = 2020
#種子數
skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
#shuffle隨機排序

X = train.loc[:, features].values
#if np.isnan(X[:, 1:].sum()):
#    X[:, 1:] = np.nan_to_num(X[:, 1:]) + np.isnan(X[:, 1:]) * NAN_VALUE
    
#y = (train['resp'].values > 0).astype(int)
resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']#抓出resp
y = np.stack([(train[c] > 0.000001).astype('int') for c in resp_cols]).T 

train_index, test_index = next(skf.split(X, y[:,0]))
#train_index, test_index = next(skf.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


# In[ ]:


#Jane Street - Tensorflow Dense
TUNNING = False
#以下為tf.keras.layers中之應用的解釋
#keras的layer類直接建立深度網絡中的layer
##Input用於實例化Keras
##BatchNormalization在每批中對上一層的激活進行歸一化，即應用一個轉換，將平均激活保持在0附近並將激活標準偏差保持在1附近
##GaussianNoise高斯噪聲
##Dropout輸入
###http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/layers/BatchNormalization.html
##Dense用於添加一個全連接層
def create_model(hp,input_dim,output_dim):
    inputs = tf.keras.layers.Input(input_dim)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.GaussianNoise(hp.Choice('noise',[0.0,0.03,0.05]))(x)
    x = tf.keras.layers.Dropout(hp.Choice('init_dropout',[0.0,0.3,0.5]))(x)    
    x = tf.keras.layers.Dense(hp.Int('num_units_1', 128, 2048, 64), activation=hp.Choice('activation_1', ['tanh','relu','swish']))(x)
    #units輸出變數
    x = tf.keras.layers.Dropout(hp.Choice(f'dropout_1',[0.0,0.3,0.5]))(x)
    x = tf.keras.layers.Dense(hp.Int('num_units_2', 128, 1024, 32), activation=hp.Choice('activation_2', ['tanh','relu','swish']))(x)
    x = tf.keras.layers.Dropout(hp.Choice(f'dropout_2',[0.0,0.3,0.5]))(x)
    x = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs,outputs=x)#設定輸出入之格式
    #compile編譯模型
    #解釋：https://dotblogs.com.tw/greengem/2017/12/17/094023
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('lr',[1e-2, 1e-3, 1e-5])),
                  #optimizer優化器選用
                  #實現Adam算法的優化器
                  #優化器解釋：https://keras.io/zh/optimizers/
                  #lr為學習率
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=hp.Choice('label_smoothing',[0.0, 0.01, 0.1])),
                  #tf.keras.losses.BinaryCrossentropy計算真實標籤和預測標籤之間的交叉熵損失
                  metrics=[tf.keras.metrics.AUC(name = 'auc')])
                  #metrics監視模型並判斷性能
                  #tf.keras.metrics.AUC通過黎曼和求出近似的AUC（曲線下的面積）
    return model
#Sequential將線性的層堆疊到一個tf.keras.Model
#https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
#創建一個“Sequential”模型並添加一個Dense層作為第一層
model = tf.keras.Sequential([
    tf.keras.Input(shape = len(features)),#特徵長度
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GaussianNoise(0.05),
    tf.keras.layers.Dropout(0.00), 
    tf.keras.layers.Dense(2048, activation='tanh'),#雙曲正切激活函數
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='swish'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2048, activation='swish'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='swish'),    
    tf.keras.layers.Dense(5, activation = 'sigmoid')#Sigmoid 激活函数
  ])
#activation的相關解釋：https://keras.io/zh/activations/
#tf.keras.layers.Dense(輸出入尺寸)
EPOCHS = 500#訓練過程中數據將被用多少次
BATCH_SIZE = 1024#batch_size=4096,#梯度下降

if TUNNING:
    import kerastuner as kt
    EPOCHS = 50#訓練過程中數據將被用多少次
    MAX_TRIAL = 20#調參過程中進行實驗的參數組合的總數目
    model_fn = lambda hp: create_model(hp, X_train.shape[-1], y_train.shape[-1])
    tuner = kt.tuners.BayesianOptimization(model_fn, kt.Objective('val_auc', direction='max'), MAX_TRIAL, seed = 2020)
    #貝葉斯優化
    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights=True)])
    model = tuner.get_best_models()[0]
else:
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    #optimizer = tf.keras.optimizers.RMSprop()
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.00)
    model.compile(loss = loss, optimizer=optimizer, metrics=[tf.keras.metrics.AUC()])
    #觀察網絡性能指標
    history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[callback], validation_data=(X_test, y_test))


# In[ ]:


if TUNNING:
    #打印結果摘要
    tuner.results_summary()


# In[ ]:


import janestreet
from tqdm.notebook import tqdm
#janestreet.competition.make_env.__called__ = False
env = janestreet.make_env()
#啟動環境，JaneStreet使用該環境提供測試行。然後，它創建一個迭代器，該迭代器允許獲取您需要預測的所有行。
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in tqdm(iter_test):
    #循環拋出所有應該預測的行。每個test_df是一個數據框，其中包含必須預測的一行。
    if test_df['weight'].item() > 0:
        #如果權重為0，則預測將對分數沒有影響。因此，為了節省計算時間，我們不預測權重== 0的時間
        x_tt = test_df.loc[:, features].values
        #如前所述，每個test_df都包含一個數據幀。但是您的模型無法使用數據框，它需要一個數組數組……然後將其轉換
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
            #x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * NAN_VALUE
            #如果缺少任何值，我們必須對其進行編輯。在這種情況下，我不確定目標是什麼。但是，填充缺失值的一種好
            #方法是用平均值（對異常值敏感），中位數（對異常值不敏感）或默認值（0）代替它們。
        action = np.mean(model(x_tt, training = False).numpy()[0])#模型結果
        if (action > 0.5):
            sample_prediction_df.action = 1
        else:
            sample_prediction_df.action = 0 
        #等同
        #pred = f(pred)
        #sample_prediction_df.action = np.where(pred >= th, 1, 0).astype(int)#只是沒有=
        ##預測動作是pred內容（將其視為具有類別0或1的概率）
    else:
        sample_prediction_df.action = 0 
    env.predict(sample_prediction_df)


# In[ ]:





import requests
import json
import numpy as np
import datetime
from tensorflow import keras
from keras.layers import LSTM,Dropout,Dense,Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
import time
xtrain=None
name=''.join([k for k in str(datetime.datetime.now())[:19] if k.isdigit()])
name='buy0514period24H'
##Model definition
print('Initializing Model')
if os.path.isdir('.\\'+name):
    model=keras.models.load_model(name)
    print('get previous model')
else:
    model = Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(50,1)))
    model.add(LSTM(64,return_sequences=False))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse',optimizer='rmsprop')
model.summary()

def get_recent_data(period):
    URL="https://api.bithumb.com/public/candlestick_trview/BTC_KRW/"+period
    response=requests.get(URL)
    response.text
    f=open('memo.txt',mode='wt')
    j=json.loads(response.text)
    print(str(len(j['data']['o']))+'개의 데이터 중: 50일 이상')
    k=int(input())
    return j['data']['o'][-k:]
def normalize_data(batch):
    nd=[]
    for window in batch:
        normalized_window=[((float(p)/float(window[0]))-1) for p in window]
        nd.append(normalized_window)
    return nd
def train(period,only_predict):
    global name
    global x_train
    start=time.time()
    data=get_recent_data(period)
    batch=[]
    seq_len=50
    seq_length=seq_len+1
    #make a train set
    for i in range(len(data)-seq_length):
        batch.append(data[i:i+seq_length])
    print(len(batch),'개의 train set')
    #normalize train set
    normalized_data=normalize_data(batch)
    #convert to numpy
    result=np.array(normalized_data)

    row=int(round(result.shape[0]*0.9))
    train=result[:row,:]
    np.random.shuffle(train)

    x_train=train[:,:-1]
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    y_train=train[:,-1]

    x_test=result[row:,:-1]
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    y_test=result[row:,-1]
    
    ##training
    if not only_predict:
        model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=10,epochs=10)
    print('train time:',time.time()-start)
    if not os.path.isdir('.\\'+name):
        os.mkdir(name)
    model.save(name)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    y_test=result[row:,-1]
    pred=model.predict(x_test)
    fig=plt.figure(facecolor='white')
    ax=fig.add_subplot(111)
    ax.plot(y_test,label='True')
    ax.plot(pred,label='Predict')
    ax.legend()
    plt.show()j
train('24H',True)

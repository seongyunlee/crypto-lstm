import requests
import json
import numpy as np
import datetime
from keras.layers import LSTM,Dropout,Dense,Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
URL="https://api.bithumb.com/public/candlestick_trview/BTC_KRW/10M"
response=requests.get(URL)
response.text
f=open('memo.txt',mode='wt')
j=json.loads(response.text)
data=j['data']['o'][:500]
batch=[]
seq_len=50
seq_length=seq_len+1
for i in range(len(data)-seq_length):
    batch.append(data[i:i+seq_length])
normalized_data=[]
for window in batch:
    normalized_window=[((float(p)/float(window[0]))-1) for p in window]
    normalized_data.append(normalized_window)
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

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(50,1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='rmsprop')
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=10,epochs=10)

pred=model.predict(x_test)
fig=plt.figure(facecolor='white')
ax=fig.add_subplot(111)
ax.plot(y_test,label='True')
ax.legend()
plt.show()

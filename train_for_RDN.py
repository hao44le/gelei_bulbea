
# coding: utf-8

# A canonical way of importing the `bulbea` module is as follows:

# In[1]:

import bulbea as bb


# In[2]:

from coinmarketcap_draw import coinmarketcap_data


# Go ahead and create a `Share` object as follows:

# In[3]:

coin_name = 'raiden-network-token'
data = coinmarketcap_data(coin_name)


# In[4]:

nsamples = 10
data.tail(nsamples)


# In order to analyse a given attribute, you could plot the same as follows:

# ##  Convert the data to hourly

# In[5]:

from datetime import timedelta
import pandas as pd

def convert_with_n_hour_gap(data,n):
    times = data.index.copy()
    first_time = times[0].to_datetime()
    v_dict = dict()

    for x in range(1,len(times)):
        t = times[x].to_datetime()
        if n == 24:
            success = t.day == first_time.day + 1
        else:
            success = (first_time + timedelta(hours=n)).hour == t.hour
        if success:
            first_time = t
            index = pd.Timestamp(t)
            v_dict[index] = data.loc[index]['Close']
    df = pd.DataFrame(list(v_dict.items()), columns=['Date', 'Close'])
    df.set_index("Date",inplace=True)
    return df


# In[6]:

share_array = []
for i in range(1,25):
    df = convert_with_n_hour_gap(data,i)
    share = bb.Share("123",'123',data=df)
    share_array.append(share)


# ### Modelling

# In[9]:

from bulbea.learn.models import RNN


# ### Training & Testing

# In[11]:

from bulbea.learn.evaluation import split
import numpy as np


# In[ ]:

rnn_arr = []
for index,share in enumerate(share_array):
    print("{} hour. {}".format(index+1,len(share.data)))
    Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)
    Xtrain  = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest  = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

    # Training
    rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
    rnn.fit(Xtrain, ytrain)
    rnn_arr.append(rnn)



# #### TESTING

# In[13]:

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pplt
from bulbea.entity.share import _reverse_cummulative_return
from datetime import datetime

for index,share in enumerate(share_array):
    print("{} hour. {}".format(index+1,len(share.data)))
    Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = True)
    Xtrain  = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest  = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

    predicted = rnn.predict(Xtest)
    sqr_err = mean_squared_error(ytest, predicted)
    print(sqr_err)


    _,_,_,ori_ytest = split(share, 'Close', normalize = False)

    new_pre = []
    for x in range(0,len(ori_ytest)):
        t = ori_ytest[x]
        predict = predicted[x]
        new_pre.append(_reverse_cummulative_return(t,predict))

    pplt.plot(ori_ytest)
    pplt.plot(new_pre)
    pplt.show()

    rnn.model.save("models/{}_{}.h5".format(coin_name,index+1))


# In[44]:




# In[ ]:

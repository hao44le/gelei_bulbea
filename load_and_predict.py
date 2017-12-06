
# coding: utf-8

# In[1]:

from keras.models import load_model


# In[18]:

import os
folder_name = "models"

def load_model_helper(name):
    models = dict()
    files = sorted(os.listdir(folder_name))
    for f in files:
        if name in f:
            index = f.split("_")[1][:-3]
            models[index] = load_model("{}/{}".format(folder_name,f))
            print("{} {}".format(index,f))
    return models


# In[19]:

coin_name = 'raiden-network-token'
models = load_model_helper(coin_name)


# In[4]:

from coinmarketcap_draw import coinmarketcap_data


# In[5]:

data = coinmarketcap_data(coin_name)


# In[6]:

import bulbea as bb
figsize = (20, 15)
get_ipython().magic('matplotlib inline')
share = bb.Share("123",'123',data=data)


# In[7]:

share.plot(figsize = figsize)


# ##  Convert the data to hourly

# In[8]:

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


# In[9]:

share_array = []
for i in range(1,25):
    df = convert_with_n_hour_gap(data,i)
    share = bb.Share("123",'123',data=df)
    share_array.append(share)

# In[12]:

from bulbea.entity.share import _reverse_cummulative_return

def rever_back(ori_ytest,predicted):
    new_pre = []
    for x in range(0,len(ori_ytest)):
        t = ori_ytest[x]
        predict = predicted[x]
        new_pre.append(_reverse_cummulative_return(t,predict))
    return new_pre


# In[13]:

from datetime import timedelta
import pandas as pd
from bulbea.learn.evaluation import split
import numpy as np

def predict_next_from_current_share(var_share,model):
    _, Xtest, _, ytest = split(var_share, 'Close', normalize = True, train = 0.0)
    _,ori_Xtest,_,ori_ytest = split(var_share, 'Close', normalize = False, train = 0.0)
    Xtest  = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

    # Format the Xtest
    last_Xtest = Xtest[-1:]
    last_Xtest  = np.reshape(last_Xtest, (last_Xtest.shape[0], last_Xtest.shape[1], 1))

    # Format the ori_ytest
    last_ori_ytest = ori_ytest[-1]

    # Get the prediction
    predict = model.predict(last_Xtest)

    # convert it back
    new_pre = rever_back([last_ori_ytest],[predict])[0][0][0]
    return new_pre

def predict_next_n_hours(n):
    for i in range(n):
        loop_share = share_array[i]
        model = models[str(i+1)]
        new_pre = predict_next_from_current_share(loop_share,model)
        print("\t\tnext {} hour price ${}".format(i+1,new_pre))


# In[14]:

predict_next_n_hours(24)


# In[16]:

print(data.tail(3))


# In[17]:




# In[ ]:

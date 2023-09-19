#!/usr/bin/env python
# coding: utf-8

# In[4]:


import yfinance as yf


# In[5]:


sp500 = yf.Ticker("^GSPC")


# In[6]:


sp500 = sp500.history(period="max")


# In[7]:


sp500


# In[8]:


sp500.index


# In[9]:


sp500.plot.line(y="Close", use_index=True)


# In[10]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[11]:


sp500["Tomorrow"] = sp500["Close"].shift(-1) 


# In[12]:


sp500


# In[13]:


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


# In[14]:


sp500


# In[15]:


sp500 = sp500.loc["1990-01-01":].copy()


# In[16]:


sp500


# In[17]:


from sklearn.ensemble import RandomForestClassifier
#random forest works by training a bunch of individual decisions with randomized parameters and then averaging the results
#they also pick up nonlinear tendencies which works great in terms of stocks
model = RandomForestClassifier(n_estimators=100, min_samples_split=100,random_state=1)
#n_estimators is the number of individual decision trees to train the higher the more accurate to a limit
#min_samples_split helps protect against overfitting the higher the less accurate 
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close","Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


# In[18]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[19]:


import pandas as pd
preds = pd.Series(preds, index=test.index)


# In[21]:


precision_score(test["Target"],preds)


# In[25]:


combined = pd.concat([test["Target"], preds], axis=1)


# In[27]:


combined.plot()


# In[29]:


def predict(train, test, predictors, model):
    model.fit(train[predictors],train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[30]:


def backtest(data,model, predictors, start=2500,step=250):
    all_predictions=[]

    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test= data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[31]:


predictions = backtest(sp500, model, predictors)


# In[32]:


predictions["Predictions"].value_counts()


# In[33]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[35]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[37]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]


# In[39]:


sp500 = sp500.dropna()


# In[40]:


sp500


# In[51]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[52]:


def predict(train, test, predictors, model):
    model.fit(train[predictors],train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6] =1
    preds[preds< .6]=0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[53]:


predictions = backtest(sp500, model, new_predictors)


# In[54]:


predictions["Predictions"].value_counts()


# In[55]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:





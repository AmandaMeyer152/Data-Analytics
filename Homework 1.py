# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:53:18 2022

@author: mail2
"""
import pandas as pd
import numpy as np 

import statsmodels.formula.api as sm # module for stats models
from statsmodels.iolib.summary2 import summary_col # module for presenting stats models outputs nicely


#%%
from pathlib import Path
import sys
import os

home = str(Path.home())
print(home)


# %%
if sys.platform == 'linux':
    inputDir = '/datasets/stocks/' 
elif sys.platform == 'win32':
    inputDir = '\\OneDrive\\School\\Data Analysis\\Data\\' 
else :
    inputDir = '/datasets/stocks/'
    
fullDir= home+inputDir
print(fullDir)


# %%
def price2ret(prices,retType='simple'):
    if retType == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret


# %%
stkName = 'NKE'
fileName = 'stk_' + stkName + '.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Adj_Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_NKE= df_stk


# %%
stkName = 'SHOO'
fileName = 'stk_' + stkName + '.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Adj_Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_SHOO= df_stk


# %%
stkName = 'CROX'
fileName = 'stk_' + stkName + '.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Adj_Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_CROX= df_stk

#%%
stkName = 'NKE'
fileName = 'stk_' + stkName + 'y.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_NKE_Y= df_stk


# %%
stkName = 'SHOO'
fileName = 'stk_' + stkName + 'y.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_SHOO_Y= df_stk


# %%
stkName = 'CROX'
fileName = 'stk_' + stkName + 'y.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_CROX_Y= df_stk

# %% Merging DF

df_CROX_merge= pd.merge(df_CROX,df_CROX_Y,left_index=True, right_index=True)
df_NKE_merge= pd.merge(df_NKE,df_NKE_Y,left_index=True, right_index=True)
df_SHOO_merge= pd.merge(df_SHOO,df_SHOO_Y,left_index=True, right_index=True)

#%% Plotting the Data

import matplotlib.pyplot as plt

plt.figure()
plt.plot(df_CROX_merge['CROX_Returns_x'], color='green',)
plt.plot(df_CROX_merge['CROX_Returns_y'], color='red',)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('CROX Comparison!')

plt.figure()
plt.plot(df_NKE_merge['NKE_Returns_x'], color='green',)
plt.plot(df_NKE_merge['NKE_Returns_y'], color='red',)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('NKE Comparison!')

plt.figure()
plt.plot(df_SHOO_merge['SHOO_Returns_x'], color='green',)
plt.plot(df_SHOO_merge['SHOO_Returns_y'], color='red',)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('SHOO Comparison!')

#%% Identifying dates where there is a difference in data sources

df_CROX_merge['Adj_Close'].equals(df_CROX_merge['Close'])
df_CROX_merge['Before']=np.where(df_CROX_merge['Adj_Close']==df_CROX_merge['Close'],"same","different")

for index, row in df_CROX_merge.iterrows():
    if row['Adj_Close'] == row ['Close']:
        pass
    else:
        print(index)
        
df_SHOO_merge['Adj_Close'].equals(df_SHOO_merge['Close'])
df_SHOO_merge['Diff']=np.where(df_SHOO_merge['Adj_Close']==df_SHOO_merge['Close'],"same","different")

for index, row in df_SHOO_merge.iterrows():
    if row['Adj_Close'] == row ['Close']:
        pass
    else:
        print(index)
    
df_NKE_merge['Adj_Close'].equals(df_NKE_merge['Close'])
df_NKE_merge['Diff']=np.where(df_NKE_merge['Adj_Close']==df_NKE_merge['Close'],"same","different")

for index, row in df_NKE_merge.iterrows():
    if row['Adj_Close'] == row ['Close']:
        pass
    else:
        print(index)
        
## this code prints all the dates were Bloomberg closing data and YahooFinance 
## closing data do not match.  There are a lot of difference because if you print
## each of the columns, it shows there are more decimals associated with the 
## YahooFinance data.  This can be fixed with a rounding function.

    
#%%  Rounding YahooFinance Data

df_CROX_merge['Close']=df_CROX_merge['Close'].round(decimals=2)
df_CROX_merge['Adj_Close'].equals(df_CROX_merge['Close'])
df_CROX_merge['Post Rounding']=np.where(df_CROX_merge['Adj_Close']==df_CROX_merge['Close'],"same","different")

df_SHOO_merge['Close']=df_SHOO_merge['Close'].round(decimals=2)
df_SHOO_merge['Adj_Close']=df_SHOO_merge['Adj_Close'].round(decimals=2)
df_SHOO_merge['Adj_Close'].equals(df_SHOO_merge['Close'])
df_SHOO_merge['Post Rounding']=np.where(df_SHOO_merge['Adj_Close']==df_SHOO_merge['Close'],"same","different")


df_NKE_merge['Close']=df_NKE_merge['Close'].round(decimals=2)
df_NKE_merge['Adj_Close'].equals(df_NKE_merge['Close'])
df_NKE_merge['Post Rounding']=np.where(df_NKE_merge['Adj_Close']==df_NKE_merge['Close'],"same","different")


#%%
index_where_true = (df_CROX_merge['Adj_Close'] - df_CROX_merge['Close']) < 0.0000001
df_CROX_merge[index_where_true].index
## Prints all dates that are the same










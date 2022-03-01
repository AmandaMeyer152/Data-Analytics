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

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Adj Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_NKE= df_stk


# %%
stkName = 'SHOO'
fileName = 'stk_' + stkName + '.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Adj Close']])
df_stk = df_stk.dropna()
df_stk.head()

df_SHOO= df_stk


# %%
stkName = 'CROX'
fileName = 'stk_' + stkName + '.csv'
readFile = fullDir + fileName

df_stk = pd.read_csv(readFile,index_col='Date',parse_dates=True)

df_stk[stkName+'_Returns'] = price2ret(df_stk[['Adj Close']])
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

# %%
df_merge= pd.merge(df_CROX,df_NKE,left_index=True, right_index=True )
df_Bloomberg= pd.merge(df_merge,df_SHOO,left_index=True, right_index=True)

df_merge= pd.merge(df_CROX_Y,df_NKE_Y,left_index=True, right_index=True )
df_Yahoo= pd.merge(df_merge,df_SHOO_Y,left_index=True, right_index=True)


#%%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df_Bloomberg['SHOO_Returns'], color='green',)
plt.plot(df_Bloomberg['CROX_Returns'], color='red',)
plt.plot(df_Bloomberg['NKE_Returns'], color='blue',)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Shoe Data from BLOOMBERG!')

plt.figure()
plt.plot(df_Yahoo['SHOO_Returns'], color='green',)
plt.plot(df_Yahoo['CROX_Returns'], color='red',)
plt.plot(df_Yahoo['NKE_Returns'], color='blue',)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Shoe Data from YAHOO!')

##  COMPARE THE DATA
## these two figures show returns data using closing data from both Bloomberg
## and YahooFinance.  There is no difference between the two.  This is becuase
## YahooFinance and Bloomberg must use the same measures when they submit their
## final closing data.









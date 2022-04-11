# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:24:56 2022

@author: Amanda Meyer
"""
# %% Set up working folder and test its existence
import os
import pandas as pd
import numpy as np 

from sec_api import QueryApi
from sec_api import ExtractorApi

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

#%%
folder='C:\\Users\\mail2\\OneDrive\\School\\Data Analysis\\10K'
exist=os.path.exists(folder)

#%% Pulls the 10ks for the stock and dates selected

queryApi = QueryApi(api_key="052592b42cdb89dbcafb640cc1eeb92e625d8b762d72d729b07cc0d85186ee44")
query = {
  "query": { "query_string": { 
      "query": "ticker:CROX AND filedAt:{2017-01-01 TO 2022-04-01} AND formType:\"10-K\"" 
    } },
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
}

filings = queryApi.get_filings(query)

df =pd.json_normalize(filings['filings'])
df.drop(df[df['formType'] != '10-K'].index, inplace= True)
df.reset_index

#%% Gathers all the item 7s into one data frame
extractorApi = ExtractorApi("052592b42cdb89dbcafb640cc1eeb92e625d8b762d72d729b07cc0d85186ee44")
Final_10k=[]
for index in df.index:
     filing_url = df['linkToFilingDetails'][index]
     section_10k_item7 = extractorApi.get_section(filing_url, "7", "text")
     Final_10k.append(section_10k_item7)
     
#with open('C:\\Users\\mail2\\OneDrive\\School\\Data Analysis\\10K\\CROX_10k_item7.txt', 'w') as f:
#    for item in Final_10k:
#        f.write(item)
        
# %% FUNCTION Text to Sentences using NLTK
def txt2sentence(txt):
    sentences = sent_tokenize(txt)
    df=pd.DataFrame(sentences)
    return df

# %% SENTIMENT ANALYSIS *need to fix year in df output*
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

df_CROX=pd.DataFrame()
for index in range(len(Final_10k)):
    df_sentence = txt2sentence(Final_10k[index])
    
    results=[]
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('COULD NOT READ A LINE')
            
    neutral=0
    negative=0
    positive=0
    
    for i in results:
        if i[0]['label'] == 'neutral':
            neutral=neutral+1
        if i[0]['label'] == 'negative':
            negative=negative+1
        if i[0]['label'] == 'positive':
            positive=positive+1
    
    CROX = [{'Ticker':"CROX",'Year':'2021','Neutral':neutral,'Positive':positive,'Negative':negative}]
    df_CROX=df_CROX.append(pd.DataFrame(CROX))
            
#%% *keep years straight before fixed
# 2021: neutral 376
# Final_10k[0]=2021
# Final_10k[5]=2016
# to use a sentiment score: results[0][0]['score']
#%% Stock Return Analysis for Comparison

def price2ret(prices,retType='simple'):
    if retType == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret

import pandas_datareader.data as web
CROX_Price= web.DataReader('CROX', 'yahoo', start='2017-01-01', end='2022-04-01')
CROX_Price['Returns']= price2ret(CROX_Price[['Adj Close']])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(CROX_Price['Adj Close'], color='Purple',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('CROX Price!')

plt.figure()
plt.plot(CROX_Price['Returns'], color='Pink',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.title('CROX Returns!')

    
    
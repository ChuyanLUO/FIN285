#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:08:30 2018

@author: jingxiaowang
"""
import pandas as pd  
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() 
import datetime 
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy import stats
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy import optimize
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts

# weights
TickerNWeights = pd.read_excel('weight.xlsx', sheet_name='Sheet1', header=0)
wts_index = TickerNWeights['Weight']
# download price
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
start_dt = datetime.datetime(2008, 10, 31)
end_dt = datetime.datetime(2018, 10, 31)
# tickers = ['ACAD','ALKS','ALNY','ALXN','AMGN','ARRY','BIIB','BMRN','CELG','EXEL','FOLD','GHDX','GILD','HALO','ILMN','IMMU','INCY','IONS','JAZZ','LGND','MDCO','MYGN','MYL','NBIX','NKTR','OPK','REGN','RGEN','SGEN','SHPG','SRPT','TECH','UTHR','VRTX']
tickers = TickerNWeights['Ticker']
stock_data = getDataBatch(tickers, start_dt, end_dt)
daily_close_px = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
# Calculate returns
daily_return = daily_close_px.pct_change().dropna()
num_periods, num_stock = daily_return.shape
# write excel
daily_close_px.to_csv('TEdata.csv', header=True, index=True)

Index_prs = np.ones(2518)

for i in range(0,2518):
    Index_prs[i] = np.sum(daily_close_px.iloc[i,:]*TickerNWeights.loc[:,"Weight"])
Index_prs = pd.DataFrame(Index_prs, index=daily_close_px.index)

daily_ret_index = daily_close_px.pct_change().dropna() 

Index_ret = np.ones(2517)
for i in range(0,2517):
    Index_ret[i] = np.sum(daily_ret_index.iloc[i,:]*TickerNWeights.loc[:,"Weight"])
    
Index_ret = pd.DataFrame(Index_ret, index=daily_ret_index.index)
Index_risk = pd.DataFrame(Index_ret**2, index=daily_ret_index.index) 


figure_count = 1
ax1=plt.subplot(111)
plt.plot(Index_prs)
plt.xlabel('Time')
plt.ylabel('Index Daily Close Px')
plt.title('Index Daily Close Px')
plt.show()

ax2=plt.subplot(111)
plt.plot(Index_ret)
plt.xlabel('Time')
plt.ylabel('Index Daily Return')
plt.title('Index Daily Return')
plt.show()

ax3=plt.subplot(111)
plt.plot(Index_risk)
plt.xlabel('Time')
plt.ylabel('Index Daily Variance')
plt.title('Index Daily Variance')
plt.show()



#C. Forecast covariance matrix
#1. #MA


##2. EWMA


#D. Build ETF
#1. Pick Assets

#2. Build covariance matrix

#3. Partitioning

#4. Cross validation

#5. Adjust ETF and repeat 

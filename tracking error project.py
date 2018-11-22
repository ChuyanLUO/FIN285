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

# download price
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
start_dt = datetime.datetime(2008, 10, 31)
end_dt = datetime.datetime(2018, 10, 31)
tickers = ['ACAD','ALKS','ALNY','ALXN','AMGN','ARRY','BIIB','BMRN','CELG','EXEL','FOLD','GHDX','GILD','HALO','ILMN','IMMU','INCY','IONS','JAZZ','LGND','MDCO','MYGN','MYL','NBIX','NKTR','OPK','REGN','RGEN','SGEN','SHPG','SRPT','TECH','UTHR','VRTX']
stock_data = getDataBatch(tickers, start_dt, end_dt)
daily_close_px = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
# write excel
daily_close_px.to_csv('TEdata.csv', header=True, index=True)
# weights
TickerNWeights = pd.read_excel('weight.xlsx', sheet_name='Sheet1', header=0, index_col=0)
wts_index = TickerNWeights['Weight']
# sort tickers by weights
daily_close_px_transpose = daily_close_px.T
daily_close_px_transpose.loc[:,'Weight'] = TickerNWeights.loc[:,'Weight']
daily_close_px_sort = daily_close_px_transpose.sort_values('Weight',ascending=False,axis=0)
#daily_close_px_sort['Weight'].head()
del daily_close_px_sort['Weight']
daily_close_px2 = daily_close_px_sort.T
# Calculate returns
daily_return = daily_close_px2.pct_change().dropna()
num_periods, num_stock = daily_return.shape
#-------------------------------Luo------------------------------------------------
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
#-----------------------------------------------------------------------------------------------------


#C. Forecast covariance matrix
# define optimizer function step by step
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def te_opt(W_Bench, C, obj_te, c_, b_):
    # function that minimize the objective function
    n = len(W_Bench)
    # change the initial guess to help test whether we find the global optimal
    guess = 2
    #W = rand_weights(n) # start with random weights
    if guess==1:
        W = rand_weights(n) # start with random weights
    elif guess==2:
        W = W_Bench # Use Bench weight as initial guess
    else:
        W = 1/n*np.ones([n,1])
    
    optimized = optimize.minimize(obj_te, W, (W_Bench, C), 
                method='SLSQP', constraints=c_, bounds=b_,  
                options={'ftol':1e-10, 'maxiter': 1000000, 'disp': False})        
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights

def obj_te(W, W_Bench, C): 
    wts_active = W - W_Bench
    return(np.sqrt(np.transpose(wts_active)@C@wts_active))

def opt_min_te(W, C, b_, c_):
    return(te_opt(W, C, obj_te, c_, b_))
    
# define EWMA covariance
def ewma_cov(rets, lam): 
    T, n = rets.shape
    ret_mat = rets.as_matrix()
    EWMA = np.zeros((T,n,n))
    S = np.cov(ret_mat.T)  
    EWMA[0,:] = S
    for i in range(1, T) :
        S = lam * S  + (1-lam) * np.matmul(ret_mat[i-1,:].reshape((-1,1)), 
                      ret_mat[i-1,:].reshape((1,-1)))
        EWMA[i,:] = S

    return(EWMA)

# Calulate Covariance Matrix
   
def tracking_error(wts_active,cov):
    TE = np.sqrt(np.transpose(wts_active)@cov@wts_active)
    return TE

lam = 0.94
# vol of the assets 
vols = daily_return.std()
rets_mean = daily_return.mean()
# demean the returns
daily_return = daily_return - rets_mean

# var_ewma calculation by calling ewma_cov function
var_ewma = ewma_cov(daily_return, lam)
var_ewma_annual = var_ewma*252 #Annualize
# take only the covariance matrix for the last date, which is the forecast for next time period
cov_end = var_ewma[-1,:]
# Annualize
cov_end_annual = cov_end*252 
std_end_annual = np.sqrt(np.diag(cov_end))*np.sqrt(252)
# calculate the correlation matrix
corr = daily_return.corr()


#1. #MA


##2. EWMA


#D. Build ETF
#1. Pick Assets

#2. Build covariance matrix

#3. Partitioning

#4. Cross validation

#5. Adjust ETF and repeat 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:08:30 2018

@author: jingxiaowang & Chuyan Luo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:27:48 2018

@author: luochuyan
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
from scipy import optimize
import matplotlib.pyplot as plt

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
TickerNWeights = TickerNWeights.sort_values('Weight',ascending=False)
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

# Descriptive plots
# index price
Index_prs = pd.DataFrame(daily_close_px2 @ TickerNWeights)
# index return
Index_ret = pd.DataFrame(daily_return @ TickerNWeights)
# index risk
Index_risk = pd.DataFrame(Index_ret ** 2)

#Figure of daily close prs
figure_count = 1
ax1=plt.subplot(111)
plt.plot(Index_prs)
plt.xlabel('Time')
plt.ylabel('Index Daily Close Px')
plt.title('Index Daily Close Px')
plt.show()
#Figure of daily returns
ax2=plt.subplot(111)
plt.plot(Index_ret)
plt.xlabel('Time')
plt.ylabel('Index Daily Return')
plt.title('Index Daily Return')
plt.show()
#Figure of daily variance
ax3=plt.subplot(111)
plt.plot(Index_risk)
plt.xlabel('Time')
plt.ylabel('Index Daily Variance')
plt.title('Index Daily Variance')
plt.show()

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

##Data partitioning
daily_ret_train = np.split(daily_return, [2015-12-31],axis=0)[0]
daily_ret_valid = np.split(daily_return, [2015-12-31],axis=0)[1]

num_periods, num_stock = daily_ret_train.shape

## EWMA Approach
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
vols = daily_ret_train.std()
rets_mean = daily_ret_train.mean()
# demean the returns
daily_ret_train = daily_ret_train - rets_mean

# var_ewma calculation by calling ewma_cov function
var_ewma = ewma_cov(daily_ret_train, lam)
var_ewma_annual = var_ewma*252 #Annualize
# take only the covariance matrix for the last date, which is the forecast for next time period
cov_end = var_ewma[-1,:]
# Annualize
cov_end_annual = cov_end*252 
std_end_annual = np.sqrt(np.diag(cov_end))*np.sqrt(252)
# calculate the correlation matrix
corr = daily_ret_train.corr()

# looping through number of stocks and save the history of TEs
num_stock_low = 10
num_stock_high = 25
numstock_2use = range(num_stock_low,num_stock_high)
wts_active_hist = np.zeros([len(numstock_2use), num_stock])
TE_hist = np.zeros([len(numstock_2use), 1])
count = 0

for i in numstock_2use:
    # only the top weight stocks + allow shorting 
    b1_c_a_ = [(-1.0,1.0) for j in range(i)] 
    # exclude bottom weighted stocks
    b1_c_b_ = [(0.0,0.0) for j in range(i,num_stock)] 
    b1_curr_ = b1_c_a_ + b1_c_b_
    # Sum of active weights = 100%
    c1_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })
    wts_min_curr = opt_min_te(wts_index, cov_end_annual, b1_curr_, c1_)
    wts_active_hist[count,:] = wts_min_curr.transpose()
    TE_optimized_c = tracking_error(wts_min_curr-wts_index,cov_end)
    TE_hist[count,:] = TE_optimized_c*10000# in bps
    count = count+1
    
    del b1_curr_, b1_c_a_, b1_c_b_,TE_optimized_c,wts_min_curr

#------plot TE as a function of number of stocks -------------
plt.figure(figure_count)
figure_count = figure_count+1
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(range(num_stock_low,num_stock_high), TE_hist, 'b')
plt.xlabel('Number of stocks in ETF', fontsize=18)
plt.ylabel('Optimized Tracking Error (bps)', fontsize=18)
plt.title('Biotech ETF', fontsize=18)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

# choose the number of index and show the min TE
num_topwtstock_include = 19
# only the top weight stocks + allow shorting 
b1a_ = [(-1.0,1.0) for i in range(num_topwtstock_include)] 
# exclude bottom weighted stocks
b1b_ = [(0.0,0.0) for i in range(num_topwtstock_include,num_stock)] 
b1_ = b1a_ + b1b_ # combining the constraints
c1_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })  # Sum of active weights = 100%
# Calling the optimizer
wts_min_trackingerror = opt_min_te(wts_index, cov_end_annual, b1_, c1_)
# calc TE achieved
wts_active = wts_min_trackingerror - wts_index
TE_optimized = tracking_error(wts_active,cov_end)
print('{0} top weighted stock replication TE in training set= {1:5.2f} bps'.format(num_topwtstock_include, TE_optimized*10000))

#  Plot bars of weights
figure_count = figure_count+1
# ---  create plot of weights fund vs benchmark
plt.figure(figure_count)
figure_count = figure_count+1
fig, ax = plt.subplots(figsize=(18,10))
index = np.arange(len(wts_index))
bar_width = 0.3
opacity = 0.6
 
rects1 = plt.bar(index, wts_index, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Index Weight')
 
rects2 = plt.bar(index + bar_width, wts_min_trackingerror, bar_width,
                 alpha=opacity,
                 color='g',
                 label='ETF fund Weight')
 
plt.xlabel('Ticker', fontsize=18)
plt.ylabel('Weights', fontsize=18)
plt.xticks(index + bar_width, (daily_close_px2.columns.values), fontsize=12)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(fontsize=20)
 
plt.tight_layout()
plt.show()

# vol of the assets 
vols_v = daily_ret_valid.std()
rets_mean_v = daily_ret_valid.mean()
# demean the returns
daily_ret_valid = daily_ret_valid - rets_mean_v

# var_ewma calculation by calling ewma_cov function
var_ewma_v = ewma_cov(daily_ret_valid, lam)
var_ewma_annual_v = var_ewma_v*252 #Annualize
# take only the covariance matrix for the last date, which is the forecast for next time period
cov_end_v = var_ewma_v[-1,:]
# Annualize
cov_end_annual_v = cov_end_v*252 
std_end_annual_v = np.sqrt(np.diag(cov_end_v))*np.sqrt(252)
# calculate the correlation matrix
corr_v = daily_ret_valid.corr()

TE_optimized_v = tracking_error(wts_active,cov_end_v)
print('{0} top weighted stock replication TE in validation set= {1:5.2f} bps'.format(num_topwtstock_include, TE_optimized_v*10000))



#1. #MA


##2. EWMA


#D. Build ETF
#1. Pick Assets

#2. Build covariance matrix

#3. Partitioning

#4. Cross validation

#5. Adjust ETF and repeat 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:40:45 2018

@author: jingxiaowang
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:20:14 2018

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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import timedelta
from collections import OrderedDict
#%%
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
start_dt = datetime.datetime(2007, 6, 4)
end_dt = datetime.datetime(2018, 10, 31)
tickers = ['BND', 'BIV', 'WIP', 'IEF', "AGG",'LQD','GLD','DBC','GSC']
bond_data = getDataBatch(tickers, start_dt, end_dt)
daily_close_bond = bond_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
daily_stock = pd.read_csv('ETF_daily_price.csv',index_col=0)
daily_stock.columns = ['stock ETF']
daily_port = daily_stock.join(daily_close_bond)
daily_portret = daily_port.pct_change().dropna()

# correlation matrix heatmap
labels = daily_portret.columns
# calculate the correlation matrix
corr = daily_portret.corr()

plt.figure(1)
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="YlGnBu"        )
plt.title('Correlation of ETFs')
print(corr)
#%%
# get our portfolio
daily_portret2 = daily_portret.loc[:,('stock ETF','BND','GLD')]

# functions that will be used
# calculate covariance matrix by using EWMA
def cov_ewma(ret_assets, lamda = 0.94):
    ret_mat = ret_assets.values
    T = len(ret_assets)
    coeff = np.zeros((T,1))
    S = ret_assets.cov()
    for i in range(1, T):
#        S = lamda * S  + (1-lamda)*np.matmul(ret_mat[i-1,:].reshape((-1,1)), 
#                          ret_mat[i-1,:].reshape((1,-1)))
        S = lamda * S  + (1-lamda)* (ret_mat[i-1,:].reshape((-1,1)) @ ret_mat[i-1,:].reshape((1,-1)) )
        
        coeff[i] = (1-lamda)*lamda**(i)
    return S/np.sum(coeff)

# risk budgeting approach optimisation object function
def obj_fun(W, cov_assets, risk_budget):
    var_p = np.dot(W.transpose(), np.dot(cov_assets, W))
    sigma_p = np.sqrt(var_p)
    risk_contribution = W*np.dot(cov_assets, W)/sigma_p
    risk_contribution_percent = risk_contribution/sigma_p
    return np.sum((risk_contribution_percent-risk_budget)**2)

# calculate risk budgeting portfolio weight give risk budget
def riskparity_opt(ret_assets, risk_budget, lamda, method='ewma',Wts_min=0.0, leverage=False):
    # number of assets
    num_assets = ret_assets.shape[1]
    # covariance matrix of asset returns
    if method=='ewma':
        cov_assets = cov_ewma(ret_assets, lamda)
    elif method=='ma':
        cov_assets = ret_assets.cov()
    else:
        cov_assets = cov_ewma(ret_assets, lamda)        
    
    # initial weights
    w0 = 1.0 * np.ones((num_assets, 1)) / num_assets
    # constraints
    #cons = ({'type': 'eq', 'fun': cons_sum_weight}, {'type': 'ineq', 'fun': cons_long_only_weight})
    if leverage == True:
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-2. }, # Sum of weights = 200%
              {'type':'ineq', 'fun': lambda W: W-Wts_min}) # weights greater than min wts
    else:
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. }, # Sum of weights = 100%
              {'type':'ineq', 'fun': lambda W: W-Wts_min}) # weights greater than min wts
    # portfolio optimisation
    return minimize(obj_fun, w0, args=(cov_assets, risk_budget), method='SLSQP', constraints=c_)

# 2. Calculate ARP excess returns
daily_portret2_demean = daily_portret2 - daily_portret2.mean()
num_assets = daily_portret2.shape[1]

lamda = 0.94
cov_port = cov_ewma(daily_portret2_demean, lamda)
#%%
# Construct risk parity portfolio
# portfolio dates - this defines the first date of portfolio construction
datestr = daily_portret2.index[daily_portret2.index >= '2008-03-31']
# previous month
mth_previous = datestr[0]
# initialise portfolio weights matrix
wts = pd.DataFrame(index=datestr, columns=daily_portret2.columns)
# initialise portfolio return matrix
ret_riskParity = pd.DataFrame(index=datestr, columns=['Risk Parity'])
# how many rolling calendar days to use for covariance calculation
window = 30
Wts_min = 0.0
risk_budget = 1.0/num_assets*np.ones([1,num_assets]) #risk-party
#risk_budget = [0.7, 0.4]
leverage = False
varmodel = 'ewma'


for t in datestr:
    # construct risk budgeting portfolio and re-balance on monthly basis
    if t.month==mth_previous:
        # keep the same portfolio weights within the month
        wts.loc[t] = wts.iloc[wts.index.get_loc(t)-1]
    else:
        # update the value of the previous month 
        mth_previous = t.month
        # re-balance the portfolio at the start of the month
        
        t_begin = t - timedelta(days=window)
        ret_used = daily_portret2.loc[t_begin:t,:]
        wts.loc[t] = riskparity_opt(ret_used, risk_budget, lamda, varmodel, Wts_min, leverage).x
    # calculate risk budgeting portfolio returns
    ret_riskParity.loc[t] = np.sum(wts.loc[t] * daily_portret2.loc[t])
    
# Due to precision issue, wts could be a tiny negative number instead of zero, make them zero
wts[wts<0]=0.0
# Construct equal weighted portfolio
ret_equalwted = pd.DataFrame(np.sum(1.0*daily_portret2[daily_portret2.index>=datestr[0]]/num_assets, axis=1), columns=['Equal Weighted'])
# Construct 60/40 weighted portfolio
#ret_equalwted = pd.DataFrame(np.sum(1.0*ret_assets[ret_assets.index>=datestr[0]]/num_assets, axis=1), columns=['Equal Weighted'])

# Calculate performance stats
ret_cumu_assets = (daily_portret2 + 1).cumprod()
ret_cumu_riskP = (ret_riskParity + 1).cumprod()
ret_cumu_equalwt = (ret_equalwted + 1).cumprod()

ret_annual_assets = ret_cumu_assets.iloc[-1]**(250/len(ret_cumu_assets))-1
std_annual_assets = daily_portret2.std()*np.sqrt(250)
sharpe_ratio_assets = ret_annual_assets/std_annual_assets

ret_annual_riskP = ret_cumu_riskP.iloc[-1]**(250/len(ret_cumu_riskP))-1
std_annual_riskP = ret_riskParity.std()*np.sqrt(250)
sharpe_ratio_riskP = ret_annual_riskP/std_annual_riskP

ret_annual_equalwt = ret_cumu_equalwt.iloc[-1]**(250/len(ret_cumu_equalwt))-1
std_annual_equalwt = ret_equalwted.std()*np.sqrt(250)
sharpe_ratio_equalwt = ret_annual_equalwt/std_annual_equalwt

#sharpe_table = [sharpe_ratio_riskP, sharpe_ratio_equalwt]
sharpe_table = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP.values),
                 ('equal_wted', sharpe_ratio_equalwt.values),
                 )))
sharpe_table1 = pd.Series(OrderedDict((('risk_parity', sharpe_ratio_riskP.values),
                 ('Stock ETF', sharpe_ratio_assets[0]),
                 ('IEF', sharpe_ratio_assets[1]),
                 ('GLD', sharpe_ratio_assets[2]),
                 )))
print('sharpe ratio of different strategies:\n',sharpe_table)
print('\nsharpe ratio of strategies vs assets:\n',sharpe_table1)
#%%
# compare the portfolio cumulative returns
figure_count = 1
plt.figure(figure_count)
figure_count = figure_count+1
pd.concat([ret_cumu_riskP, ret_cumu_equalwt], axis=1).plot()
plt.ylabel('Cumulative Return')
plt.show()

# compare the portfolio cumulative returns vs. asset returns
plt.figure(figure_count)
figure_count = figure_count+1
pd.concat([ret_cumu_riskP, ret_cumu_assets], axis=1).plot()
plt.ylabel('Cumulative Return')
plt.show()

# plot the historical weights of the assets
# area plot showing the weights
plt.figure(figure_count)
figure_count = figure_count + 1
wts.plot.area()
plt.ylabel('asset weights')


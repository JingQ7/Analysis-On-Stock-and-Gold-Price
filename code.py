import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot
from pandas import Series
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, breaks_cusumolsresid


#0. get data from csv file and get average, data cleaning
data = pd.read_csv('data.csv')
oldgold = data['Gold Price']
gold_data = oldgold.replace([np.inf, -np.inf], np.nan).dropna()

oldstock = data['Stock Index']
stock_data = oldstock.replace([np.inf, -np.inf], np.nan).dropna()

#-----------------------------------------------------------------
#1. data visualization
goldSeries = pd.Series(gold_data)
goldSeries.plot()
pyplot.show()

stockSeries = pd.Series(stock_data)
stockSeries.plot()
pyplot.show()
# result showed that is non-stationary time series

#-----------------------------------------------------------------
#2. check stationary-ADF test
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
adf_test(gold_data)
adf_test(stock_data)
# result shows the test statistic > critical value and p-value > 5%, so the series is non-stationary.

#-----------------------------------------------------------------
#3. first-diff
gold_diff0 = gold_data - gold_data.shift(1)
gold_diff.dropna().plot()
pyplot.show()

stock_diff0 = stock_data - stock_data.shift(1)
stock_diff.dropna().plot()
pyplot.show()
# result shows that series is stationary

gold_diff = gold_diff0.replace([np.inf, -np.inf], np.nan).dropna()
stock_diff = stock_diff0.replace([np.inf, -np.inf], np.nan).dropna()
#get data without nuh inf

#-----------------------------------------------------------------
#4. check stationary-ADF test after first-diff
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
adf_test(gold_diff)
adf_test(stock_diff)

#-----------------------------------------------------------------
#5.check stationary-visual test
goldSeries = pd.Series(gold_diff)
goldSeries.plot()
pyplot.show()

stockSeries = pd.Series(stock_diff)
stockSeries.plot()
pyplot.show()
# result showed that is non-stationary time series

#-----------------------------------------------------------------
#6. test for cointergration
result = sm.tsa.stattools.coint(gold_data, stock_data)
print(result)
# p-value =  0.4474903899530853 < 5% 10%
# then we can reject the null hypothesis that there is no cointegrating relationship

#-----------------------------------------------------------------
#7. VAR model
#built data
newdiff = {'goldPrice': gold_diff,'stockIndex': stock_diff}
newdiffSeries = pd.DataFrame(newdiff)
dataframe = newdiffSeries[['goldPrice','stockIndex']]

#choose lag
lag = select_order(dataframe, 12)
print(lag)

#model result
mod = sm.tsa.VAR(dataframe)
fitMod = mod.fit(2)
print(fitMod.summary())

#-----------------------------------------------------------------
#8. granger causality test 
granger_result = grangercausalitytests(dataframe, maxlag=2)
print(granger_result)

#The Null hypothesis for grangercausalitytests is that the time series
#in the second column, x2, does NOT Granger cause the time series in
#the first column, x1.

#-----------------------------------------------------------------
#9. Breusch Godfrey Lagrange Multiplier tests for residual autocorrelation
 
#resid
resid = mod.resid()
print(resid)

acorr_result = acorr_breusch_godfrey(resid, nlags=2)
print(acorr_result)

'''Returns:	
lm (float) – Lagrange multiplier test statistic
lmpval (float) – p-value for Lagrange multiplier test
fval (float) – fstatistic for F test,
fval (float) – pvalue for F test'''

#-----------------------------------------------------------------

''' Take another model into consideration
#9. VECM model

#built data
newdata = {'goldPrice': gold_data,'stockIndex': stock_data}
newdataSeries = pd.DataFrame(newdata)
dataframe = newdataSeries[['goldPrice','stockIndex']]
#print(dataframe)


#choose lag
lag = select_order(dataframe, 12)
print(lag)

#VECM model
mod1 = sm.tsa.VECM(dataframe, k_ar_diff=1)
fitMod = mod1.fit()
print(fitMod.summary())

'''

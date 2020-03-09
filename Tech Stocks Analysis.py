# Here in this Project, we'll look forward to analyse data from the stock market for the Top 5 tech stocks
# We'll use pandas library to extract data and analyse information, visualize it and look at different 
# perspective for risk analysis based on historical data.

#%% Importing Libraries
import numpy as np 
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime as dt
import matplotlib.pyplot as  plt
from matplotlib import style
import seaborn as sns

sns.set_style('whitegrid') 
style.use('ggplot')

#%%
end = dt.now()# Setting it to today

start = dt(end.year-1,end.month,end.day)# Start date set 1 year back from today

#%% We're going to analyse top 5 tech stocks: Apple, Microsoft, Amazon, Google and Ali Baba.
tech_list = ['AAPL','MSFT','AMZN','GOOG','BABA']

#%% Grabbing data using yahoo for the above stocks using pandas_datareader library.
for stock in tech_list:
    globals()[stock] = web.get_data_yahoo(stock,start,end)

#%% Viewing stock information
GOOG.tail()
GOOG.describe()
GOOG.info()

#%% Visualizing Adjusted Close and volume, to notice the volume of trades over time.
GOOG[['Adj Close','Volume']].plot(legend=True, subplots=True, figsize=(12,7))

#%% Adding Moving Average: 10,20,50 to the dataframe.
ma_days = [10,20,50]

for ma in ma_days:
    column_name = 'MA For %s Days' %(str(ma))
    GOOG[column_name] = GOOG['Adj Close'].rolling(window=ma,center = False).mean()
    
#%% 
#GOOG.tail()
GOOG[['Adj Close','MA For 10 Days','MA For 20 Days','MA For 50 Days']].plot(legend=True, subplots=False, figsize=(12,7))
#Moving Averages for 50 days are more smoother, as they're less reliable on daily fluctuations.

#%% Calculating Daily Returns using pandas.
GOOG['Daily Return'] = GOOG['Adj Close'].pct_change()
GOOG['Daily Return'].plot(legend=True,marker='o',linestyle='--',figsize=(13,7))

#%%
plt.figure(figsize=(13,6))
sns.distplot(GOOG['Daily Return'].dropna(), bins=100,color='red')

#%% Reading only Adj Close column 
close_df = web.DataReader(tech_list,'yahoo',start,end)['Adj Close']
close_df.tail()

#%% Calculating daily returns of the stocks to a new data frame, to check the correlation between the daily returns
rets_df = close_df.pct_change()
rets_df.tail()

#%%
plt.figure(figsize=(13,5))
sns.jointplot('MSFT','MSFT',rets_df,kind='scatter',color='blue')
# The relationship is perfectly linear, as we're trying to correlate something with itself.
# Further we'll check for different stocks
#%% 
sns.jointplot('MSFT','GOOG',rets_df,kind='scatter',color='blue')

#%%
plt.figure(figsize=(13,6))
sns.pairplot(rets_df.dropna())

#%%
rets_corr = rets_df.corr()
rets_corr

#%%
plt.figure(figsize=(13,7))
ax = sns.heatmap(rets_corr,vmin=-1,vmax=1,annot = True)
bottom, top =ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
# Microsoft and Apple seem to have the highest correlation.
# Secondly Google and Microsoft too have nice correlation.
# An intresting to note here is that all the tech stocks have a positive correlation with one another.

#%%
rets = rets_df.dropna()

#%% Calculating the value at risk we put by investing in a particular tech stock.
# The most basic way to quatify risk is to compare the expected return(mean of a stock's daily return) with the standard deviation of stock's daily return.
plt.figure(figsize=(12,6))
plt.title('Measuring Risk')
plt.scatter(rets.mean(),rets.std(),s=25)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(label,
                 xy=(x,y),xytext=(-10,10),
                 textcoords = 'offset points', ha='right', va='bottom',
                 arrowprops = dict(arrowstyle ='->',connectionstyle ='arc3,rad=-0.5'))

# We may wish to have invest in stock with less risk and more expected return.
# Amazon, Google and Microsoft seem to be the stable and less risk options.
# Meanwhile, Apple does have a high expected but comparebly more riskier than the all except for Ali Baba as it more risky and have less expected return than the others.

#%% Microsoft suits our criterion the best, so we go further analysing it and run simulations over.
# Calculating Value At Risk.
# Value at Risk is simply the amount of money we could expect to lose at a given confidence level.
# We go first with extracting the VAR values using the "Bootstrap Method",
#where we calculate the empirical quantile using histogram, 
#as the quantiles will help us at the very best to define our confidence interval.

plt.figure(figsize=(12,6))
sns.distplot(rets_df['MSFT'].dropna(),bins=100,color='red')

#%% Using pandas builtin quantile method
rets.head()
rets['MSFT'].quantile(0.05)
#The 0.05 empirical quantile of daily returns is at -0.0150. 
#This means that with 95% confidence, the worst daily loss will not exceed 2.55% (of the investment).

#%% Now we go with the Second Method "Monte Carlo Method" for predicting the stock behaviour.
# In this method, we run simulations to predict the future many times, 
#and aggregate the results in the end for some quantifiable value.
days = 365
dt=1/365
mu=rets.mean()['MSFT']
sigma=rets.std()['MSFT']

#%% Function takes in start_price, days , mu, sigma
def stock_monte_carlo(start_price,days,mu,sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for i in range(1,days):
        #Using the shock and drift formulas from the Monte Carlo Formula
        shock[i] = np.random.normal(loc = mu*dt, scale = sigma*np.sqrt(dt))
        drift[i] = mu*dt
        # New price = Old Price + Old Price*(drift+shock)
        price[i] = price[i-1] + (price[i-1] * (drift[i] + shock[i]))
        
        return price
    
#%%
MSFT.head()

#%% 
start_price = 112.389 # taken from above

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.title('Monte Carlo Analysis for Microsoft')
plt.xlabel('Days')
plt.ylabel('Price')

#%%
runs = 10000

simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
    
#%%
q = np.percentile(simulations,1)

plt.hist(simulations,bins=200)

plt.figtext(0.6,0.8,s="Start price: $%.2f" %start_price)

plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())

plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (start_price -q,))

plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)

plt.axvline(x=q, linewidth=4, color='r')

plt.title(u"Final price distribution for Microsoft Stock after %s days" %days, weight='bold')

#We can infer from this that, Microsoft's stock is pretty stable. 
#The starting price that we had was USD112.39.
# And the average final price over 10,000 runs was USD122.77.
#The red line indicates the value of stock at risk at the desired confidence interval.
#For every stock, we'd be risking USD3.49, 99% of the time.

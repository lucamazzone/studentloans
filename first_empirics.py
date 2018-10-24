from urllib.request import urlopen
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fredapi import Fred
fred = Fred(api_key='b8e0ac5377e623b6c813e300bc36614e')
from pandas import Series, DataFrame, Panel
from matplotlib.backends.backend_pdf import PdfPages
import plotly
from matplotlib.patches import Rectangle



plotly.tools.set_credentials_file(username='lumazz87', api_key='6LFRoy1iBns8G31bi96r')


# recessions are marked as 1 in the data
recs = fred.get_series('USREC')

# Select the two recessions over the time period
recs_9k = recs.ix['1990']
recs_2k = recs.ix['2001']
recs_2k8 = recs.ix['2008':'2009']

# now we can grab the indices for the start
# and end of each recession
recs2k_bgn = recs_2k.index[0]
recs2k_end = recs_2k.index[-1]

recs2k8_bgn = recs_2k8.index[0]
recs2k8_end = recs_2k8.index[-1]

recs_9k_bgn = recs_9k.index[0]
recs_9k_end = recs_9k.index[-1]




'''
url =  'https://www2.ed.gov/offices/OSFAP/defaultmanagement/peps300.xlsx'
defaultcohort_xlsx = urlopen(url)
def_data = pd.read_excel(defaultcohort_xlsx, index_col=0, parse_dates=True)



#print(yahoo.head())
sorted_by_gross = def_data.sort_values(['DRate 2'], ascending=False)
#print(sorted_by_gross.head(10))

def_data['DRate 3'].plot(kind="hist")
plt.show()

print(def_data.describe())
'''
yannelis_repay = pd.read_csv('repay_outcomes.csv', index_col=0, parse_dates=True)

#print(yannelis.tail())

fullsample = yannelis_repay.loc[yannelis_repay['full_sample'] == 1]
sample = fullsample.loc['1988':'2014']
default_2 = sample['i_cdr2']  #*fullsample['tot_bal']
default_3 = sample['i_cdr3']
default_5 = sample['i_cdr5']
#fig, ax = plt.subplots(figsize=(9, 5))
#ax.plot(default_2 , label='2 years')
#ax.plot(default_3 , label='3 years')
#ax.plot(default_5 , label='5 years')
#plt.show()

print(default_3)

#total_default.plot()
#plt.show()

#fullsample['alt_cdr2'].plot()
#plt.show()

peps_data = pd.read_excel('peps300.xlsx', index_col=0, parse_dates=True)
#print(peps_data)

total_students = peps_data["Dual\nDenom 3"].sum()
print(total_students)
total_defaulters = peps_data["Dual\nNum 3"].sum()
ratio_2012 = total_defaulters/total_students

total_students = peps_data["Dual\nDenom 2"].sum()
print(total_students)
total_defaulters = peps_data["Dual\nNum 2"].sum()
ratio_2013 = total_defaulters/total_students

total_students = peps_data["Dual\nDenom 1"].sum()
print(total_students)
total_defaulters = peps_data["Dual\nNum 1"].sum()
ratio_2014 = total_defaulters/total_students

#eeh = pd.Series([ratio_2012,ratio_2013,ratio_2014],index=['2012-01-01','2013-01-01','2014-01-01'])
default_2.loc['2012'] = ratio_2012
default_2.loc['2013'] = ratio_2013
default_2.loc['2014'] = ratio_2014


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(default_2 , label='2 years default rate')
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
ax.axvspan(recs2k_bgn, recs2k_end, color='grey', alpha=0.5)
ax.axvspan(recs2k8_bgn, recs2k8_end,  color='grey', alpha=0.5)
ax.axvspan(recs_9k_bgn, recs_9k_end,  color='grey', alpha=0.5)
fig.savefig("default_picture.pdf", bbox_inches='tight')
plt.show()



##################

yannelis_aggregate = pd.read_csv('Looney Yannelis Data Appendix 2/aggregate_fy_stocks.csv', index_col=0, parse_dates=True)
fullsample2 = yannelis_aggregate.loc[ (yannelis_aggregate['full_sample'] == 1)  & (yannelis_aggregate['grad_bor'] > 1) ]
used_sample = fullsample2.loc['1988':'2013']
total_balances = used_sample['tot_bal']/used_sample['cpi_adj']
#total_balances.plot()
#plt.title('Real Balances')
#plt.show()



data = fred.get_series('GDP')

used_gdp = data.loc['1988':'2017']
#used_gdp.plot()
#plt.show()

debt = used_sample['tot_bal']

#print(pd.PeriodIndex(['1988', '2013'], freq='A'))

#print(used_gdp)
#used_gdp.dropna(axis=0, how='all', inplace=True)
#used_gdp['month'] = used_gdp.astype('timedelta64[D]').Date1.apply(dt.date.strftime, args=('%Y.%m',))
#used_gdp.groupby(['month', 'Reference'])['Value'].aggregate(sum).unstack()

#used_gdp.groupby('release_year')
annual_gdp = used_gdp.resample('AS').mean()
#annual_gdp.plot()
#plt.show()



#share  = debt/(annual_gdp*1000)

#share.plot()
#plt.show()


#

portfolio = pd.read_excel('PortfolioSummary.xls', index_col=0, parse_dates=True)
#print(portfolio.iloc[15:30,7].index)

#print(portfolio.iloc[15:25,7])
#debt.append(portfolio.iloc[15,7])
#print(debt)

ciao = pd.Series([portfolio.iloc[15,7],portfolio.iloc[19,7],portfolio.iloc[23,7],portfolio.iloc[27,7]],index=['2014-01-01','2015-01-01','2016-01-01','2017-01-01'])
print(ciao)
#print(debt.append(ciao))
print(debt.index.union(ciao.index))

debt = debt.reindex(debt.index.union(ciao.index))
print(debt)
print(debt.append(ciao))
debt.loc['2014'] = portfolio.iloc[15,7]*1e3
debt.loc['2015'] = portfolio.iloc[19,7]*1e3
debt.loc['2016'] = portfolio.iloc[23,7]*1e3
debt.loc['2017'] = portfolio.iloc[27,7]*1e3
print(debt)

share  = debt/(annual_gdp*1000)

print(share)
share.plot()
plt.show()








mortgage_data = fred.get_series('MVLOAS')  # MDOAH is total mortgage, MDOTHIOH is individual mortgages, HHMSDODNS is hh+nonprofit mortgages,
# MVLOAS is motor vehicle loans
used_mortgage = mortgage_data.loc['1988':'2017']
annual_mortgage = used_mortgage.resample('AS').mean()

hh_totdebt = fred.get_series('HDTGPDUSQ163N')
used_hhtotdebt = hh_totdebt.loc['1988':'2017']
annual_hhtotdebt = used_hhtotdebt.resample('AS').mean()

mortgage_ratio = annual_mortgage/(annual_gdp)

fig, ax = plt.subplots()  #figsize=(9, 5)
ax.plot(share, 'b-',label = 'Student Debt',linewidth=3.0) #'r-',share.index,
ax.plot(mortgage_ratio,'r-', label='Car Loans',linewidth=3.0) #,'b-', mortgage_ratio.index ,
ax.axvspan(recs2k_bgn, recs2k_end, color='grey', alpha=0.5)
ax.axvspan(recs2k8_bgn, recs2k8_end,  color='grey', alpha=0.5)
ax.axvspan(recs_9k_bgn, recs_9k_end,  color='grey', alpha=0.5)
ax.legend(loc='lower right')
#vals = ax.get_yticks()
#ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
#update = {'data':[{'fill': 'tozeroy'}]}  # this updates BOTH traces now
#plot_url = plotly.plotly.plot_mpl(fig, update=update, filename='mpl-multi-fill')
plt.show()


fig, ax1 = plt.subplots()
lns1 = ax1.plot(share, 'b-',label = 'Student Debt')
#ax1.set_xlabel('time (s)')
# Make the y-axis label, ticks and tick labels match the line color.
#ax1.set_ylabel('exp', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
lns2 = ax2.plot(mortgage_ratio, 'r-', label='Household Mortgages')
#ax2.set_ylabel('sin', color='r')
ax2.tick_params('y', colors='r')

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs) #, loc=0

fig.tight_layout()
fig.savefig("trends_in_debt.pdf", bbox_inches='tight')

plt.show()




import plotly.plotly as py
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import plotly
import matplotlib.pyplot as plt
import requests
import lxml.html as lh
import sys
from bs4 import BeautifulSoup
import urllib.request
from bs4 import BeautifulSoup
import html5lib
from urllib.request import urlopen
from fredapi import Fred
fred = Fred(api_key='b8e0ac5377e623b6c813e300bc36614e')
from params import *
from itertools import cycle, islice



plotly.tools.set_credentials_file(username='lumazz87', api_key='6LFRoy1iBns8G31bi96r')

def wage(y, theta, x, t):
    cosa1 = 0
    for i in range(t):
        cosa1 = cosa1 + beta * (beta * (1 - lamb)) ** i
    return f(A, y, x) - k_fun(y)/(cosa1*q_fun(theta)) -  (max(y - g_fun(y,x),0))*5

def f( A, y, x):
    return A * (y**alpha  + g_fun(y, x)**alpha )**(1/alpha) #A * (y**alpha)*g_fun(y, x)**(1-alpha)

def l_fun(dis):
    if np.abs(dis):
        dis = dis/np.abs(dis)
    return 1/(1+np.exp(-2*100*dis))


def q_fun(theta):
    if (theta==0.0):
        return 1.0
    else:
        return (1 - np.exp(-eta * theta))/theta

def k_fun(y):
    return kappa*(y**gamma)/gamma

def g_fun(p, x):
    y = p*adj
    return x + phi*(y-x)*l_fun(y-x)


'''
scl = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'], \
       [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]
'''



##### THIS IS ONE MAP #######
'''
df = pd.read_excel('https://www.newyorkfed.org/medialibrary/Interactives/householdcredit/data/xls/student-loan-by-state.xlsx',
                   sheetname='T2.average debt per borrower', index_col=0,skipcolumns=1,skiprows=5)  #, parse_dates=True
print(df.iloc[1:10,:])

growth = 100*(df['4Q2010'] - df['4Q2004'])/df['4Q2004']

for col in df.columns:
    df[col] = df[col].astype(str)
    growth = growth.astype(str)


scl = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'], \
       [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]



print(df['state'])



df['text'] = df['state'] + '<br>' + \
              '4Q2004' + df['4Q2004']  + '4Q2005' + df['4Q2005'] + '<br>' + \
              '4Q2006' + df['4Q2006']  + '4Q2007' + df['4Q2007'] + '<br>' + \
              '4Q2008' + df['4Q2008']  + '4Q2009' + df['4Q2009'] + '<br>' + \
              '4Q2010' + df['4Q2010']  + '4Q2011' + df['4Q2011'] + '<br>' + \
              '4Q2012' + df['4Q2012']



data = [dict(
    type='choropleth',
    colorscale=scl,
    autocolorscale=True,
    locations=df['state'],
    z=growth.astype(float),          #df['4Q2009'].astype(float),
    locationmode='USA-states',
    text=df['text'],
    marker=dict(
        line=dict(
            color='rgb(255,255,255)',
            width=1
        )),
    colorbar=dict(
        title="percent")
)]


layout = dict(
    title='Growth Rate of average debt per borrower', #<br>(absolute numbers)
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(200, 255, 255)'),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='d3-cloropleth-map')
#py.image.save_as(fig, filename='a-simple-plot.png')

##### THIS IS ANOTHER MAP #######

data_debt = pd.read_excel('https://www.newyorkfed.org/medialibrary/Interactives/householdcredit/data/xls/student-loan-by-state.xlsx',
                   sheetname='T3.delinquent borrowers', index_col=0,skipcolumns=1,skiprows=5)  #, parse_dates=True
print(data_debt.iloc[1:10,:])

growth = 100*(data_debt['4Q2012'] - data_debt['4Q2004'])/data_debt['4Q2004']

for col in data_debt.columns:
    data_debt[col] = data_debt[col].astype(str)
    growth = growth.astype(str)


scl = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'], \
       [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]


data_debt['text'] = data_debt['Unnamed: 1'] + '<br>' + \
              '4Q2004' + data_debt['4Q2004']  + '4Q2005' + data_debt['4Q2005'] + '<br>' + \
              '4Q2006' + data_debt['4Q2006']  + '4Q2007' + data_debt['4Q2007'] + '<br>' + \
              '4Q2008' + data_debt['4Q2008']  + '4Q2009' + data_debt['4Q2009'] + '<br>' + \
              '4Q2010' + data_debt['4Q2010']  + '4Q2011' + data_debt['4Q2011'] + '<br>' + \
              '4Q2012' + data_debt['4Q2012']



data = [dict(
    type='choropleth',
    colorscale=scl,
    autocolorscale=True,
    locations=data_debt['Unnamed: 1'],
    z=growth.astype(float),          #df['4Q2009'].astype(float),
    locationmode='USA-states',
    text=data_debt['text'],
    marker=dict(
        line=dict(
            color='rgb(255,255,255)',
            width=1
        )),
    colorbar=dict(
        title="percent")
)]


layout = dict(
    title='Change in percentage of defaulting borrowers', #<br>(absolute numbers)
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(200, 255, 255)'),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='d3-cloropleth-map')
py.image.save_as(fig, filename='change_in_default.png')


'''
#########################################################

'''

repay_plan = pd.read_excel('https://studentaid.ed.gov/sa/sites/default/files/fsawg/datacenter/library/DLPortfoliobyRepaymentPlan.xls',
                   sheetname='DLPortfoliobyRepaymentPlan', header=0, index_col=0,skiprows=5)  #, parse_dates=True


repay_plan.drop(repay_plan.tail(3).index,inplace=True)
repay_plan.drop(repay_plan.head(1).index,inplace=True)



total_pop = repay_plan['Unnamed: 3'] + repay_plan['Unnamed: 5']  + repay_plan['Unnamed: 7'] \
            + repay_plan['Unnamed: 9'] + repay_plan['Unnamed: 11']+ repay_plan['Unnamed: 13']  + \
            repay_plan['Unnamed: 15']

notibr_total = repay_plan['Unnamed: 3'] + repay_plan['Unnamed: 5'] + repay_plan['Unnamed: 7']+ repay_plan['Unnamed: 9']
ibr_total = repay_plan['Unnamed: 11'] + repay_plan['Unnamed: 13'] + repay_plan['Unnamed: 15']

notibr_ratio = notibr_total / total_pop
ibr_ratio = ibr_total / total_pop

#fig, ax = plt.subplots()  #figsize=(9, 5)
#ax.plot(notibr_ratio, label='borrowers not on IBR') #'r-',share.index,
#ax.plot(ibr_ratio, label='borrowers on IBR') #,'b-', mortgage_ratio.index ,
#ax.legend(loc='lower right')
#vals = ax.get_yticks()
#ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
#update = {'data':[{'fill': 'tozeroy'}]}  # this updates BOTH traces now
#plot_url = plotly.plotly.plot_mpl(fig, update=update, filename='mpl-multi-fill')
#plt.show()


df2 = pd.DataFrame({'not on IBR': notibr_total, 'on IBR': ibr_total}, columns=['not on IBR', 'on IBR'])

#plt.figure()
df2.plot.bar(stacked=True)
#plt.show()


###################################################################



df = pd.read_excel('/Users/lucamazzone/iCloud Drive (Archive)/Desktop/Work/Student Loans/FAFSA/FAFSAdata.xlsx',
                   sheetname='Cost of tuition', dtype={'Year and control of institution':str},
                   header=[0,1],  index_col= 0, parse_dates=True, skiprows=1 )  #


constant_dollars = df['Constant 2015–16 dollars1']
all_constant = constant_dollars['All\ninstitutions'][0:18]  # public, private non profit and for profit
public_constant = constant_dollars['All\ninstitutions'][19:37]
private_constant = constant_dollars['All\ninstitutions'][38:57]


data = fred.get_series('MEHOINUSA672N')  # Real Median Household Income in the United States

used_gdp = data.loc['1996':'2016']
last_90 = all_constant.get_value('1996')


ciao = pd.Series([last_90,last_90,last_90,last_90],
                 index=['1997-01-01','1998-01-01','1999-01-01','2000-01-01'])



all_constant = all_constant.reindex(all_constant.index.union(ciao.index))


all_constant.loc['1997'] = ciao.iloc[0]
all_constant.loc['1998'] = ciao.iloc[0]
all_constant.loc['1999'] = ciao.iloc[0]
all_constant.loc['2000'] = ciao.iloc[0]



nuovo = all_constant/used_gdp
'''

'''
fig, ax = plt.subplots()  #figsize=(9, 5)
ax.plot(public_constant, label='public schools, all institutions') #'r-',share.index,
ax.plot(private_constant, label='private schools, all institutions')
ax.legend(loc='lower right')
plt.show()
'''
##################################################################
'''
enrol_pop_data = pd.read_excel('https://nces.ed.gov/programs/digest/d16/tables/xls/tabn302.60.xls',
                   header=0,  index_col= 0, parse_dates=True,skiprows=2 )  # index_col= 0, parse_dates=True,


enrol_pop_data  = enrol_pop_data.loc[enrol_pop_data.index.dropna()]
perc_enrolled = enrol_pop_data['Total, all students']
level = enrol_pop_data['Level of institution']
enrol_ratio = perc_enrolled[25:47]
new_index = all_constant.index
enrol_ratio = enrol_ratio.reset_index()
enrol_ratio = enrol_ratio.drop(['Year'], axis=1)
enrat = enrol_ratio.reindex(new_index)
for i,a in enumerate(new_index):
    enrat.loc[a] = enrol_ratio.iloc[i]*0.01
enrat.loc['1986']  = 'NaN'


fig, ax = plt.subplots()  #figsize=(9, 5)
ax.plot(enrat, 'r-', label='share of enrolled students on aged 18-24',linewidth=3.0) #'r-',share.index,
ax.plot(nuovo, 'b-', label='average fees on median household income',linewidth=3.0) #'r-',share.index,
ax.legend(loc='lower right')
plt.show()


'''
#############################################################

'''
enrol_pop_data = pd.read_excel('TrendStats_institution.xls', sheetname='new_data', skiprows=2, index_col=0) #skiprows=1, index_col=0, parse_dates=True

print(enrol_pop_data.head())

Total = enrol_pop_data.iloc[2:8,0:3]


my_colors = list(islice(cycle(['b','orange', 'r' , 'y', 'g']), None, len(Total)))



Research = enrol_pop_data.iloc[12:18,0:3]

Liberal = enrol_pop_data.iloc[40:46,0:3]

PfP = enrol_pop_data.iloc[124:130,0:3]

print(PfP)


fig, axes = plt.subplots(nrows=2, ncols=2)
plt.figure()
Total.plot.bar(stacked=False,ax=axes[0,0],rot=0,legend=False,color=my_colors); axes[0,0].set_title('Total')
Research.plot.bar(stacked=False,ax=axes[0,1],rot=0,legend=False,color=my_colors); axes[0,1].set_title('Research University')
Liberal.plot.bar(stacked=False,ax=axes[1,0],rot=0,legend=False,color=my_colors); axes[1,0].set_title('Liberal Arts College')
PfP.plot.bar(stacked=False,ax=axes[1,1],rot=0,legend=True,color=my_colors); axes[1,1].set_title('Private for Profit')
plt.show()



print(Research)
'''




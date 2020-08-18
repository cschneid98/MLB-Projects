#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import essentials
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Load in data
dframe = pd.read_excel(r'Downloads\Draft Datasets\MLB Draft Pick Database.xlsx')
print (dframe)


# In[4]:


dframe['Signed'].value_counts()


# In[5]:


dframe.pivot_table(index=['Signed'], columns='Type', aggfunc='size', fill_value=0)


# In[6]:


df = dframe[dframe['Signed'] == 'Y']


# In[7]:


#Make new Made Majors dummy variable column
def Made_Majors(df):
    
    if df['G'] > 0 or df['G.1'] > 0:
        return 1
    else:
        return 0
    
df['Made_Majors'] = df[['G','G.1']].apply(Made_Majors,axis=1)


# In[8]:


#Make league variable column
AL = {'Red Sox', 'Yankees', 'Blue Jays', 'Rays', 'Orioles', 'Tigers', 'Indians', 'Royals', 'Twins', 'White Sox', 'Astros', 'Athletics','Rangers','Angels','Mariners'}
NL = {'Expos','Braves', 'Mets', 'Nationals', 'Phillies', 'Marlins', 'Cardinals', 'Cubs', 'Reds', 'Pirates', 'Brewers', 'Giants', 'Dodgers','Diamondbacks','Rockies','Padres'}

def League(df):
    
    if df['Tm'] in AL:
        return 'American'
    else:
        return 'National'

df['League'] = df[['Tm']].apply(League,axis=1)


# In[9]:


#Make state variable column
df['State'] = df['Drafted Out of'].str[-3:-1]
df['State'].value_counts()


# In[10]:


#Fix State column for players from Puerto Rico and Canada
df.loc[df['State'] == 'AB', 'State'] = 'CAN'
df.loc[df['State'] == 'QC', 'State'] = 'CAN'
df.loc[df['State'] == 'BC', 'State'] = 'CAN'
df.loc[df['State'] == 'NS', 'State'] = 'CAN'
df.loc[df['State'] == 'ON', 'State'] = 'CAN'
df.loc[df['State'] == 'co', 'State'] = 'PR'
df['State'].value_counts()


# In[11]:


#EDA Pt1
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
graf = df.groupby(['Tm','Made_Majors']).mean().sort_values('WAR',ascending=False)


# In[12]:


#Average WAR per player that made majors per team
sns.boxplot(graf[1:31])


# In[13]:


#Counts of players who made majors based on league
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=280, facecolor='w', edgecolor='k')
sns.countplot('League',data=df,hue='Made_Majors',palette='coolwarm')


# In[14]:


#Made Major counts based on type of institution
figure(num=None, figsize=(8, 6), dpi=280, facecolor='w', edgecolor='k')
sns.countplot('Type',data=df,hue='Made_Majors',palette='coolwarm')


# In[15]:


MajorsDF = df[df['Made_Majors'] == 1]

sns.distplot(MajorsDF['WAR'])

# df.groupby(['Tm','Made_Majors']).mean().sort_values('WAR',ascending=False)


# In[16]:


df['Made_Majors'] = df['Made_Majors'].astype(int)
pp = df.pivot_table(index='Tm', columns='Made_Majors', aggfunc='size', fill_value=0)
pp['Prop'] = (pp.iloc[:, 1]) / (pp.iloc[:, 0] + pp.iloc[:, 1]) * 100
pp.sort_values(['Prop'])


# In[17]:


pvt = pd.pivot_table(df, index='Year',columns='Made_Majors', aggfunc='size', fill_value=0)
pvt['Prop'] = (pvt.iloc[:, 1]) / (pvt.iloc[:, 0] + pvt.iloc[:, 1]) * 100
pvt.sort_values(['Prop'])


# In[18]:


pvt.iloc[:, 0:2].plot()


# In[19]:


pivot2 = pd.pivot_table(df, index=['Year','Made_Majors'], columns='Type', aggfunc='size', fill_value=0)
pivot2


# In[20]:


df['Year'] = df['Year'].astype(float)
df['Drafted Out of'] = df['Drafted Out of'].astype(str)


# In[21]:


sns.catplot(x="Year", y="WAR", data=df)


# In[22]:


#Counts of making majors based on type of institution
df.pivot_table(index=['Made_Majors'], columns='Type', aggfunc='size', fill_value=0)


# In[23]:


#WAR change over the years
sns.lmplot('Year','WAR',MajorsDF,order=2,
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})


# In[24]:


df


# In[25]:


#Plot of draft pick vs WAR
sns.lmplot('OvPck','WAR',MajorsDF,hue='Made_Majors')


# In[26]:


#Make new Made Majors dummy variable column
def Player_Performance(df):
    
    if df['WAR'] <= 0 and df['Year'] > 2013:
        return 'Prospect'
    elif df['WAR'] < 5 and df['Year'] < 2013:
        return 'Bust'
    elif 5 < df['WAR'] < 15:
        return 'Serviceable Player'
    elif 15 < df['WAR'] < 25:
        return 'Good Player'
    elif 25 < df['WAR'] < 35:
        return 'Great Player'
    elif 25 < df['WAR'] < 35 and df['Year'] > 2008:
        return 'Great Player'
    elif 35 < df['WAR'] < 50:
        return 'Stud'
    elif df['WAR'] > 50:
        return 'Potential HOF'
df['Player_Performance'] = df[['WAR','Year']].apply(Player_Performance,axis=1)


# In[27]:


df.pivot_table(index=['Tm'], columns='Player_Performance', aggfunc='size', fill_value=0)


# In[28]:


pivot3 = df.pivot_table(index=['Year'], columns='Player_Performance', aggfunc='size', fill_value=0)
pivot3.plot()


# In[29]:


df


# In[30]:


pivot5 = df.pivot_table(index=['Rnd'], values='WAR', aggfunc='sum', fill_value=0)
pivot5


# In[ ]:





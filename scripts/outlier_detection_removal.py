#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


# In[26]:


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_colwidth",None)
pd.option_context('mode.use_inf_as_na', True)


# In[6]:


df1=pd.read_csv("after_univariate.csv",index_col="Unnamed: 0")
df2=pd.read_csv("laptop_cleaned_dataset.csv")
df=df1.copy()


# In[8]:


df["thickness_num"]=df2["thickness"]
df["weight_num"]=df2["weight"]


# In[151]:


df.info()


# In[152]:


cat_cols=["brand","thickness","weight","processor_brand","antiglare","touch_screen","hdmi","ethernet","multi_card_reader","thunderbolt","display_port","vga","backlit","fingerprint_sensor","usb2","usb3","typec","processor_gen","processor_brand","processor_model","graphics_brand","graphics_model","everyday_use","business","performance","gaming","popularity","quality_type","ppi_type"]
for col in cat_cols:
    df[col]=df[col].astype("category")


# # 1.Price

# In[24]:


df["price"].describe()


# In[56]:


sns.histplot(df["price"],kde=True)


# In[48]:


px.box(x=df["price"])


# In[51]:


q1=df["price"].quantile(0.25)
q2=df["price"].quantile(0.75)
iqr=q2-q1
lower_bound=q1-1.5*iqr
upper_bound=q2+1.5*iqr
outliers=df[(df["price"]<lower_bound) | (df["price"]>upper_bound)]
num_outliers=outliers.shape[0]
outliers_stats=outliers["price"].describe()
num_outliers,outliers_stats


# In[58]:


sns.histplot(outliers["price"],kde=True,bins=30)


# In[61]:


outliers[outliers["price"]>500000]
#so by looking at the data we saw that all the outlier points are valid


# # 2.Screen Size

# In[65]:


sns.histplot(df["screen_size"],kde=True)


# In[67]:


px.box(df["screen_size"])
#There are not much outlier kde plot shows normal distribution not much outliers everything looks fine.


# # 3.PPI

# In[72]:


sns.histplot(df["ppi"],kde=True)
#data is right skewed


# In[74]:


px.box(df,x="ppi")


# In[82]:


lower_bound=127
upper_bound=178
outliers=df[(df["ppi"]<127)|(df["ppi"]>178)]
outliers.shape#There are 175 outliers
outliers["ppi"].describe()#values are not much higher or much lower so the outliers are valid.


# # 4.Threads

# In[84]:


px.box(df,x="threads")


# # 5.Ram

# In[85]:


px.box(df,x="ram")


# # 6.Cores

# In[86]:


px.box(df,x="cores")


# # 7.Battery Capacity

# In[87]:


px.box(df,x="battery_capacity")#There is one outlier where the battery shows 6Wh


# In[88]:


sns.kdeplot(df["battery_capacity"])


# In[97]:


df[df["battery_capacity"]==6]
df2.iloc[173]#i didn't find the correct source of information but by seeing data i think it is an outlier 


# In[100]:


df["battery_capacity"]=df["battery_capacity"].replace(6,np.nan)


# In[101]:


px.box(df,x="battery_capacity")


# # 8.Battery Cell

# In[103]:


sns.histplot(df["battery_cell"],kde=True)


# # 9.ssd

# In[104]:


sns.histplot(df["ssd"],kde=True)


# In[114]:


q1=df["ssd"].quantile(0.25)
q2=df["ssd"].quantile(0.75)
iqr=q2-q1
lower_bound=q1-1.5*iqr
upper_bound=q2+1.5*iqr
outliers=df[(df["ssd"]<lower_bound)|(df["ssd"]>upper_bound)|(df["ssd"]!=0)]
outliers.shape[0]
outliers["ssd"].describe()


# In[155]:


df[df["ssd"]==0].shape#ssd are only 0 where hdd present in the laptop


# # 10.HDD

# In[154]:


df[df["hdd"].notna()].shape


# In[158]:


px.box(df,x="hdd")#there are verry few values are present in the hdd columns but all are valid


# # 11.Thickness_num

# In[120]:


px.box(df,x="thickness_num")# there are some laptops which are clearly seen as outliers


# In[119]:


sns.histplot(df["thickness_num"],kde=True)


# In[131]:


df.loc[df["thickness_num"] > 30.8, "thickness_num"] = np.nan#i covert all the values that are greater than 30.8 to nan values


# In[136]:


df.loc[df["thickness_num"]==0,"thickness_num"]=np.nan


# In[141]:


df[df["thickness_num"]<9]#these laptops are heavy it means the thickness is not given in mm they are given in inches 
df.loc[df["thickness_num"]<9,"thickness_num"]*=25.4#here i convert them to mm.


# In[145]:


px.box(df,x="thickness_num")
sns.histplot(df["thickness_num"],kde=True)
#Now i think the values are correct and the outliers are valid


# # 12.Weight_num

# In[146]:


sns.histplot(df["weight_num"],kde=True)


# In[148]:


px.box(df,x="weight_num")
#it looks like all points are valid


# In[159]:


df["graphics_capacity"].value_counts()


# In[162]:


num_cols=["price","screen_size","ppi","threads","ram","cores","battery_capacity","battery_cell","graphics_capacity","hdd","ssd","thickness_num","weight_num"]
for col in num_cols:
    if df[col].dtype=="int64":
        df[col]=df[col].astype("int32")
    elif df[col].dtype=="float64":
        df[col]=df[col].astype("float32")
    else:
        df[col]=df[col].astype("float32")
        


# In[169]:


df.info()


# In[171]:


df.to_csv('outlier_detection.csv', index=False)


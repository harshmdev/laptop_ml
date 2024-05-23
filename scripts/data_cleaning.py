#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[6]:


df1=pd.read_csv("data/raw/pre_cleaning.csv")
df=df1.copy()



#her i replace "," with nothing
df["num_votes"]=df["num_votes"].str.replace(",","")


# In[66]:


#here we drop num_reviews column
df.drop("num_reviews",axis=1,inplace=True)


# In[68]:


#here i drop os column
df.drop("os",axis=1,inplace=True)


# In[148]:


# here i make a copy of df just in case any mistake happens
temp_df=df.copy()


# In[149]:


#in this case we fetch all the values which contains "thickness" in "utility" column store it in thickness column and return null value
x=df[df["utility"].str.startswith("Thickness:")]
temp_df.loc[x.index,"thickness"]=x["utility"]
temp_df.loc[x.index,"utility"]=np.nan


# In[151]:


#we fetch values of weight from warranty column store it in weight and return null values
x=df[~df["warranty"].str.contains("Year")]
temp_df.loc[x.index,"weight"]=x["warranty"]
temp_df.loc[x.index,"warranty"]=np.nan


# In[152]:


#in this we replace any value which not contain thickness in thickness column with null value.
x=temp_df[~temp_df["thickness"].str.startswith("Thickness:")]
temp_df.loc[x.index,"thickness"]=np.nan


# In[153]:


#we replace any value which not contain weight in "weight" column with null value
x=temp_df[temp_df["weight"].str.contains("|".join(["Thickness:","Utility:"]))]
temp_df.loc[x.index,"weight"]=np.nan


# In[156]:


#here we again store temp_df to df
df=temp_df


# In[178]:


#this step is to be done so that masking should be applied
df["antiglare"].fillna("False",inplace=True)
df["aspect_ratio"].fillna("False",inplace=True)
df["touchscreen"].fillna("False",inplace=True)


# In[181]:


temp_df=df


# In[185]:


#first shift 2 antiglare values in ppi to antiglare column
x=temp_df[temp_df["ppi"].str.contains("Anti Glare")]
temp_df.loc[x.index,"antiglare"]=x["ppi"]


# In[189]:


#now do the same thing for the ppi column 
x=temp_df[temp_df["ppi"].str.contains("Anti Glare")]
temp_df.loc[x.index,"ppi"]=x["resolution"]


# In[192]:


#here we set the values in resolution in which ppi is given to null
temp_df.loc[temp_df[temp_df["resolution"].str.contains("PPI")].index,"resolution"]=np.nan


# In[210]:


#in this we make a new column "antiglare1" and store all antiglare values from antiglare touchscreen and aspect ratio column
x=temp_df[temp_df["antiglare"].str.contains("Anti Glare")]
y=temp_df[temp_df["aspect_ratio"].str.contains("Anti Glare")]
z=temp_df[temp_df["touchscreen"].str.contains("Anti Glare")]
temp_df["antiglare1"]=x["antiglare"]
temp_df.loc[y.index,"antiglare1"]=y["aspect_ratio"]
temp_df.loc[z.index,"antiglare1"]=z["touchscreen"]


# In[211]:


#in this we make a new column "aspect_ratio1" and store all antiglare values from antiglare touchscreen and aspect ratio column
x=temp_df[temp_df["antiglare"].str.contains("Aspect Ratio")]
y=temp_df[temp_df["aspect_ratio"].str.contains("Aspect Ratio")]
z=temp_df[temp_df["touchscreen"].str.contains("Aspect Ratio")]
temp_df["aspect_ratio1"]=x["antiglare"]
temp_df.loc[y.index,"aspect_ratio1"]=y["aspect_ratio"]
temp_df.loc[z.index,"aspect_ratio1"]=z["touchscreen"]


# In[215]:


#in this we make a new column "touch_screen1" and store all antiglare values from antiglare touchscreen and aspect ratio column
x=temp_df[temp_df["antiglare"].str.contains("Touch")]
y=temp_df[temp_df["aspect_ratio"].str.contains("Touch")]
z=temp_df[temp_df["touchscreen"].str.contains("Touch")]
temp_df["touch_screen1"]=x["antiglare"]
temp_df.loc[y.index,"touch_screen1"]=y["aspect_ratio"]
temp_df.loc[z.index,"touch_screen1"]=z["touchscreen"]


# In[220]:


# here we drop extra columns
temp_df.drop(["antiglare","aspect_ratio","touchscreen"],axis=1,inplace=True)
df=temp_df


# ## Cores

# In[256]:


#in this we delete "cores" column because same information is in "threads" column.
df.drop("cores",axis=1,inplace=True)


# ## Threads

# In[283]:


#in this we split this column into two columns "cores","threads"
df["cores"]=df["threads"].str.split(",").str[0]
df["cores"].fillna("False",inplace=True)
df["cores1"]=df[df["cores"].str.contains("Core")]["cores"]
    


# In[298]:


temp_df=df.copy()


# In[299]:


temp_df["threads"]=temp_df["threads"].str.split(",")


# In[317]:


temp_df.fillna("False",inplace=True)


# In[ ]:


list=[]
for i in temp_df["threads"]:
    if len(i)==2:
        list.append(i[1])
    elif i[0].endswith("Threads"):
        list.append(i[0])
    else:
        list.append(np.nan)

temp_df["threads"]=pd.Series(list)


# In[323]:


df["threads"]=temp_df["threads"] 


# ## Cache

# In[325]:


df.drop("cache",axis=1,inplace=True)


# ## Battery

# In[329]:


df["battery1"].fillna("False",inplace=True)
new_list=[]
for i in df["battery1"]:
    my_list = ast.literal_eval(i)
    new_list.append(my_list)
df["battery"]=pd.Series(new_list)


# In[338]:


temp_df=df.copy()


# In[406]:


capacity=[]
cell=[]
for i in temp_df["battery"]:
    if i is not False:
        for j in i:
            if len(j.split(","))==2:
                for k in j.split(","):
                    if "Wh" in k:
                        capacity.append(k)
                    elif "Cell" in k:
                        cell.append(k)
            elif len(j.split(","))==1:
                for k in j.split(","):
                    if "Wh" in k:
                        capacity.append(k)
                        cell.append(np.nan)
                    elif "Cell" in k:
                        cell.append(k)
                        capacity.append(np.nan)
                    else:
                        capacity.append(np.nan)
                        cell.append(np.nan)
                
            
    else:
        capacity.append(np.nan)
        cell.append(np.nan)
temp_df[["battery_capacity","battery_cell"]]=pd.Series({"battery_capacity":capacity,"battery_cell":cell})

        


# In[413]:


df=temp_df


# ## HDMI

# In[424]:


df["hdmi"].str.split(",").str.len().value_counts()


# In[425]:


temp_df=df.copy()


# In[431]:


temp_df["hdmi"]=temp_df["hdmi"].str.split(",")


# In[450]:


temp_df["hdmi"].fillna("False",inplace=True)


# In[455]:


col1=[]
col2=[]
col3=[]
col4=[]
col5=[]
for i in temp_df["hdmi"]:
    if i is not False:
        if len(i)==5:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(i[2])
            col4.append(i[3])
            col5.append(i[4])
        elif len(i)==4:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(i[2])
            col4.append(i[3])
            col5.append(np.nan)
        elif len(i)==3:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(i[2])
            col4.append(np.nan)
            col5.append(np.nan)
        elif len(i)==2:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(np.nan)
            col4.append(np.nan)
            col5.append(np.nan)
        elif len(i)==1:
            col1.append(i[0])
            col2.append(np.nan)
            col3.append(np.nan)
            col4.append(np.nan)
            col5.append(np.nan)
    else:
        col1.append(np.nan)
        col2.append(np.nan)
        col3.append(np.nan)
        col4.append(np.nan)
        col5.append(np.nan)

temp_df["col1"]=col1
temp_df["col2"]=col2
temp_df["col3"]=col3
temp_df["col4"]=col4
temp_df["col5"]=col5
       


# In[464]:


temp_df["col1"].fillna("False",inplace=True)
temp_df["col2"].fillna("False",inplace=True)
temp_df["col3"].fillna("False",inplace=True)
temp_df["col4"].fillna("False",inplace=True)
temp_df["col5"].fillna("False",inplace=True)


# In[467]:


a=temp_df[temp_df["col1"].str.contains("HDMI")]
b=temp_df[temp_df["col2"].str.contains("HDMI")]
c=temp_df[temp_df["col3"].str.contains("HDMI")]
d=temp_df[temp_df["col4"].str.contains("HDMI")]
e=temp_df[temp_df["col5"].str.contains("HDMI")]
temp_df["hdmi1"]=a["col1"]
temp_df.loc[b.index,"hdmi1"]=b["col2"]
temp_df.loc[c.index,"hdmi1"]=c["col3"]
temp_df.loc[d.index,"hdmi1"]=d["col4"]
temp_df.loc[e.index,"hdmi1"]=e["col5"]


# In[471]:


a=temp_df[temp_df["col1"].str.contains("Ethernet")]
b=temp_df[temp_df["col2"].str.contains("Ethernet")]
c=temp_df[temp_df["col3"].str.contains("Ethernet")]
d=temp_df[temp_df["col4"].str.contains("Ethernet")]
e=temp_df[temp_df["col5"].str.contains("Ethernet")]
temp_df["ethernet"]=a["col1"]
temp_df.loc[b.index,"ethernet"]=b["col2"]
temp_df.loc[c.index,"ethernet"]=c["col3"]
temp_df.loc[d.index,"ethernet"]=d["col4"]
temp_df.loc[e.index,"ethernet"]=e["col5"]


# In[472]:


a=temp_df[temp_df["col1"].str.contains("Multi")]
b=temp_df[temp_df["col2"].str.contains("Multi")]
c=temp_df[temp_df["col3"].str.contains("Multi")]
d=temp_df[temp_df["col4"].str.contains("Multi")]
e=temp_df[temp_df["col5"].str.contains("Multi")]
temp_df["multi_card_reader"]=a["col1"]
temp_df.loc[b.index,"multi_card_reader"]=b["col2"]
temp_df.loc[c.index,"multi_card_reader"]=c["col3"]
temp_df.loc[d.index,"multi_card_reader"]=d["col4"]
temp_df.loc[e.index,"multi_card_reader"]=e["col5"]


# In[475]:


a=temp_df[temp_df["col1"].str.contains("VGA")]
b=temp_df[temp_df["col2"].str.contains("VGA")]
c=temp_df[temp_df["col3"].str.contains("VGA")]
d=temp_df[temp_df["col4"].str.contains("VGA")]
e=temp_df[temp_df["col5"].str.contains("VGA")]
temp_df["vga"]=a["col1"]
temp_df.loc[b.index,"vga"]=b["col2"]
temp_df.loc[c.index,"vga"]=c["col3"]
temp_df.loc[d.index,"vga"]=d["col4"]
temp_df.loc[e.index,"vga"]=e["col5"]


# In[473]:


a=temp_df[temp_df["col1"].str.contains("Thunderbolt")]
b=temp_df[temp_df["col2"].str.contains("Thunderbolt")]
c=temp_df[temp_df["col3"].str.contains("Thunderbolt")]
d=temp_df[temp_df["col4"].str.contains("Thunderbolt")]
e=temp_df[temp_df["col5"].str.contains("Thunderbolt")]
temp_df["thunderbolt"]=a["col1"]
temp_df.loc[b.index,"thunderbolt"]=b["col2"]
temp_df.loc[c.index,"thunderbolt"]=c["col3"]
temp_df.loc[d.index,"thunderbolt"]=d["col4"]
temp_df.loc[e.index,"thunderbolt"]=e["col5"]


# In[474]:


a=temp_df[temp_df["col1"].str.contains("Display")]
b=temp_df[temp_df["col2"].str.contains("Display")]
c=temp_df[temp_df["col3"].str.contains("Display")]
d=temp_df[temp_df["col4"].str.contains("Display")]
e=temp_df[temp_df["col5"].str.contains("Display")]
temp_df["display_port"]=a["col1"]
temp_df.loc[b.index,"display_port"]=b["col2"]
temp_df.loc[c.index,"display_port"]=c["col3"]
temp_df.loc[d.index,"display_port"]=d["col4"]
temp_df.loc[e.index,"display_port"]=e["col5"]


# In[484]:


df=temp_df


# ## Wifi

# In[487]:


temp_df=df.copy()


# In[492]:


temp_df["wifi"].value_counts()
temp_df["wifi"].isnull().sum()
# only three laptops which dont have wifi so it is a useless column
df.drop("wifi",axis=1,inplace=True)


# ## USB

# In[618]:


temp_df=df.copy()


# In[619]:


temp_df["usb"]=temp_df["usb"].str.split(",")


# In[620]:


col1=[]
col2=[]
col3=[]
temp_df["usb"].fillna("False",inplace=True)
for i in temp_df["usb"]:
    if i is not False:
        if len(i)==3:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(i[2])
        elif len(i)==2:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(np.nan)
        elif len(i)==1:
            col1.append(i[0])
            col2.append(np.nan)
            col3.append(np.nan)
        else:
            col1.append(np.nan)
            col2.append(np.nan)
            col3.append(np.nan)
    else:
        col1.append(np.nan)
        col2.append(np.nan)
        col3.append(np.nan)
temp_df["usb_2"]=col1
temp_df["usb_3"]=col2
temp_df["type_c"]=col3
    


# In[621]:


temp_df["usb_2"].fillna("False",inplace=True)
temp_df["usb_3"].fillna("False",inplace=True)
temp_df["type_c"].fillna("False",inplace=True)


# In[622]:


a=temp_df[temp_df["usb_2"].str.contains("USB 2.0")]
b=temp_df[temp_df["usb_3"].str.contains("USB 2.0")]
c=temp_df[temp_df["type_c"].str.contains("USB 2.0")]
temp_df["usb2"]=a["usb_2"]
temp_df.loc[b.index,"usb2"]=b["usb_3"]
temp_df.loc[c.index,"usb2"]=c["type_c"]


# In[623]:


a=temp_df[temp_df["usb_2"].str.contains("USB 3.0")]
b=temp_df[temp_df["usb_3"].str.contains("USB 3.0")]
c=temp_df[temp_df["type_c"].str.contains("USB 3.0")]
temp_df["usb3"]=a["usb_2"]
temp_df.loc[b.index,"usb3"]=b["usb_3"]
temp_df.loc[c.index,"usb3"]=c["type_c"]


# In[624]:


a=temp_df[temp_df["usb_2"].str.contains("Type-C")]
b=temp_df[temp_df["usb_3"].str.contains("Type-C")]
c=temp_df[temp_df["type_c"].str.contains("Type-C")]
temp_df["typec"]=a["usb_2"]
temp_df.loc[b.index,"typec"]=b["usb_3"]
temp_df.loc[c.index,"typec"]=c["type_c"]


# In[626]:


df[["usb2","usb3","typec"]]=temp_df[["usb2","usb3","typec"]]


# ## Camera

# In[539]:


df["camera"].isnull().sum()
# there are almost 983 null values
df.drop("camera",axis=1,inplace=True)


# ## others

# In[548]:


temp_df=df.copy()


# In[549]:


temp_df["others"]=temp_df["others"].str.split(",")


# In[551]:


temp_df["others"].fillna("False",inplace=True)


# In[560]:


col1=[]
col2=[]
col3=[]
for i in temp_df["others"]:
    if i is not False:
        if len(i)==3:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(i[2])
        elif len(i)==2:
            col1.append(i[0])
            col2.append(i[1])
            col3.append(np.nan)
        elif len(i)==1:
            col1.append(i[0])
            col2.append(np.nan)
            col3.append(np.nan)
        else:
            col1.append(np.nan)
            col2.append(np.nan)
            col3.append(np.nan)
    else:
        col1.append(np.nan)
        col2.append(np.nan)
        col3.append(np.nan)

temp_df["col11"]=col1
temp_df["col21"]=col2
temp_df["col31"]=col3

            


# In[566]:


temp_df["col11"].fillna("False",inplace=True)
temp_df["col21"].fillna("False",inplace=True)
temp_df["col31"].fillna("False",inplace=True)


# In[570]:


a=temp_df[temp_df["col11"].str.contains("Backlit")]
b=temp_df[temp_df["col21"].str.contains("Backlit")]
c=temp_df[temp_df["col31"].str.contains("Backlit")]
temp_df["backlit"]=a["col11"]
temp_df.loc[b.index,"backlit"]=b["col21"]
temp_df.loc[c.index,"backlit"]=c["col31"]


# In[571]:


a=temp_df[temp_df["col11"].str.contains("Sensor")]
b=temp_df[temp_df["col21"].str.contains("Sensor")]
c=temp_df[temp_df["col31"].str.contains("Sensor")]
temp_df["fingerprint_sensor"]=a["col11"]
temp_df.loc[b.index,"fingerprint_sensor"]=b["col21"]
temp_df.loc[c.index,"fingerprint_sensor"]=c["col31"]


# In[572]:


a=temp_df[temp_df["col11"].str.contains("Inbuilt")]
b=temp_df[temp_df["col21"].str.contains("Inbuilt")]
c=temp_df[temp_df["col31"].str.contains("Inbuilt")]
temp_df["inbuilt_microphone"]=a["col11"]
temp_df.loc[b.index,"inbuilt_microphone"]=b["col21"]
temp_df.loc[c.index,"inbuilt_microphone"]=c["col31"]


# In[576]:


df=temp_df


# In[607]:


df.columns


# In[590]:


df.drop(["col1","col2","col3","col4","col5","col11","col21","col31"],axis=1,inplace=True)


# In[608]:


df.drop(["battery1","battery","hdmi","others","usb"],axis=1,inplace=True)


# In[670]:


df.drop(["cores"],axis=1,inplace=True)


# In[668]:


# one value is different in aspect_ratio1 so we replace it with np.nan.
df["aspect_ratio1"] = np.where(df["aspect_ratio1"] == "250 nits, 141 ppi, Color Gamut: 45%NTSC Aspect Ratio", np.nan, df["aspect_ratio1"])    



# ## Processor

# In[681]:


temp_df=df.copy()


# In[726]:


# There is one column in which wrong brand is given
temp_df["processor"].replace("AMD Core i3 N305","Intel Core i3 N305",inplace=True)


# In[728]:


# Here i make a processor brand column and store all the processor_brands in it.
list1=[]
for i in temp_df["processor"]:
    if "Intel" in i or "intel" in i:
        list1.append("intel")
    elif "AMD" in i or "Amd" in i:
        list1.append("amd")
    elif "Apple" in i:
        list1.append("apple")
    elif "MediaTek" in i:
        list1.append("mediatek")
    else:
        list1.append(np.nan)
temp_df["processor_brand"]=list1


# In[739]:


# Here i make a processor_gen column and store all the generation in it.
list1=[]
for i in temp_df["processor"]:
    if "13th" in i:
        list1.append("13")
    elif "12th" in i:
        list1.append("12")
    elif "11th" in i:
        list1.append("11")
    elif "10th" in i:
        list1.append("10")
    elif "9th" in i:
        list1.append("9")
    elif "8th" in i:
        list1.append("8")
    elif "7th" in i:
        list1.append("7")
    elif "6th" in i:
        list1.append("6")
    elif "5th" in i:
        list1.append("5")
    elif "4th" in i:
        list1.append("4")
    elif "3rd" in i:
        list1.append("3")
    else:
        list1.append(np.nan)
temp_df["processor_gen"]=list1


# In[740]:


# Here i make a processor_model column and store all the processor model in it.
list1=[]
for i in temp_df["processor"]:
    if "i5" in i:
        list1.append("i5")
    elif "i3" in i:
        list1.append("i3")
    elif "i7" in i:
        list1.append("i7")
    elif "i9" in i:
        list1.append("i9")
    elif "M1" in i:
        list1.append("M1")
    elif "M2" in i:
        list1.append("M2")
    elif "3" in i:
        list1.append("3")
    elif "5" in i:
        list1.append("5")
    elif "7" in i or "Ryzen 7040":
        list1.append("7")
    elif "9" in i:
        list1.append("9")
    elif "Athlon" in i:
        list1.append("athlon")
    elif "Celeron" in i:
        list1.append("celeron")
    elif "Pentium" in i:
        list1.append("pentium")
    else:
        list1.append(np.nan)
temp_df["processor_model"]=list1


# In[741]:


df=temp_df


# In[743]:


df.drop("processor",axis=1,inplace=True)


# ## Graphic Card

# In[752]:


df["graphic_card"].isnull().sum()
temp_df=df.copy()


# In[757]:


list1=[]
for i in temp_df["graphic_card"]:
    if "Intel" in i or "Iris" in i or "UHD" in i:
        list1.append("intel")
    elif "NVIDIA" in i or "Nvidia" in i or "Geforce" in i:
        list1.append("nvidia")
    elif "AMD" in i or "Radeon" in i :
        list1.append("amd")
    elif "ARM" in i:
        list1.append("arm")
    elif "Apple" in i or "8 Core" in i or "10 Core" in i or "16 Core" in i or "38 Core" in i or "8-Core" in i or "10-Core" in i or "16-core" in i or "38-core" in i:
        list1.append("apple")
    else:
        list1.append(np.nan)
temp_df["graphics_brand"]=list1
        


# In[1058]:


temp_df["graphics_capacity"]=temp_df["graphic_card"].str.split().str.get(0).str.strip()


# In[1069]:


temp_df["graphics_capacity"]=temp_df["graphics_capacity"].apply(lambda x: x if x in ['2','4','6','8','10','12','16'] else np.nan)


# In[1070]:


df["graphics_capacity"]=temp_df["graphics_capacity"]


# In[766]:


list1=[]
for i in temp_df["graphic_card"]:
    if "RTX 2050" in i or "RTX2050" in i:
        list1.append("rtx2050")
    elif "RTX 2060" in i or "RTX2060" in i:
        list1.append("rtx2060")
    elif "RTX 3050" in i or "RTX3050" in i:
        list1.append("rtx3050")
    elif "RTX 3060" in i or "RTX3060" in i:
        list1.append("rtx3060")
    elif "RTX 4050" in i or "RTX4050" in i:
        list1.append("rtx4050")
    elif "RTX 4060" in i or "RTX4060" in i:
        list1.append("rtx4060")
    elif "RTX 4070" in i or "RTX4070" in i:
        list1.append("rtx4070")
    elif "RTX 4080" in i or "RTX4080" in i:
        list1.append("rtx4080")
    elif "RTX 4090" in i or "RTX4090" in i:
        list1.append("rtx4090")
    elif "RTX 3070 Ti" in i or "RTX3070 Ti" in i:
        list1.append("rtx3070ti")
    elif "RTX 3060 Ti" in i or "RTX3060 Ti" in i:
        list1.append("rtx3060ti")
    elif "RTX 3080 Ti" in i or "RTX3080 Ti" in i:
        list1.append("rtx3080ti")
    elif "GTX 1650" in i or "GTX1650" in i:
        list1.append("gtx1650")
    elif "GTX 2050" in i or "GTX 2050" in i:
        list1.append("gtx2050")
    elif "T500" in i or "T 500" in i:
        list1.append("t500")
    elif "T 550" in i or "T550" in i:
        list1.append("t550")
    elif "T 600" in i or "T600" in i:
        list1.append("t600")
    elif "MX130" in i or "MX 130" in i:
        list1.append("mx130")
    elif "MX450" in i or "MX 450" in i:
        list1.append("mx450")
    elif "MX 550" in i or "MX550" in i:
        list1.append("mx550")
    elif "MX 570" in i or "MX570" in i:
        list1.append("mx570")
    elif "RX 5500M" in i or "RX5500M" in i:
        list1.append("rx5500m")
    elif "RX 5600M" in i or "RX5600M" in i:
        list1.append("rx5600m")
    elif "RX 6500M" in i or "RX6500M" in i:
        list1.append("rx6500m")
    elif "RX 7600S" in i or "RX7600S" in i:
        list1.append("rx7600s")
    else:
        list1.append("Integrated")
    
temp_df["graphics_model"]=list1
        


# In[768]:


df=temp_df




# In[784]:


temp_df=df.copy()


# In[790]:


temp_df["utility"]=temp_df["utility"].str.replace("Utility:","").str.strip().str.split(",")


# In[798]:


temp_df["utility"].fillna("False",inplace=True)


# In[809]:


temp_df['everyday_use'] = temp_df['utility'].apply(lambda x: 1 if 'Everyday Use' in x else 0)
temp_df['business'] = temp_df['utility'].apply(lambda x: 1 if 'Business' in x else 0)
temp_df['performance'] = temp_df['utility'].apply(lambda x: 1 if 'Performance' in x else 0)
temp_df['gaming'] = temp_df['utility'].apply(lambda x: 1 if 'Gaming' in x else 0)
        
    



# In[813]:


df=temp_df


# In[815]:


df.drop("utility",axis=1,inplace=True)



# In[821]:


temp_df=df.copy()


# In[824]:


temp_df["thickness"].fillna("False",inplace=True)


# In[834]:


temp_df["thickness"]=temp_df["thickness"].str.replace("Thickness:","").str.strip().str.split().str.get(0).str.replace("False","0")


# In[836]:


df=temp_df





temp_df=df.copy()


# In[871]:


temp_df["weight"]=temp_df["weight"].str.split().str.get(0).str.strip().astype("float")


# In[878]:


temp_df["weight"]=temp_df["weight"].apply(lambda x: x if x<10 else x/1000)


# In[879]:


df=temp_df



# In[889]:


df["warranty"]=df["warranty"].str.split().str.get(0)


# In[901]:


df["warranty"]=df["warranty"].astype("float")


# **screen size**



# In[902]:


df["screen_size"]=df["screen_size"].str.split().str.get(0).astype("float")


# **Resolution**



# In[1081]:


df["resolution_width"]=df["resolution"].str.split().str.get(0).str.strip().astype("float")
df["resolution_height"]=df["resolution"].str.split().str.get(2).str.strip().astype("float")


# In[1082]:


df.drop("resolution",axis=1,inplace=True)


# **PPI**



# In[914]:


df["ppi"]=df["ppi"].str.split().str.get(1)


# In[918]:


df["ppi"]=df["ppi"].astype("int")


# **Threads**


# In[921]:


df["threads"]=df["threads"].str.split().str.get(0).str.strip()


# In[926]:


df["threads"]=df["threads"].astype("float")


# In[928]:


df.drop("graphic_card",axis=1,inplace=True)


# **RAM**



# In[938]:


df["ram"]=df["ram"].str.split().str.get(0).astype("int")


# **Hard-Disk**


# In[1168]:


df["hard_disk"].fillna("False",inplace=True)


# In[1177]:


df["hdd"]=df["hard_disk"].str.split(",").str.get(0).str.strip()


# In[1178]:


df["ssd"]=df["hard_disk"].str.split(",").str.get(1).str.strip()


# In[1180]:


temp_df["ssd"]=df["hdd"].apply(lambda x: x if "SSD" in x else np.nan )


# In[1182]:


df.loc[temp_df["ssd"].index,"ssd"]=temp_df["ssd"]


# In[1186]:


df["ssd"]=df["ssd"].str.split().str.get(0).astype("float")


# In[1188]:


df["ssd"].fillna(0,inplace=True)


# In[1190]:


df["ssd"]=df["ssd"].apply(lambda x: x*1024 if x<10 else x)


# In[1193]:


df["hdd"]=df["hdd"].apply(lambda x: x if "HARD" in x else 0)


# In[1195]:


df["hdd"]=df["hdd"].str.split().str.get(0).str.strip().astype("float")


# In[1196]:


df["hdd"]=df["hdd"].apply(lambda x: x*1024 if x<10 else x)


# **Antiglare**


# In[944]:


df["antiglare1"].fillna("False",inplace=True)


# In[947]:


df["antiglare1"]=df["antiglare1"].apply(lambda x: 1 if "Anti Glare" in x else 0)


# **Touchscreen**



# In[952]:


df["touch_screen1"].fillna("False",inplace=True)


# In[953]:


df["touch_screen1"]=df["touch_screen1"].apply(lambda x: 1 if "Touch Screen" in x else 0).astype("int")


# **Cores**


# In[955]:


df["cores1"].fillna("False",inplace=True)


# In[958]:


df["cores1"]=df["cores1"].str.split().str.get(0)


# In[963]:


def core(x):
    if x=="Hexa":
        return 6
    elif x=="Octa":
        return 8
    elif x=="Quad":
        return 4
    elif x=="Dual":
        return 2
    elif x=="False":
        return np.nan
    else:
        return x
df["cores1"]=df["cores1"].apply(core).astype("float")


# **Battery Capacity**



# In[969]:


df["battery_capacity"]=df["battery_capacity"].str.split().str.get(0)


# In[970]:


df["battery_capacity"]=df["battery_capacity"].astype("float")


# **Battery Cell**



# In[974]:


df["battery_cell"]=df["battery_cell"].str.split().str.get(0).astype("float")


# **HDMI**


# In[976]:


df["hdmi1"].fillna("False",inplace=True)


# In[979]:


df["hdmi1"]=df["hdmi1"].str.strip().apply(lambda x: 1 if "HDMI" in x else 0).astype("int")


# **Ethernet**


# In[982]:


df["ethernet"].fillna("False",inplace=True)


# In[986]:


df["ethernet"]=df["ethernet"].str.split().str.get(0).apply(lambda x: 1 if "Ethernet" in x else 0).astype("int")


# **Multi Card Reader**



# In[988]:


df["multi_card_reader"].fillna("False",inplace=True)


# In[989]:


df["multi_card_reader"]=df["multi_card_reader"].str.strip().apply(lambda x: 1 if "Multi Card Reader" in x else 0).astype("int")


# **Thunderbolt**


# In[992]:


df["thunderbolt"].fillna("False",inplace=True)


# In[995]:


df["thunderbolt"]=df["thunderbolt"].str.strip().apply(lambda x: 1 if "Thunderbolt" in x else 0).astype("int")


# **Display Port**


# In[998]:


df["display_port"].fillna("False",inplace=True)


# In[999]:


df["display_port"]=df["display_port"].str.strip().apply(lambda x: 1 if "Display Port" in x else 0).astype("int")


# **VGA**



# In[1002]:


df["vga"].fillna("False",inplace=True)


# In[1003]:


df["vga"]=df["vga"].str.strip().apply(lambda x: 1 if "VGA" in x else 0).astype("int")


# **Back Light**



# In[1005]:


df["backlit"].fillna("False",inplace=True)


# In[1006]:


df["backlit"]=df["backlit"].str.strip().apply(lambda x: 1 if "Backlit Keyboard" in x else 0).astype("int")


# **Fingerprint Sensor**



# In[1010]:


df["fingerprint_sensor"].fillna("False",inplace=True)


# In[1011]:


df["fingerprint_sensor"]=df["fingerprint_sensor"].str.strip().apply(lambda x: 1 if "Fingerprint Sensor" in x else 0).astype("int")


# **Inbuilt Microphone**



# In[1013]:


df.fillna({"inbuilt_microphone":"False"},inplace=True)


# In[1015]:


df["inbuilt_microphone"]=df["inbuilt_microphone"].str.strip().apply(lambda x: 1 if "Inbuilt Microphone" in x else 0).astype("int")


# **USB 2.0**



# In[1019]:


df["usb2"]=df["usb2"].str.split("x").str.get(0).astype("float")


# **USB 3.0**


# In[1025]:


df["usb3"]=df["usb3"].str.split("x").str.get(0).str.strip().astype("float")


# **Type-C**


# In[1027]:


df["typec"]=df["typec"].str.split("x").str.get(0).str.strip().astype("float")


# **Processor Gen**


# In[1029]:


df["processor_gen"]=df["processor_gen"].astype("float")





# **Graphic Capacity**


# In[1072]:


df["graphics_capacity"]=df["graphics_capacity"].astype("float")


# **Graphic Model**


# In[1205]:


df["thickness"]=df["thickness"].astype("float")


# In[1207]:


df["num_votes"]=df["num_votes"].astype("float")


# In[1210]:


#i think we should add one more column of "laptop_model"
temp_df=pd.read_csv("data/raw/raw_data.csv",index_col=0)


# In[1213]:


df.loc[temp_df["name"].index,"laptop_model"]=temp_df["name"]


# In[7]:


df.rename(columns={"Brand":"brand","Ratings":"ratings","antiglare1":"antiglare","aspect_ratio1":"aspect_ratio","touch_screen1":"touch_screen","cores1":"cores","hdmi1":"hdmi"},inplace=True)


# In[11]:


df=df.loc[:,['laptop_model','brand','price', 'num_votes', 'ratings', 'thickness', 'weight',
       'warranty', 'screen_size','resolution_width', 'resolution_height', 'ppi', 'threads', 'ram', 'antiglare',
       'aspect_ratio', 'touch_screen', 'cores', 'battery_capacity',
       'battery_cell', 'hdmi', 'ethernet', 'multi_card_reader', 'thunderbolt',
       'display_port', 'vga', 'backlit', 'fingerprint_sensor',
       'inbuilt_microphone', 'usb2', 'usb3', 'typec', 'processor_gen',
       'processor_brand', 'processor_model', 'graphics_brand',
       'graphics_capacity', 'graphics_model', 'everyday_use', 'business',
       'performance', 'gaming', 'hdd',
       'ssd']]




# In[35]:


df["aspect_ratio"]=df["aspect_ratio"].str.split().str[0].str.replace("16:09","16:9").str.strip()


# In[69]:


df.fillna({"aspect_ratio":"0"},inplace=True)
selected_rows=df[df["aspect_ratio"].str.match(r'16$|16:$')]


# In[70]:


df.loc[selected_rows.index,"aspect_ratio"]="16:9"


df.drop_duplicates(inplace=True)

#On the basis of histogram i think we can divide this in 3 categories
#0->regular[0-199]
#1->popular[199-2000]
#2->viral[2000<]
# Define the thresholds
regular_threshold = 200
popular_threshold = 2000

# Create a new column to store the popularity level
df['popularity'] = pd.cut(df['num_votes'], bins=[-1, regular_threshold, popular_threshold, float('inf')], labels=['regular', 'popular', 'viral'])


scaler=StandardScaler()
scaled_data=scaler.fit_transform(df[["ratings"]])
#there are multiple breaks in the slope so there is no clear cut definition of how many categories should be formed but i still try to divide it in 3 categories.
n_clusters = 3

# Fit the KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data)

# Predict the cluster assignments for each row
cluster_assignments = kmeans.predict(scaled_data)
df["quality_type"]=cluster_assignments
category_mapping={0:"medium",1:"low",2:"high"}
df["quality_type"]=df["quality_type"].replace(category_mapping)
df.sample(5)[["ratings","quality_type"]]


df["thickness"]=df["thickness"].replace(0,np.nan)
#Here we convert some values that i thought are in inches and given into mm.
df.loc[df["thickness"] < 9, "thickness"] *= 25.4
#On the basis of histogram i think we can divide this in 3 categories
#0->slim[<18]
#1->medium[18-22]
#2->thick[>22]
# Define the thresholds
slim_threshold = 18
medium_threshold = 22

# Create a new column to store the popularity level
df['thickness'] = pd.cut(df['thickness'], bins=[-1, slim_threshold, medium_threshold, float('inf')], labels=['slim', 'medium', 'thick'])


#On the basis of histogram i think we can divide this in 3 categories
#0->lite[<1.5]
#1->medium[1.5-2]
#2->heavy[>2]
# Define the thresholds
lite_threshold = 1.5
medium_threshold = 2

# Create a new column to store the popularity level
df['weight'] = pd.cut(df['weight'], bins=[-1, lite_threshold, medium_threshold, float('inf')], labels=['lite', 'medium', 'heavy'])



#Here we see that k-means clustering doesn't give us appropriate results so by looking data we decide to divide this into three category
#100-140->low
#140-144->medium
#more than 144 ->high
low_threshold=139
medium_threshold=145
df["ppi_type"]=pd.cut(df["ppi"],bins=[-1,low_threshold,medium_threshold,float("inf")],labels=["low","medium","high"])


df["aspect_ratio"]=df["aspect_ratio"].replace("0",np.nan)


#usb2 , usb3 and typec have null values now we are going to fill them with 0. which shows that these features are not present in that laptop
df["usb2"]=df["usb2"].replace(np.nan,0)
df["usb3"]=df["usb3"].replace(np.nan,0)
df["typec"]=df["typec"].replace(np.nan,0)




file_path="data/processed/processed_dataset.csv"

if os.path.exists(file_path):
    os.remove(file_path)

df.to_csv(file_path,index=False)








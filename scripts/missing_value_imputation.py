import pandas as pd
import numpy as np 
import os

df1=pd.read_csv("data/processed/processed_dataset.csv")
df=df1.copy()

df.drop(["laptop_model","num_votes","ratings","warranty","resolution_width","resolution_height","aspect_ratio","inbuilt_microphone"],axis=1,inplace=True)
df["battery_capacity"]=df["battery_capacity"].replace(6,np.nan)
num_cols=["price","screen_size","ppi","threads","ram","cores","battery_capacity","battery_cell","graphics_capacity","hdd","ssd"]
for col in num_cols:
    if df[col].dtype=="int64":
        df[col]=df[col].astype("int32")
    elif df[col].dtype=="float64":
        df[col]=df[col].astype("float32")
    else:
        df[col]=df[col].astype("float32")




#price of 1024gb hdd is equal to 512gb ssd
df["hdd"]=df["hdd"].replace(1024,512)


#here i replace o in ssd with the corresponding values in hdd 
df.loc[df['ssd'] == 0, 'ssd'] = df.loc[df['ssd'] == 0, 'hdd']




for i in df[df["thickness"].isnull()].index:
    if df.loc[i,"weight"]=="NaN":
        continue
    else:
        if df.loc[i,"weight"]=="heavy":
            df.loc[i,"thickness"]="thick"
        elif df.loc[i,"weight"]=="lite":
            df.loc[i,"thickness"]="slim"
        elif df.loc[i,"weight"]=="medium":
            df.loc[i,"thickness"]="medium"
            
        




df.loc[(df["weight"].isnull()) & ~(df["graphics_model"]=="Integrated"),"weight"]="heavy"



df.loc[(df["thickness"].notna()) & (df["weight"].isnull()),"weight"]="lite"



df.loc[(df["thickness"].isnull()) & (df["weight"].notna()),"thickness"]="thick"






#we are going to put thickness and weight on the basis of price
#0-40000->slim,lite
#40000-70000->medium,medium
#70000->thick,heavy
rows=df[(df["thickness"].isnull()) & (df["weight"].isnull())]
for i in rows.index:
    if rows["price"][i]<=40000:
        df.loc[i,"thickness"]="slim"
        df.loc[i,"weight"]="lite"
    elif 70000>rows["price"][i]>40000:
        df.loc[i,"thickness"]="medium"
        df.loc[i,"weight"]="medium"
    else:
        df.loc[i,"thickness"]="thick"
        df.loc[i,"weight"]="heavy"
        


# # 3.Threads


df.loc[(df["threads"].isnull())&(df["brand"]=="Apple"),"threads"]=8.0


#by seeing the heatmap we found out that
#if 3,5,7->12
#i3->8
#i5->12
#i7->16
#i9->20
rows=df[df["threads"].isnull()]
for row in rows.index:
    if df["processor_model"][row]=="i3":
        df.loc[row,"threads"]=8
    elif df["processor_model"][row]=="i7":
        df.loc[row,"threads"]=16
    elif df["processor_model"][row]=="i9":
        df.loc[row,"threads"]=20
    else:
        df.loc[row,"threads"]=12
    






rows=df[df["cores"].isnull()]
#by seeing price i decide that
df.loc[rows.index[0],"cores"]=2
df.loc[rows.index[1],"cores"]=12
df.loc[rows.index[2],"cores"]=10
df.loc[rows.index[3],"cores"]=6



rows=df[(df["battery_capacity"].isnull()) & (df["battery_cell"].notna())]
for row in rows.index:
    if df["battery_cell"][row]==2:
        df.loc[row,"battery_capacity"]=38
    elif df["battery_cell"][row]==3:
        df.loc[row,"battery_capacity"]=50
    elif df["battery_cell"][row]==4:
        df.loc[row,"battery_capacity"]=72
    elif df["battery_cell"][row]==6:
        df.loc[row,"battery_capacity"]=86
        


rows=df[df["battery_cell"].isnull()]
for row in rows.index:
    if df["weight"][row]=="lite":
        df.loc[row,"battery_cell"]=2.0
    elif df["weight"][row]=="medium":
        if 50000<=df["price"][row]<=100000:
            df.loc[row,"battery_cell"]=3.0
        elif 50000>df["price"][row]:
            df.loc[row,"battery_cell"]=2.0
        else:
            df.loc[row,"battery_cell"]=4.0
    elif df["weight"][row]=="heavy":
        df.loc[row,"battery_cell"]=6.0
        
        
        
rows=df[(df["battery_capacity"].isnull()) & (df["battery_cell"].notna())]
for row in rows.index:
    if df["battery_cell"][row]==2:
        df.loc[row,"battery_capacity"]=38
    elif df["battery_cell"][row]==3:
        df.loc[row,"battery_capacity"]=50
    elif df["battery_cell"][row]==4:
        df.loc[row,"battery_capacity"]=72
    elif df["battery_cell"][row]==6:
        df.loc[row,"battery_capacity"]=86


#7->7th
#5->5th
#i3->12th
#m1,m2->1,2
#3->7
#i7->13th
rows=df[df["processor_gen"].isnull()]
for row in rows.index:
    if df["processor_model"][row]=="7" or df["processor_model"][row]=="3" :
        df.loc[row,"processor_gen"]=7
    elif df["processor_model"][row]=="5":
        df.loc[row,"processor_gen"]=5
    elif df["processor_model"][row]=="i3":
        df.loc[row,"processor_gen"]=12
    elif df["processor_model"][row]=="M1":
        df.loc[row,"processor_gen"]=1
    elif df["processor_model"][row]=="M2":
        df.loc[row,"processor_gen"]=2
    elif df["processor_model"][row]=="i7":
        df.loc[row,"processor_gen"]=13
    



#so we are going to put 0 where graphics_capacity is integrated
df.loc[df[(df["graphics_capacity"].isnull())&(df["graphics_model"]=="Integrated")].index,"graphics_capacity"]=0


#df.drop(798,axis=0,inplace=True)

df.loc[200,"graphics_brand"]="intel"


df.loc[(df["graphics_capacity"].isnull())&(df["graphics_model"]!="Integrated"),"graphics_capacity"]=16


file_path="data/processed/after_missing_value_imputation.csv"

if os.path.exists(file_path):
    os.remove(file_path)

df.to_csv(file_path,index=False)



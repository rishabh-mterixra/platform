# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:20:22 2023

@author: Marketpulse
"""

import pandas as pd
import numpy as np
import re
import os
import polars as pl
from functools import partial
import streamlit as st


mapping=pd.read_csv("aux files//mapping csv2.csv")
all_csv=os.listdir("CSV Files/")


#%%

# UDF

def period_dfs_creation(dataframe):
    period_dfs_list=[]
    for f in selected_freq:
        if f=='Monthly':
            mnth_df=dataframe[dataframe['MONTH_YEAR']==month_year]
            period_dfs_list.append(mnth_df)
            
        elif f=='Quarterly':
            if date_max.month>2:
                qtr=date_max.month//3
                year=date_max.year
            else:
                qtr=4
                year=date_max.year-1    
            qtr_df=dataframe[(dataframe['DATE']>=pd.Timestamp(day=1,month=3*qtr-2,year=year)) & 
                             (dataframe['DATE']<=pd.Timestamp(day=1,month=3*qtr,year=year))]
            period_dfs_list.append(qtr_df)
            #print(qtr,year)
        elif f=='Financial Year':
            fin_year_df=dataframe[(pd.Timestamp(day=31,month=3,year=fin_year)<dataframe['DATE']) &
                              (dataframe['DATE']<pd.Timestamp(day=2,month=date_max.month,year=date_max.year))]
            period_dfs_list.append(fin_year_df)
       
        elif f=='Calendar Year':
            cal_year_df=dataframe[(pd.Timestamp(day=31,month=12,year=date_max.year-1)<dataframe['DATE']) &
                              (dataframe['DATE']<pd.Timestamp(day=2,month=date_max.month,year=date_max.year))]
            period_dfs_list.append(cal_year_df)
        
        elif f=='Rolling Year':
            rolling_year_df=dataframe[(pd.Timestamp(day=2,month=date_max.month,year=date_max.year-1)<dataframe['DATE']) &
                            (dataframe['DATE']<pd.Timestamp(day=2,month=date_max.month,year=date_max.year))]
            period_dfs_list.append(rolling_year_df)
            
        elif f=='Financial Half-Year':
            if date_max.month>9:
                fin_halfyear_df=dataframe[(dataframe['DATE']>pd.Timestamp(day=31,month=3,year=date_max.year)) & 
                                          (dataframe['DATE']<pd.Timestamp(day=2,month=9,year=date_max.year))]
            elif date_max.month<4:
                fin_halfyear_df=dataframe[(dataframe['DATE']>pd.Timestamp(day=31,month=3,year=date_max.year-1)) & 
                                          (dataframe['DATE']<pd.Timestamp(day=2,month=9,year=date_max.year-1))]
            else:
                fin_halfyear_df=dataframe[(dataframe['DATE']>pd.Timestamp(day=30,month=9,year=date_max.year-1)) & 
                                          (dataframe['DATE']<pd.Timestamp(day=1,month=4,year=date_max.year))]
            period_dfs_list.append(fin_halfyear_df)
            
        elif f=='Calendar Half-Year':
            if date_max.month>6:
                cal_halfyear_df=dataframe[(dataframe['DATE']>pd.Timestamp(day=31,month=12,year=date_max.year-1)) & 
                                          (dataframe['DATE']<pd.Timestamp(day=2,month=6,year=date_max.year))]
            else:
                cal_halfyear_df=dataframe[(dataframe['DATE']>=pd.Timestamp(day=1,month=7,year=date_max.year-1)) & 
                                          (dataframe['DATE']<pd.Timestamp(day=2,month=12,year=date_max.year-1))]
            period_dfs_list.append(cal_halfyear_df)
                
    return period_dfs_list


def market_size_calc(dataframe):
    market_size_dfs=[]
    for g in selected_geography:
        if model!=None:
            sub_dataframe=dataframe[dataframe['MODELS']==model]
            model_size_df=pd.pivot_table(data=sub_dataframe,index=[g,'MODELS'],values=['SALES UNITS','SALES VALUE INR'],
                               aggfunc='sum')
            market_size_dfs.append(model_size_df)
        elif specification!=None:
            spec_col=specifications_df[specifications_df['Specifications']==specification]['Column name'].values[0]
            sub_dataframe=dataframe[dataframe[spec_col]==specification]
            spec_size_df=pd.pivot_table(data=sub_dataframe,index=[g,spec_col],values=['SALES UNITS','SALES VALUE INR'],
                               aggfunc='sum')
            market_size_dfs.append(spec_size_df)
            
        else:
            sub_dataframe=dataframe[dataframe['MODELS']=='All Models']
            brands_size_df=pd.pivot_table(data=sub_dataframe,index=[g,'BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                           aggfunc='sum')
            total_df=dataframe[dataframe['BRANDS']=='Total']
            total_brand_size=pd.pivot_table(data=total_df,index=[g,'BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                           aggfunc='sum')
            combined_df=pd.concat([total_brand_size,brands_size_df],axis=0)
            market_size_dfs.append(combined_df)
    
    return market_size_dfs



def share_nd_wd_calc(dataframe):
    market_metrics_dfs=[]
    total_vals=dataframe[dataframe['BRANDS']=='Total'][['SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION']].sum()
    if model==None:
        models_df=dataframe[dataframe['MODELS']=='All Models']
    else:
        models_df=dataframe[dataframe['MODELS']==model]
    for g in selected_geography:
        nd_wd_df=pd.pivot_table(data=models_df,index=[g,'BRANDS'],
            values=['SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION'],
            aggfunc='sum')
        nd_wd_df['SALES UNITS']=nd_wd_df['SALES UNITS'].apply(lambda z:(z/total_vals['SALES UNITS'])*100)        
        nd_wd_df['SALES VALUE INR']=nd_wd_df['SALES VALUE INR'].apply(lambda z:(z/total_vals['SALES VALUE INR'])*100)        
        nd_wd_df['NUMERIC DISTRIBUTION']=nd_wd_df['NUMERIC DISTRIBUTION'].apply(lambda z:(z/total_vals['NUMERIC DISTRIBUTION'])*100)        
        nd_wd_df['WEIGHTED DISTRIBUTION']=nd_wd_df['WEIGHTED DISTRIBUTION'].apply(lambda z:(z/total_vals['WEIGHTED DISTRIBUTION'])*100)        
        market_metrics_dfs.append(nd_wd_df)        
            
    return market_metrics_dfs




def asp_calc(dataframe):
    asp_dfs=[]
    if model!=None:
        dataframe=dataframe[dataframe['MODELS']==model]
    for g in selected_geography:
        asp_brands=pd.pivot_table(data=dataframe,index=[g,'BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                           aggfunc='sum')
        asp_brands['ASP']=asp_brands[['SALES UNITS','SALES VALUE INR']].apply(lambda z:z[1]/z[0],axis=1)
        asp_dfs.append(asp_brands)
        
    return asp_dfs    



def model_contribution(dataframe):
    model_df=dataframe[dataframe['MODELS']==model]
    try:
        brand=model_df['BRANDS'].unique()[0]
        brand_df=dataframe[dataframe['BRANDS']==brand]
        contri_dfs=[]
        for g in selected_geography:
            model_agg=pd.pivot_table(data=brand_df,index=[g,'MODELS'],values=['SALES UNITS','SALES VALUE INR'],
                           aggfunc='sum').reset_index()
            brand_df=brand_df[brand_df['MODELS']=='All Models']
            brand_agg=pd.pivot_table(data=brand_df,index=[g,'BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                               aggfunc='sum').reset_index()
            brand_agg.rename(columns={'SALES UNITS':'Total SALES UNITS',
                                      'SALES VALUE INR':'Total SALES VALUE INR'},inplace=True)      
            agg_df=model_agg.merge(brand_agg)
            agg_df['%Vol Contribution']=agg_df[['SALES UNITS','Total SALES UNITS']].apply(lambda z:(z[0]/z[1])*100,axis=1)
            agg_df['%Val Contribution']=agg_df[['SALES VALUE INR','Total SALES VALUE INR']].apply(lambda z:(z[0]/z[1])*100,axis=1)
            agg_df.drop(columns=['SALES UNITS', 'SALES VALUE INR', 'BRANDS',
                   'Total SALES UNITS', 'Total SALES VALUE INR'],inplace=True)
            agg_df=agg_df[agg_df['MODELS']==model]
            contri_dfs.append(agg_df)
        return contri_dfs    
    
    except:
        print("Model not sold for the period!",) 



#%%

# User Inputs:
product=st.selectbox("Select Product",mapping['SUB CATEGORY'].unique()) # To be selected from drop-down list # Category/Sub-category

model_flag=st.checkbox("Model Level summary")
model_options=mapping[mapping['SUB CATEGORY']==product]['MODELS'].unique()
if model_flag:
    model=st.selectbox("Select Model",model_options)
else:    
    model=None

freq_type=["Monthly",'Quarterly','Financial Year','Calendar Year',
           'Rolling Year','Financial Half-Year','Calendar Half-Year']    
selected_freq=st.multiselect("Select the period types",freq_type,default=freq_type[0])

comparison_type=[]
#last_date=""

geography_level=["COUNTRY",'STATES','ZONE','AREA'] 
selected_geography=st.multiselect("Select the Geography Level",geography_level,default=geography_level[0])

################ Output = ? #############

#%%

# Getting & reading the relevant CSV file
csv_name=mapping[mapping['SUB CATEGORY']==product]['CSV Files'].unique()[0]
pl_df=pl.read_csv("CSV Files//"+csv_name)
df=pl_df.to_pandas()

# Subsetting dataframe for the particular product:
df=df[df['SUB CATEGORY']==product]

# Subsetting dataframe for the defined date range:
df['DATE']=pd.to_datetime(df['PERIOD'])
df['MONTH_YEAR']=df['DATE'].apply(lambda z:str(z.month)+"_"+str(z.year))
#df=df[df['DATE']>=pd.Timestamp(last_date)]    

# Removing Modern Trade from df:
df=df[df['CHANNEL']!='Modern Trade']
    
# Option to select Specification:
specifications_df=pd.DataFrame(columns=['Column name','Specifications'])
for j in df.columns:
    if j.startswith('SPECIFICATION'):
        temp_df=pd.DataFrame(df[j].unique(),columns=['Specifications'])
        temp_df['Column name']=j
        specifications_df=pd.concat([specifications_df,temp_df],axis=0)
specifications_df['Specifications']=specifications_df['Specifications'].astype('str')
specifications_df=specifications_df[specifications_df['Specifications']!='None']
specification_flag=st.checkbox("Specification Level summary")
if specification_flag and model_flag==False:
    specification=st.selectbox("Select Specification",specifications_df['Specifications'])
else:
    specification=None

date_max=df['DATE'].max()
#date_max=pd.Timestamp(day=1,month=2,year=2023)
month_year=str(date_max.month)+"_"+str(date_max.year)
fin_year=[date_max.year-1 if date_max.month<4 else date_max.year][0]
st.write(date_max)


#%%

all_period_dfs=period_dfs_creation(df)
for d,q in zip(all_period_dfs,selected_freq):
    st.write(q,d['DATE'].min(),d['DATE'].max())
st.header(" ")


market_size_dfs=map(market_size_calc,all_period_dfs)
market_size_lst=[]
for m in market_size_dfs:
    market_size_lst.append(m)

market_mterics_dfs=map(share_nd_wd_calc,all_period_dfs)
market_metrics_lst=[]
for mm in market_mterics_dfs:
    market_metrics_lst.append(mm)

asp_dfs=map(asp_calc,all_period_dfs)
asp_dfs_lst=[]
for asp in asp_dfs:
    asp_dfs_lst.append(asp)


with st.container():
    freq_select=st.selectbox("Select Period to view",selected_freq)
    geography_select=st.selectbox("Select Geography level to view",selected_geography)
    freq_index=selected_freq.index(freq_select)
    geography_index=selected_geography.index(geography_select)
    


if model==None:
    t1,t2,t3=st.tabs(['Market Size','Market Metrics','ASP',])    
    
    with t1:
        st.dataframe(market_size_lst[freq_index][geography_index],use_container_width=True)
    
    with t2:
        st.dataframe(market_metrics_lst[freq_index][geography_index],use_container_width=True)
    
    with t3:
        st.dataframe(asp_dfs_lst[freq_index][geography_index],use_container_width=True)            
  

else:
    t1,t2,t3,t4=st.tabs(['Market Size','Market Metrics','ASP','% Contribution'])
    
    with t1:
        st.dataframe(market_size_lst[freq_index][geography_index],use_container_width=True)
        
    with t2:
        st.dataframe(market_metrics_lst[freq_index][geography_index],use_container_width=True)

    with t3:
        st.dataframe(asp_dfs_lst[freq_index][geography_index],use_container_width=True)            




























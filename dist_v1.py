# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:39:33 2023

@author: Marketpulse
"""

from __future__ import print_function
import pickle
import os.path
import io
import shutil
import requests
from mimetypes import MimeTypes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from io import StringIO
import pandas as pd
import numpy as np
import re
import polars as pl
import os
import streamlit as st
st.set_page_config(page_title='SuperBoard')
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore")
from itertools import product
from io import BytesIO
from functools import partial

#%%

SCOPES = ['https://www.googleapis.com/auth/drive']
creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
  
if not creds or not creds.valid:  
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
  
    # Save the access token in token.pickle
    # file for future usage
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('drive', 'v3', credentials=creds)

results = service.files().list(
    pageSize=10, fields="files(id, name)").execute()
items = results.get('files', [])
 
items_df=pd.DataFrame(items) 
all_csv=items_df['name']
items_df.set_index("name",inplace=True)

#%%

mapping=pd.read_csv("aux files/mapping_rule.csv")

order_col_dict={"Period":['Monthly', 'Quarterly', 'Fiscal Year', 'Calendar Year', 'Financial Year TD',
 'Financial Half-Year', 'Calendar Half-Year', 'Rolling Year'],
                "Precedence":[1, 2, 3, 4, 5, 6, 7, 8]}
order_col=pd.DataFrame(order_col_dict)

order_geo_dict={"Geography":['COUNTRY', 'ZONE', 'STATES', 'AREA'],
                "Precedence":[1,2,3,4]}
order_geo=pd.DataFrame(order_geo_dict)

size_metrics=['SALES UNITS','SALES VALUE INR','ASP']
comp_size_metrics=['\u0394 SALES UNITS','\u0394 SALES VALUE INR','\u0394 ASP']
market_metrics=['MS SALES UNITS','MS SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION','COUNTER SHARE']
market_metrics_dict={'MS SALES UNITS':'SALES UNITS','MS SALES VALUE INR':'SALES VALUE INR',
                     'NUMERIC DISTRIBUTION':'NUMERIC DISTRIBUTION',
                     'WEIGHTED DISTRIBUTION':'WEIGHTED DISTRIBUTION',
                     'COUNTER SHARE':'COUNTER SHARE'}
market_metrics_dict_exp=['% Vol MS','% Val MS','% NUMERIC DISTRIBUTION','% WEIGHTED DISTRIBUTION', '% COUNTER SHARE']
comp_market_metrics=['\u0394 MS SALES UNITS','\u0394 MS SALES VALUE INR',
    '\u0394 % NUMERIC DISTRIBUTION','\u0394 % WEIGHTED DISTRIBUTION','\u0394 % COUNTER SHARE']


comparison_type=['Seq','YoY']

fin_mnth_dict={1:[4,5,6,7,8,9,10,11,12,1],2:[4,5,6,7,8,9,10,11,12,1,2],3:[4,5,6,7,8,9,10,11,12,1,2,3],
               4:[4],5:[4,5],6:[4,5,6],7:[4,5,6,7],8:[4,5,6,7,8],9:[4,5,6,7,8,9],
               10:[4,5,6,7,8,9,10],11:[4,5,6,7,8,9,10,11],12:[4,5,6,7,8,9,10,11,12]}

qtr_dict={1:'aJFM',2:'aJFM',3:'aJFM',4:'bAMJ',5:'bAMJ',6:'bAMJ',
          7:'cJAS',8:'cJAS',9:'cJAS',10:'dOND',11:'dOND',12:'dOND'}

qtr_range_dict={1:'JFM',2:'AMJ',3:'JAS',4:'OND'}

period_dict={'Monthly':'MONTH_YEAR',
 'Quarterly':'Qtr_year',
 'Financial Year TD':'Fin_year td',
 'Calendar Year':'Cal_year',
 'Rolling Year':'Rolling_year',
 'Fiscal Year':'Fiscal_year',
 'Financial Half-Year':'NTD',
 'Calendar Half-Year':'NTD',}

mnth_dict={'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun',
           "07":'Jul',"08":'Aug',"09":'Sep',"10":"Oct","11":"Nov","12":"Dec"}

ui_freq_dict={'Month':'Monthly',
'Quarter':'Quarterly',
'FYTD':'Financial Year TD',
'CYTD':'Calendar Year',
'Rolling Year':'Rolling Year',
'Financial Half-Year':'Financial Half-Year',
'Calendar Half-Year':'Calendar Half-Year',
'FY':'Fiscal Year'}
    

#%%

# UDF

def period_dfs_creation(dataframe,date_max):
    period_dfs_list=[]
    month_year=str(date_max.year)[2:]+"_"+str(date_max.strftime("%m"))
    fin_year=[date_max.year-1 if date_max.month<4 else date_max.year][0]
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
        elif f=='Financial Year TD':
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
        
        elif f=='Fiscal Year':
            if date_max.month!=3:
                fisc_year_df=dataframe[(pd.Timestamp(day=31,month=3,year=fin_year-1)<dataframe['DATE']) &
                              (dataframe['DATE']<pd.Timestamp(day=2,month=3,year=fin_year))]
            period_dfs_list.append(fisc_year_df)    
                
    return period_dfs_list


def market_size_calc(dataframe):    
    if type(dataframe)==pd.DataFrame:        
        market_size_dfs=[]
        for g in selected_geography:            
            sub_dataframe=dataframe[dataframe['MODELS']=='All Models']
            brands_size_df=pd.pivot_table(data=sub_dataframe,index=[g,'BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                           aggfunc='sum')
            brands_size_df['ASP']=brands_size_df[['SALES UNITS','SALES VALUE INR']].apply(lambda z:z[1]/z[0],axis=1)
            total_df=dataframe[dataframe['BRANDS']=='Total']
            total_brand_size=pd.pivot_table(data=total_df,index=[g,'BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                           aggfunc='sum')
            total_brand_size['ASP']=total_brand_size[['SALES UNITS','SALES VALUE INR']].apply(lambda z:z[1]/z[0],axis=1)
            combined_df=pd.concat([total_brand_size,brands_size_df],axis=0)
            market_size_dfs.append(combined_df)
    
        return market_size_dfs

    elif dataframe==None:
        return None


def share_nd_wd_calc(dataframe):
    if type(dataframe)==pd.DataFrame:
        models_df=dataframe[dataframe['MODELS']=='All Models']
        market_metrics_dfs=[]
        for g in selected_geography:
            total_vals=dataframe[dataframe['BRANDS']=='Total'][[g,'SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION']]
            total_vals=(pd.pivot_table(data=total_vals,index=g,
            values=['SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION'],aggfunc='sum')).reset_index()
            total_vals.columns=[g, 'Total_NUMERIC DISTRIBUTION', 'Total_SALES UNITS', 'Total_SALES VALUE INR',
                   'Total_WEIGHTED DISTRIBUTION']
            nd_wd_df=(pd.pivot_table(data=models_df,index=[g,'BRANDS'],
                values=['SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION'],
                aggfunc='sum')).reset_index()
            combined_df=nd_wd_df.merge(total_vals)
            combined_df['SALES UNITS']=combined_df[['SALES UNITS','Total_SALES UNITS']].apply(lambda z:(z[0]/z[1])*100,axis=1)        
            combined_df['SALES VALUE INR']=combined_df[['SALES VALUE INR','Total_SALES VALUE INR']].apply(lambda z:(z[0]/z[1])*100,axis=1)        
            combined_df['NUMERIC DISTRIBUTION']=combined_df[['NUMERIC DISTRIBUTION','Total_NUMERIC DISTRIBUTION']].apply(lambda z:(z[0]/z[1])*100,axis=1)        
            combined_df['WEIGHTED DISTRIBUTION']=combined_df[['WEIGHTED DISTRIBUTION','Total_WEIGHTED DISTRIBUTION']].apply(lambda z:(z[0]/z[1])*100,axis=1)        
            
            cs_dataframe=dataframe[dataframe['CHANNEL']=='General Trade']
            if len(cs_dataframe)==0:
                continue
            cs_models_df=cs_dataframe[cs_dataframe['MODELS']=='All Models']
            cs_total_vals=cs_dataframe[cs_dataframe['BRANDS']=='Total'][[g,'SALES UNITS','WEIGHTED DISTRIBUTION']]
            cs_total_vals=(pd.pivot_table(data=cs_total_vals,index=g,
            values=['SALES UNITS','WEIGHTED DISTRIBUTION'],aggfunc='sum')).reset_index()
            cs_total_vals.columns=[g,'Total_SALES UNITS','Total_WEIGHTED DISTRIBUTION']
            cs_df=(pd.pivot_table(data=cs_models_df,index=[g,'BRANDS'],
                values=['SALES UNITS','WEIGHTED DISTRIBUTION'], aggfunc='sum')).reset_index()
            cs_combined_df=cs_df.merge(cs_total_vals)
            cs_combined_df['SALES UNITS']=cs_combined_df[['SALES UNITS','Total_SALES UNITS']].apply(lambda z:(z[0]/z[1])*100,axis=1)        
            cs_combined_df['WEIGHTED DISTRIBUTION']=cs_combined_df[['WEIGHTED DISTRIBUTION','Total_WEIGHTED DISTRIBUTION']].apply(lambda z:(z[0]/z[1])*100,axis=1)        
            combined_df['COUNTER SHARE']=cs_combined_df[['SALES UNITS','WEIGHTED DISTRIBUTION']].apply(lambda z:(z[0]/z[1])*100,axis=1)    
            combined_df=(combined_df[[g,'BRANDS','SALES UNITS', 'SALES VALUE INR', 'NUMERIC DISTRIBUTION', 'WEIGHTED DISTRIBUTION','COUNTER SHARE']]).set_index([g,'BRANDS'])
            market_metrics_dfs.append(combined_df)        
        
        return market_metrics_dfs
    
    elif dataframe==None:
        return None


def seq_dfs_creation(dataframe,date_max):
    seq_dfs_list=[]    
    for f in selected_freq:
        if f=='Monthly':
            seq_date_max=date_max-DateOffset(months=1)
            month_year=str(seq_date_max.year)[2:]+"_"+str(seq_date_max.strftime("%m"))
            mnth_df=dataframe[dataframe['MONTH_YEAR']==month_year]
            seq_dfs_list.append(mnth_df)
            
        elif f=='Quarterly':
            if date_max.month>2:
                qtr=date_max.month//3
                year=date_max.year
            else:
                qtr=4
                year=date_max.year-1
            seq_strt_date=pd.Timestamp(day=1,month=3*qtr-2,year=year)-DateOffset(months=3)
            seq_end_date=pd.Timestamp(day=1,month=3*qtr,year=year)-DateOffset(months=3)
            qtr_df=dataframe[(dataframe['DATE']>=seq_strt_date) & 
                             (dataframe['DATE']<=seq_end_date)]
            seq_dfs_list.append(qtr_df)
            
        elif f=='Financial Half-Year':
            if date_max.month>9:
                fy_c1_start=pd.Timestamp(day=31,month=3,year=date_max.year)-DateOffset(months=6)
                fy_c1_end=pd.Timestamp(day=2,month=9,year=date_max.year)-DateOffset(months=6)
                fin_halfyear_df=dataframe[(dataframe['DATE']>fy_c1_start) & 
                                          (dataframe['DATE']<fy_c1_end)]
            elif date_max.month<4:
                fy_c2_start=pd.Timestamp(day=31,month=3,year=date_max.year-1)-DateOffset(months=6)
                fy_c2_end=pd.Timestamp(day=2,month=9,year=date_max.year-1)-DateOffset(months=6)
                fin_halfyear_df=dataframe[(dataframe['DATE']>fy_c2_start) & 
                                          (dataframe['DATE']<fy_c2_end)]
            else:
                fy_c3_start=pd.Timestamp(day=30,month=9,year=date_max.year-1)-DateOffset(months=6)
                fy_c3_end=pd.Timestamp(day=1,month=4,year=date_max.year)-DateOffset(months=6)
                fin_halfyear_df=dataframe[(dataframe['DATE']>fy_c3_start) & 
                                          (dataframe['DATE']<fy_c3_end)]
            seq_dfs_list.append(fin_halfyear_df)
            
        elif f=='Calendar Half-Year':
            if date_max.month>6:
                cy_c1_start=pd.Timestamp(day=31,month=12,year=date_max.year-1)-DateOffset(months=6)
                cy_c1_end=pd.Timestamp(day=2,month=6,year=date_max.year)-DateOffset(months=6)
                cal_halfyear_df=dataframe[(dataframe['DATE']>cy_c1_start) & 
                                          (dataframe['DATE']<cy_c1_end)]
            else:
                cy_c2_start=pd.Timestamp(day=1,month=7,year=date_max.year-1)-DateOffset(months=6)
                cy_c2_end=pd.Timestamp(day=2,month=12,year=date_max.year-1)-DateOffset(months=6)              
                cal_halfyear_df=dataframe[(dataframe['DATE']>=cy_c2_start) & 
                                          (dataframe['DATE']<cy_c2_end)]
            seq_dfs_list.append(cal_halfyear_df)
        
        else:
            seq_dfs_list.append(None)
                
    return seq_dfs_list


def mrkt_size_growth(current_size,compare_size):
    growth_series=((current_size-compare_size)/compare_size)*100
    
    return growth_series

def mrkt_metrics_growth(current_size,compare_size):
    growth_series=current_size-compare_size
    
    return growth_series

def growth_df_gen(measure,df_list_var):
    yoy_size_growth_lst=[]
    seq_size_growth_lst=[]

    for l1 in range(len(selected_geography)):
        yoy_geo_df=pd.DataFrame()
        seq_geo_df=pd.DataFrame()
        for l2 in range(len(selected_freq)):
            if measure in size_metrics:
                yoy_series=mrkt_size_growth(df_list_var[0][l2][l1][str(measure)],df_list_var[1][l2][l1][str(measure)])
                yoy_geo_df=pd.concat([yoy_geo_df,yoy_series],axis=1)
                try:
                    seq_series=mrkt_size_growth(df_list_var[0][l2][l1][str(measure)],df_list_var[2][l2][l1][str(measure)])
                except:
                    seq_series=None
                seq_geo_df=pd.concat([seq_geo_df,seq_series],axis=1)
            elif measure in market_metrics:
                yoy_series=mrkt_metrics_growth(df_list_var[0][l2][l1][str(market_metrics_dict[measure])],df_list_var[1][l2][l1][market_metrics_dict[measure]])
                yoy_geo_df=pd.concat([yoy_geo_df,yoy_series],axis=1)
                try:
                    seq_series=mrkt_metrics_growth(df_list_var[0][l2][l1][market_metrics_dict[measure]],df_list_var[2][l2][l1][market_metrics_dict[measure]])
                except:
                    seq_series=None
                seq_geo_df=pd.concat([seq_geo_df,seq_series],axis=1)     
        yoy_geo_df.columns=selected_freq
        seq_geo_df.columns=pd.Series(selected_freq,index=selected_freq).drop(['Financial Year TD',
        'Calendar Year','Rolling Year','Fiscal Year'],axis=0,errors='ignore').values
        yoy_size_growth_lst.append(yoy_geo_df)    
        seq_size_growth_lst.append(seq_geo_df)
    
    return (yoy_size_growth_lst,seq_size_growth_lst)


def excel_conv_upd(list_dfs):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    for df,sn in zip(list_dfs,selected_geography):
        df.to_excel(writer,sheet_name=sn)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def rol_yr(date_in_format):
    rol_yr_num=((date_max-date_in_format).days//365)+1
    rol_yr_strt_dt=(date_max-DateOffset(years=rol_yr_num)+DateOffset(months=1)).strftime("%b-%y")
    rol_yr_end_dt=(date_max-DateOffset(years=rol_yr_num-1)).strftime("%b-%y")
    ret_string=rol_yr_strt_dt+" to "+rol_yr_end_dt
     
    return ret_string

def topn_brands(n,sort_metric):
    sub_dataframe=current_mnth_df[current_mnth_df['MODELS']=='All Models']
    brands_size_df=pd.pivot_table(data=sub_dataframe,index=['COUNTRY','BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                   aggfunc='sum')
    total_df=current_mnth_df[current_mnth_df['BRANDS']=='Total']
    total_brand_size=pd.pivot_table(data=total_df,index=['COUNTRY','BRANDS'],values=['SALES UNITS','SALES VALUE INR'],
                   aggfunc='sum')
    combined_df=pd.concat([total_brand_size,brands_size_df],axis=0)
    
    total_vals=current_mnth_df[current_mnth_df['BRANDS']=='Total'][['SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION']].sum()
    nd_wd_df=pd.pivot_table(data=sub_dataframe,index=['COUNTRY','BRANDS'],
        values=['SALES UNITS','SALES VALUE INR','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION'],
        aggfunc='sum')
    nd_wd_df['SALES UNITS']=nd_wd_df['SALES UNITS'].apply(lambda z:(z/total_vals['SALES UNITS'])*100)        
    nd_wd_df['SALES VALUE INR']=nd_wd_df['SALES VALUE INR'].apply(lambda z:(z/total_vals['SALES VALUE INR'])*100)        
    nd_wd_df['NUMERIC DISTRIBUTION']=nd_wd_df['NUMERIC DISTRIBUTION'].apply(lambda z:(z/total_vals['NUMERIC DISTRIBUTION'])*100)        
    nd_wd_df['WEIGHTED DISTRIBUTION']=nd_wd_df['WEIGHTED DISTRIBUTION'].apply(lambda z:(z/total_vals['WEIGHTED DISTRIBUTION'])*100)        
    nd_wd_df=nd_wd_df[['SALES UNITS', 'SALES VALUE INR', 'NUMERIC DISTRIBUTION', 'WEIGHTED DISTRIBUTION']]
    nd_wd_df.rename(columns={'SALES UNITS':'MARKET SHARE-VOLUME',
                                              'SALES VALUE INR':'MARKET SHARE-VALUE'},inplace=True)
    current_mnth_agg=(combined_df.merge(nd_wd_df,left_on=['BRANDS','COUNTRY'],
                                                  right_on=['BRANDS','COUNTRY'])).reset_index()

    return list(current_mnth_agg.nlargest(n,sort_metric)['BRANDS'])

def prev_period_growth(composite_col,df_trend,type_period,type_metric):
    prefix,col=composite_col[0],composite_col[1]
    if (type_period=="MONTH_YEAR" or type_period=="Qtr_year"):
        prev_col=col.split("-")[0]+"-"+str(int(col.split("-")[1])-1)    
    if (type_period=="Fin_year td" or type_period=="Fiscal_year"):
        prev_col=col.split(" ")[0]+" "+str(int(col.split(" ")[1].split("-")[0])-1)+"-"+str(int(col.split(" ")[1].split("-")[1])-1)
    if type_period=="Cal_year":
        prev_col=col.split(" ")[0]+" "+str(int(col.split(" ")[1])-1)
    if type_period=='Rolling_year':
        mid_step=[ i.split("-")[0]+"-"+str(int(i.split("-")[1])-1)  for i in col.split("to")]
        prev_col=mid_step[0]+" to"+mid_step[1]
    prev_composite_col=tuple([prefix,prev_col])
    try:
        if type_metric=="Market Size":
            growth_series=pd.Series(mrkt_size_growth(df_trend[composite_col],df_trend[prev_composite_col]),name="YoY "+prefix+" Growth "+prev_col+" vs "+col)
        else:
            growth_series=pd.Series(mrkt_metrics_growth(df_trend[composite_col],df_trend[prev_composite_col]),name="YoY "+prefix+" Growth "+prev_col+" vs "+col)
        
    except:
        growth_series=None
        
    return growth_series 

def trend_calc(period,metrics_type,op_flag=0):
    if period!='NTD':
        sub_df=df[df[period]!="No Use"]
        groupby_obj=sub_df.groupby(period)
        dfs_list=[]
        index_lst=groupby_obj[period].unique().index
        for g in index_lst:
            dfs_list.append(groupby_obj.get_group(g))
        if metrics_type=='Market Size':
            market_dfs=map(market_size_calc,dfs_list)
            market_lst=[]
            for m in market_dfs:
                market_lst.append(m)
            index_lst_rev=[re.sub('[abcd]','',e) for e in index_lst]
            if period=='MONTH_YEAR':
                index_lst_rev=[mnth_dict[q.split("_")[1]]+"-"+q.split("_")[0] for q in index_lst_rev]
            elif period=='Qtr_year':
                index_lst_rev=[q.split("_")[1]+"-"+q.split("_")[0] for q in index_lst_rev]
            possible=list(product(size_metrics,index_lst_rev))
        
        elif metrics_type=='Market Metrics':
            market_dfs=map(share_nd_wd_calc,dfs_list)
            market_lst=[]
            for m in market_dfs:
                market_lst.append(m)
            index_lst_rev=[re.sub('[abcd]','',e) for e in index_lst]    
            if period=='MONTH_YEAR':
                index_lst_rev=[mnth_dict[q.split("_")[1]]+"-"+q.split("_")[0] for q in index_lst_rev]
            elif period=='Qtr_year':
                index_lst_rev=[q.split("_")[1]+"-"+q.split("_")[0] for q in index_lst_rev]
            possible=list(product(market_metrics_dict_exp,index_lst_rev))
            
        geo_df_3=[]
        for g in range(len(selected_geography)):
            temp_df=pd.DataFrame()
            for p in range(len(market_lst)):
                append_df=market_lst[p][g]
                formatting_dict={'SALES UNITS':'aSALES UNITS',
                 'SALES VALUE INR':'bSALES VALUE INR',
                 'NUMERIC DISTRIBUTION':'cNUMERIC DISTRIBUTION',
                 'WEIGHTED DISTRIBUTION':'dWEIGHTED DISTRIBUTION',
                 'ASP':'bbASP',
                 'COUNTER SHARE':'ddCOUNTER SHARE'}
                append_df.columns=[formatting_dict[co] for co in append_df.columns]
                append_df.columns=[col+"_"+index_lst[p] for col in append_df.columns]
                temp_df=pd.concat([temp_df,append_df],axis=1)
                temp_df=temp_df[sorted(temp_df.columns)]
                temp_df.sort_values(temp_df.columns[0],ascending=False)
                if metrics_type=='Market Metrics':
                    temp_df=temp_df.applymap(lambda nu:np.round(nu,1))
                else:
                    temp_df=temp_df.applymap(lambda nu:np.round(nu,0))
                if brand_flag:
                    temp_df=temp_df.reset_index()
                    temp_df=temp_df[temp_df['BRANDS'].isin(brand_select)]
                    temp_df=temp_df.set_index([selected_geography[g],'BRANDS'])
            geo_df_3.append(temp_df)
        
        col_series=pd.Series(possible,index=possible)
        columns_op_3=pd.MultiIndex.from_tuples(col_series)
        if op_flag==1:
            geo_df_mod=[]
            for dfm,sel_geo in zip(geo_df_3,selected_geography):
                dfm.columns=columns_op_3            
                id_series=pd.DataFrame(np.full((len(dfm),1),sel_geo),index=dfm.index,columns=pd.MultiIndex.from_tuples([('Geography Type','')]))
                parent_series=[parent_dict[item+"_"+sel_geo]['PARENT'] for item in dfm.index.get_level_values((sel_geo))]
                parent_series=pd.DataFrame(parent_series,index=dfm.index,columns=pd.MultiIndex.from_tuples([('Parent Geo','')]))
                dfm=pd.concat([id_series,parent_series,dfm],axis=1)
                if brand_flag:
                    dfm=dfm.reset_index()
                    dfm=dfm[dfm[('BRANDS','')].isin(brand_select)]
                    dfm=dfm.set_index([(sel_geo,''),('Geography Type',''),
                                ('Parent Geo',''),('BRANDS','')])
                else:
                    dfm=dfm.reset_index().set_index([(sel_geo,''),('Geography Type',''),
                                ('Parent Geo',''),('BRANDS','')])
                    
                parfunc_trend = partial(prev_period_growth, df_trend=dfm,type_period=period,type_metric=metrics_type)
                trend_yoy_map=map(parfunc_trend,list(dfm.columns))
                trend_yoy_list=[]
                for w in trend_yoy_map:
                    trend_yoy_list.append(w)
                trend_yoy_df=pd.concat(trend_yoy_list,axis=1)    
                trend_df_cols=[(i.split("Growth ")[0]+"Growth",i.split("Growth ")[1]) for i in trend_yoy_df.columns]    
                trend_yoy_df.columns=pd.MultiIndex.from_tuples(trend_df_cols)                    
                trend_growth_df=pd.concat([dfm,trend_yoy_df],axis=1)                    
                geo_df_mod.append(trend_growth_df)

            return geo_df_mod
        
        else:
            geo_df_4=[]
            for dfm in geo_df_3:
                dfm.columns=columns_op_3
                parfunc_trend = partial(prev_period_growth, df_trend=dfm,type_period=period,type_metric=metrics_type)
                trend_yoy_map=map(parfunc_trend,list(dfm.columns))
                trend_yoy_list=[]
                for w in trend_yoy_map:
                    trend_yoy_list.append(w)
                trend_yoy_df=pd.concat(trend_yoy_list,axis=1)    
                trend_df_cols=[(i.split("Growth ")[0]+"Growth",i.split("Growth ")[1]) for i in trend_yoy_df.columns]    
                trend_yoy_df.columns=pd.MultiIndex.from_tuples(trend_df_cols)                    
                trend_growth_df=pd.concat([dfm,trend_yoy_df],axis=1)
                geo_df_4.append(trend_growth_df)
            return geo_df_4

    else:
        empty_df_lst=[]
        for egeo in selected_geography:
            empty_df_lst.append(pd.DataFrame())
        return empty_df_lst



def display_func_2(final_geo_df_list,market_type):
    if len(selected_geography)==1:
        a1=st.tabs(selected_geography)[0]
        with a1:
            st.dataframe(final_geo_df_list[0],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[0]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[0])+" "+str(f)+" Trend" +" .xlsx")           
        down_df=excel_conv_upd([final_geo_df_list[0]])
        st.download_button(label="Click To Download!",data=down_df,
                           file_name=str(selected_geography[0])+" "+str(market_type)+" .xlsx")
        
    elif len(selected_geography)==2:
        a1,a2=st.tabs(selected_geography)
        with a1:
            st.dataframe(final_geo_df_list[0],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[0]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[0])+" "+str(f)+" Trend" +" .xlsx")
        with a2:
            st.dataframe(final_geo_df_list[1],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[1]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[1])+" "+str(f)+" Trend" +" .xlsx")            
        down_df=excel_conv_upd([final_geo_df_list[0],final_geo_df_list[1]])
        st.download_button(label="Click To Download!",data=down_df,
                           file_name=str(selected_geography[0])+" "+str(selected_geography[1])+" "+str(market_type)+" .xlsx")
    elif len(selected_geography)==3:
        a1,a2,a3=st.tabs(selected_geography)
        with a1:
            st.dataframe(final_geo_df_list[0],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[0]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[0])+" "+str(f)+" Trend" +" .xlsx")
        with a2:
            st.dataframe(final_geo_df_list[1],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[1]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[1])+" "+str(f)+" Trend" +" .xlsx")            
        with a3:
            st.dataframe(final_geo_df_list[2],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[2]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[2])+" "+str(f)+" Trend" +" .xlsx")
        down_df=excel_conv_upd([final_geo_df_list[0],final_geo_df_list[1],final_geo_df_list[2]])
        st.download_button(label="Click To Download!",data=down_df,
        file_name=str(selected_geography[0])+" "+str(selected_geography[1])+" "+str(selected_geography[2])+" "+str(market_type)+" .xlsx")    
    elif len(selected_geography)==4:
        a1,a2,a3,a4=st.tabs(selected_geography)        
        with a1:
            st.dataframe(final_geo_df_list[0],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[0]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[0])+" "+str(f)+" Trend" +" .xlsx")
        with a2:
            st.dataframe(final_geo_df_list[1],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[1]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[1])+" "+str(f)+" Trend" +" .xlsx")            
        with a3:
            st.dataframe(final_geo_df_list[2],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[2]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[2])+" "+str(f)+" Trend" +" .xlsx")    
        with a4:
            st.dataframe(final_geo_df_list[3],use_container_width=True)
            for f in selected_freq:
                with st.expander(f+" Trend"):
                    trend_df=trend_calc(period_dict[f], market_type)[3]
                    st.dataframe(trend_df)
                    trend_file=excel_conv_upd([trend_df])
                    st.download_button(label="Click To Download Trend!",data=trend_file,
                                       file_name=str(selected_geography[3])+" "+str(f)+" Trend" +" .xlsx")
        down_df=excel_conv_upd([final_geo_df_list[0],final_geo_df_list[1],
                                final_geo_df_list[2],final_geo_df_list[3]])
        st.download_button(label="Click To Download!",data=down_df,
        file_name=str(selected_geography[0])+" "+str(selected_geography[1])+" "+str(selected_geography[2])+" "+str(selected_geography[3])+" "+str(market_type)+" .xlsx")    

def excel_conv_format(size_dfs_list,metrics_dfs_list):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    fin_size_lst=[]
    for s_df,sel_geo in zip(size_dfs_list,selected_geography):
        s_df.rename(columns={'COUNTRY':'Geography','STATES':'Geography',
                                 'ZONE':'Geography','AREA':'Geography'},inplace=True)
        size_id_series=pd.DataFrame(np.full((len(s_df),1),sel_geo),index=s_df.index,columns=pd.MultiIndex.from_tuples([('Geography Type','','')]))
        size_parent_series=[parent_dict[item+"_"+sel_geo]['PARENT'] for item in s_df.index.get_level_values((sel_geo, '', ''))]
        size_parent_series=pd.DataFrame(size_parent_series,index=s_df.index,columns=pd.MultiIndex.from_tuples([('Parent Geo','','')]))
        s_df=pd.concat([size_id_series,size_parent_series,s_df],axis=1)
        fin_size_lst.append(s_df)
    size_exp_df=pd.concat(fin_size_lst)
    size_exp_df=size_exp_df.reset_index().set_index([size_exp_df.reset_index().columns[0],('Geography Type','',''),
                ('Parent Geo','',''),('BRANDS','','')])    
    size_exp_df.index.names=['Geography','Geography Type','Parent Geo','Brands']
    size_exp_df.reset_index(inplace=True)
    size_exp_df.to_excel(writer,sheet_name="Market Size Summary")
    
    fin_metrics_lst=[]
    for m_df,sel_geo in zip(metrics_dfs_list,selected_geography):
        m_df.rename(columns={'COUNTRY':'Geography','STATES':'Geography',
                                 'ZONE':'Geography','AREA':'Geography'},inplace=True)
        metrics_id_series=pd.DataFrame(np.full((len(m_df),1),sel_geo),index=m_df.index,columns=pd.MultiIndex.from_tuples([('Geography Type','','')]))
        metrics_parent_series=[parent_dict[item+"_"+sel_geo]['PARENT'] for item in m_df.index.get_level_values((sel_geo, '', ''))]
        metrics_parent_series=pd.DataFrame(metrics_parent_series,index=m_df.index,columns=pd.MultiIndex.from_tuples([('Parent Geo','','')]))
        m_df=pd.concat([metrics_id_series,metrics_parent_series,m_df],axis=1)
        fin_metrics_lst.append(m_df)
    metrics_exp_df=pd.concat(fin_metrics_lst)  
    metrics_exp_df=metrics_exp_df.reset_index().set_index([metrics_exp_df.reset_index().columns[0],('Geography Type','',''),
                ('Parent Geo','',''),('BRANDS','','')])
    metrics_exp_df.index.names=['Geography','Geography Type','Parent Geo','Brands']
    metrics_exp_df.reset_index(inplace=True)
    metrics_exp_df.to_excel(writer,sheet_name="Market Metrics Summary")
    
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def trend_file_creation(type_market):
    all_freq_dfs=[]
    for fre in selected_freq:
        if (fre=='Financial Half-Year' or fre=='Calendar Half-Year'):
            continue
        else:    
            freq_df_list=trend_calc(period_dict[fre], type_market,1)
            freq_df=pd.concat(freq_df_list,axis=0)
            all_freq_dfs.append(freq_df)
    
    req_df=all_freq_dfs[0]
    req_df['Refer']=[x for x in range(len(req_df))]
    for j in all_freq_dfs[1:]:
        req_df=req_df.merge(j,how='outer',left_index=True,right_index=True,sort = True)
    req_df=(((req_df.reset_index()).sort_values('Refer')).set_index([req_df.reset_index().columns[0],('Geography Type',''),
                ('Parent Geo',''),('BRANDS','')])).drop(columns=['Refer'])
    req_df.index.names=['Geography','Geography Type','Parent Geo','Brands']
    return req_df    

def excel_conv_trend(size_trend_df,metrics_trend_df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    size_trend_df.reset_index(inplace=True)
    size_trend_df.to_excel(writer,sheet_name="Market Size Trend")
    metrics_trend_df.reset_index(inplace=True)    
    metrics_trend_df.to_excel(writer,sheet_name="Market Metrics Trend")
    writer.close()
    processed_data = output.getvalue()
    return processed_data


#%%

# User Inputs:
product_m=st.selectbox("Select Product",mapping['SUB CATEGORY'].unique()) # To be selected from drop-down list # Category/Sub-category

freq_type=list(ui_freq_dict.keys())
    
selected_freq=st.multiselect("Select the period types",freq_type,default=freq_type[0])
selected_freq=[ui_freq_dict[value] for value in selected_freq]

order_col=order_col.merge(pd.Series(selected_freq,name='Period'))
order_col=order_col.sort_values(by='Precedence')
selected_freq=list(order_col['Period'])


geography_level=["COUNTRY",'STATES','ZONE','AREA'] 
selected_geography=st.multiselect("Select the Geography Level",geography_level,default=geography_level[0])

order_geo=order_geo.merge(pd.Series(selected_geography,name='Geography'))
order_geo=order_geo.sort_values(by='Precedence')
selected_geography=list(order_geo['Geography'])


#%%

# Getting & reading the relevant CSV file
rule=mapping[mapping['SUB CATEGORY']==product_m]['Rule'].unique()[0]
csv_name=all_csv[all_csv.str.contains(rule)].values[0]

# Specifying file id required here.
file_id=items_df.loc[csv_name,'id']

@st.cache_resource
def get_file(f_id):
    request = service.files().get_media(fileId=f_id)
    fh = io.BytesIO()
      
    # Initialise a downloader object to download the file
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*1024)
    done = False
    
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)      
    
    pl_df=pl.read_csv(fh)
    pd_df=pl_df.to_pandas()
    del pl_df
    
    return pd_df

df=get_file(file_id)
# Subsetting dataframe for the particular product:
df=df[df['SUB CATEGORY']==product_m]

# Appending Missing date data:
append_df_list=[]
add_subset_df=df[['CATEGORY', 'SUB CATEGORY', 'COUNTRY', 'ZONE',
       'STATES', 'AREA', 'CHANNEL','COMPANY', 'BRANDS', 'MODELS',
       'SPECIFICATION-1', 'SPECIFICATION-2', 'SPECIFICATION-3',
       'SPECIFICATION-4', 'SPECIFICATION-5']].drop_duplicates()
unq_period=pd.to_datetime(df['PERIOD'].unique())
if unq_period.min()>pd.Timestamp(day=1,month=5,year=2020):
    pass
else:
    for j in ['01-04-2020','01-05-2021']:
        if j not in df['PERIOD'].unique():
            loop_add_subset_df=add_subset_df.copy()
            remove_cols=list(add_subset_df.columns)+['PERIOD']
            missing_df_cols=df.columns.difference(remove_cols)
            cols_num=len(missing_df_cols)
            missing_df=pd.DataFrame(np.zeros((len(add_subset_df),cols_num)),columns=missing_df_cols)
            missing_df['PERIOD']=j
            for i in missing_df.columns:
                loop_add_subset_df[i]=missing_df[i].values
            df=pd.concat([df,loop_add_subset_df],axis=0)

df=df.sort_values("PERIOD")
# Subsetting dataframe for the defined date range:
df['DATE']=pd.to_datetime(df['PERIOD'])
df['MONTH_YEAR']=df['DATE'].apply(lambda z:z.strftime("%y_%m"))   

# Option to select Channel:
channel_opts=list(df['CHANNEL'].unique())    
selected_channel=st.multiselect("Select the Channels",channel_opts,default='General Trade')   
df=df[df['CHANNEL'].isin(selected_channel)]
    
# Subseting dataframe for the specified time period:
var_date_sel=pd.to_datetime(df['DATE'].unique())
var_date_sel=[sel_dt.strftime("%b-%y") for sel_dt in var_date_sel]
start_date=st.selectbox("Select Start Date",var_date_sel,index=0)
end_date=st.selectbox("Select End Date",var_date_sel,index=len(var_date_sel)-1)

start_date=pd.Timestamp(datetime.strptime(start_date,"%b-%y"))
end_date=pd.Timestamp(datetime.strptime(end_date,"%b-%y"))
df=df[(df['DATE']>=start_date) & (df['DATE']<=end_date)]
    

date_max=df['DATE'].max()
#date_max=pd.Timestamp(day=1,month=2,year=2023)

st.write("Current Date: ",date_max.strftime("%b-%Y"))

parent_zone=(df[['COUNTRY', 'ZONE']].drop_duplicates()).rename(columns={"COUNTRY":"PARENT",'ZONE':"CHILD"})
parent_zone['CHILD']=parent_zone['CHILD'].apply(lambda name:name+"_ZONE")
parent_states=(df[['ZONE','STATES']].drop_duplicates()).rename(columns={"ZONE":"PARENT",'STATES':"CHILD"})
parent_states['CHILD']=parent_states['CHILD'].apply(lambda name:name+"_STATES")
parent_area=(df[['STATES','AREA']].drop_duplicates()).rename(columns={"STATES":"PARENT",'AREA':"CHILD"})
parent_area['CHILD']=parent_area['CHILD'].apply(lambda name:name+"_AREA")

parent_df=pd.concat([parent_zone,parent_states,parent_area])
parent_df=pd.concat([parent_df,pd.DataFrame([['India','India_COUNTRY']],columns=['PARENT','CHILD'])])

parent_df=parent_df.set_index("CHILD")
parent_dict=parent_df.T.to_dict()

    
#%%

# Driver Code

brand_flag=False
topn_flag=st.checkbox("Top N Brands View Only",key='topbrand')
if topn_flag:
    sort_metric=st.selectbox("Select Metric!",['SALES UNITS','SALES VALUE INR',
                'MARKET SHARE-VOLUME','MARKET SHARE-VALUE','NUMERIC DISTRIBUTION','WEIGHTED DISTRIBUTION'])
    select_n=st.number_input("Select n",step=5,value=10)
    current_mnth_df=df[df['DATE']==date_max]
    n_brands=topn_brands(select_n, sort_metric)+['Total']
    df=df[df['BRANDS'].isin(n_brands)]
    st.caption("The Top N Brands are calculated at country level for the current month")

if topn_flag==False:
    brand_flag=st.checkbox("Brand View",key='brandspecific')
    if brand_flag:
#        brand_select=st.selectbox("Select the Brand!",df['BRANDS'].unique())
        brand_select=st.multiselect("Select the Brands!",df['BRANDS'].unique())
        brand_select=brand_select+["Total"]


all_period_dfs=period_dfs_creation(df,date_max)
for d,q in zip(all_period_dfs,selected_freq):
    st.write('Period Type- ',q)
    st.write('Period Start: ',d['DATE'].min().strftime("%b-%y"),'\t','Period End: ',d['DATE'].max().strftime("%b-%y"))
st.header(" ")

date_range=[]
for per_df,fr in zip(all_period_dfs,selected_freq):
    base_date=per_df['DATE'].min()
    if fr=='Monthly':
        date_range.append(base_date.strftime("%b-%y"))        
    elif fr=='Quarterly':
        date_range.append(qtr_range_dict[base_date.quarter]+"-"+base_date.strftime("%y"))
    elif fr=='Fiscal Year':
        date_range.append("FY "+str(base_date.year)[2:]+"-"+str(base_date.year+1)[2:])
    elif fr=='Calendar Year':
        date_range.append("CYTD "+str(base_date.year)[2:])
    elif fr=='Financial Year TD':
        date_range.append("FYTD "+str(base_date.year)[2:]+"-"+str(base_date.year+1)[2:])
    elif fr=='Financial Half-Year':
        date_range.append(per_df['DATE'].min().strftime("%b:%y")+" - "+per_df['DATE'].max().strftime("%b:%y"))
    elif fr=='Calendar Half-Year':
        date_range.append(per_df['DATE'].min().strftime("%b:%y")+" - "+per_df['DATE'].max().strftime("%b:%y"))
    elif fr=='Rolling Year':
        date_range.append(per_df['DATE'].min().strftime("%b-%y")+" to "+per_df['DATE'].max().strftime("%b-%y"))
date_range_size=np.array([3*[arr_el] for arr_el in date_range]).flatten().tolist()
date_range_metrics=np.array([5*[arr_el] for arr_el in date_range]).flatten().tolist()


yoy_date_max=pd.Timestamp(day=date_max.day,month=date_max.month,year=date_max.year-1)
yoy_period_dfs=period_dfs_creation(df,yoy_date_max)

seq_period_dfs=seq_dfs_creation(df, date_max)


periods_df=[all_period_dfs,yoy_period_dfs,seq_period_dfs]

market_size_dfs_list=[]
for p in periods_df:
    market_size_dfs=map(market_size_calc,p)
    market_size_lst=[]
    for m in market_size_dfs:
        market_size_lst.append(m)
    market_size_dfs_list.append(market_size_lst)    


market_metrics_dfs_list=[]
for p in periods_df:
    market_metrics_dfs=map(share_nd_wd_calc,p)
    market_metrics_lst=[]
    for mm in market_metrics_dfs:
        market_metrics_lst.append(mm)
    market_metrics_dfs_list.append(market_metrics_lst)    


vol_yoy_growth_lst,vol_seq_growth_lst=growth_df_gen('SALES UNITS',market_size_dfs_list)
val_yoy_growth_lst,val_seq_growth_lst=growth_df_gen('SALES VALUE INR',market_size_dfs_list)
asp_yoy_growth_lst,asp_seq_growth_lst=growth_df_gen('ASP',market_size_dfs_list)


geo_df_fin_lst=[]
for g in range(len(selected_geography)):
    geo_df=pd.concat([vol_seq_growth_lst[g],vol_yoy_growth_lst[g],
                      val_seq_growth_lst[g],val_yoy_growth_lst[g],
                     asp_seq_growth_lst[g],asp_yoy_growth_lst[g]],axis=1)
    geo_df=geo_df.applymap(lambda num:np.round(num,1))
    geo_df_fin_lst.append(geo_df)

mrkt_size_possible=list(product(comp_size_metrics,selected_freq,comparison_type))
p_series=pd.Series(mrkt_size_possible,index=mrkt_size_possible)
to_drop=list(product(comp_size_metrics,['Calendar Year','Rolling Year','Financial Year TD','Fiscal Year'],
                     ['Seq']))
p_series=p_series.drop(to_drop,axis=0,errors='ignore')

columns_op=pd.MultiIndex.from_tuples(p_series)
for dfe in geo_df_fin_lst:
    dfe.columns=columns_op

current_periods_size_df_list=market_size_dfs_list[0]
abs_size_dfs_list=[]
for g in range(len(selected_geography)):
    tempry_geo_df=pd.DataFrame()
    for p in range(len(selected_freq)):
        tempry_geo_df=pd.concat([tempry_geo_df,current_periods_size_df_list[p][g]],axis=1)
    tempry_geo_df=tempry_geo_df.applymap(lambda num:np.round(num,0))    
    abs_size_dfs_list.append(tempry_geo_df)    
    
abs_size_cols=list(product(selected_freq,size_metrics))
abs_size_cols=[tuple(list(reversed(element))+[date_range_size[dtr]]) for element,dtr in zip(abs_size_cols,range(len(date_range_size)))]    

columns_abs_size=pd.MultiIndex.from_tuples(abs_size_cols)
for abs_df in abs_size_dfs_list:
    abs_df.columns=columns_abs_size

exp_geo_size=[]
for i in range(len(selected_geography)):
    combined_size_df=((abs_size_dfs_list[i].reset_index().merge(geo_df_fin_lst[i].reset_index(),
                    right_on=[('level_0','',''),('level_1','','')],
                    left_on=[(selected_geography[i],'',''),('BRANDS','','')])).drop(columns=[('level_0','',''),
                                                                                 ('level_1','','')]))
    combined_size_df=combined_size_df.sort_values([combined_size_df.columns[0],combined_size_df.columns[2]],ascending=[True,False])                          
    if brand_flag:
        brand_only_df=combined_size_df[combined_size_df[('BRANDS','','')].isin(brand_select)]
        combined_size_exp_df=brand_only_df.set_index([(selected_geography[i],'',''),('BRANDS','','')])
    else:
        combined_size_exp_df=combined_size_df.set_index([(selected_geography[i],'',''),('BRANDS','','')])
    exp_geo_size.append(combined_size_exp_df)
                                                                                 
#%%

ms_vol_yoy_growth_lst,ms_vol_seq_growth_lst=growth_df_gen('MS SALES UNITS',market_metrics_dfs_list)
ms_val_yoy_growth_lst,ms_val_seq_growth_lst=growth_df_gen('MS SALES VALUE INR',market_metrics_dfs_list)
nd_yoy_growth_lst,nd_seq_growth_lst=growth_df_gen('NUMERIC DISTRIBUTION',market_metrics_dfs_list)
wd_yoy_growth_lst,wd_seq_growth_lst=growth_df_gen('WEIGHTED DISTRIBUTION',market_metrics_dfs_list)
cs_yoy_growth_lst,cs_seq_growth_lst=growth_df_gen('COUNTER SHARE',market_metrics_dfs_list)


geo_df_fin_lst_2=[]
for g in range(len(selected_geography)):
    geo_df_2=pd.concat([ms_vol_seq_growth_lst[g],ms_vol_yoy_growth_lst[g],ms_val_seq_growth_lst[g],ms_val_yoy_growth_lst[g],
        nd_seq_growth_lst[g],nd_yoy_growth_lst[g],wd_seq_growth_lst[g],wd_yoy_growth_lst[g],
        cs_seq_growth_lst[g],cs_yoy_growth_lst[g]],axis=1)
    geo_df_2=geo_df_2.applymap(lambda num:np.round(num,1))
    geo_df_fin_lst_2.append(geo_df_2)

mrkt_metrics_possible=list(product(comp_market_metrics,selected_freq,comparison_type))
p_series_2=pd.Series(mrkt_metrics_possible,index=mrkt_metrics_possible)
to_drop_2=list(product(comp_market_metrics,['Calendar Year','Rolling Year','Financial Year TD','Fiscal Year'],
                     ['Seq']))
p_series_2=p_series_2.drop(to_drop_2,axis=0,errors='ignore')

columns_op_2=pd.MultiIndex.from_tuples(p_series_2)
for dfe in geo_df_fin_lst_2:
    dfe.columns=columns_op_2    

current_periods_metrics_df_list=market_metrics_dfs_list[0]
abs_metrics_dfs_list=[]
for g in range(len(selected_geography)):
    tempry_geo_df_2=pd.DataFrame()
    for p in range(len(selected_freq)):
        tempry_geo_df_2=pd.concat([tempry_geo_df_2,current_periods_metrics_df_list[p][g]],axis=1)
    tempry_geo_df_2=tempry_geo_df_2.applymap(lambda numm:np.round(numm,1))    
    abs_metrics_dfs_list.append(tempry_geo_df_2)    
    
abs_metrics_cols=list(product(selected_freq,market_metrics_dict_exp))
abs_metrics_cols=[tuple(list(reversed(element))+[date_range_metrics[dtr]]) for element,dtr in zip(abs_metrics_cols,range(len(date_range_metrics)))]    

columns_abs_metrics=pd.MultiIndex.from_tuples(abs_metrics_cols)
for abs_df in abs_metrics_dfs_list:
    abs_df.columns=columns_abs_metrics

exp_geo_metrics=[]
for j in range(len(selected_geography)):
    combined_metrics_df=((abs_metrics_dfs_list[j].reset_index().merge(geo_df_fin_lst_2[j].reset_index(),
                    right_on=[('level_0','',''),('level_1','','')],
                    left_on=[(selected_geography[j],'',''),('BRANDS','','')])).drop(columns=[('level_0','',''),
                    ('level_1','','')]))
    combined_metrics_df.sort_values([combined_metrics_df.columns[0],combined_metrics_df.columns[2]],
                                    ascending=[True,False],inplace=True)                                                                                         
    if brand_flag:
        brand_only_df_2=combined_metrics_df[combined_metrics_df[('BRANDS','','')].isin(brand_select)]
        combined_metrics_exp_df=brand_only_df_2.set_index([(selected_geography[j],'',''),('BRANDS','','')])
    else:
        combined_metrics_exp_df=combined_metrics_df.set_index([(selected_geography[j],'',''),('BRANDS','','')])
    combined_metrics_exp_df=combined_metrics_exp_df.applymap(lambda num:np.round(num,2))    
    exp_geo_metrics.append(combined_metrics_exp_df)



#%%
# labelling each month:
    
df['Qtr_year']=df['DATE'].apply(lambda dt:str(dt.year)[2:]+"_"+qtr_dict[dt.month])
qtr_chk=df.groupby('Qtr_year')['DATE'].nunique()
df['Qtr_year']=df['Qtr_year'].apply(lambda qt:qt if (qtr_chk[qt]==3) else "No Use")

df['Fin_year td']=df['DATE'].apply(lambda dt:"FYTD "+str(dt.year)[2:]+"-"+str(dt.year+1)[2:] if dt.month in fin_mnth_dict[date_max.month] else 'No Use')
fytd_chk=df.groupby('Fin_year td')['DATE'].nunique()
fytd_mnth_len=len(fin_mnth_dict[date_max.month])
df['Fin_year td']=df['Fin_year td'].apply(lambda fy:fy if (fytd_chk[fy]==fytd_mnth_len) else "No Use")


df['Fiscal_year']=df['DATE'].apply(lambda dt:"FY "+str(dt.to_period('Q-MAR').qyear-1)[2:]+"-" +str(dt.to_period('Q-MAR').qyear)[2:])
fy_chk=df.groupby('Fiscal_year')['DATE'].nunique()
df['Fiscal_year']=df['Fiscal_year'].apply(lambda fs:fs if (fy_chk[fs]==12) else "No Use")


df['Cal_year']=df['DATE'].apply(lambda dt:"CYTD "+str(dt.year) if dt.month<=date_max.month else 'No Use')
cy_chk=df.groupby('Cal_year')['DATE'].nunique()
df['Cal_year']=df['Cal_year'].apply(lambda cy: cy if (cy_chk[cy]==date_max.month) else "No Use")


unq_dates=df[['DATE']].drop_duplicates()
unq_dates['Rolling_year']=unq_dates['DATE'].apply(lambda dt:rol_yr(dt))
df=df.merge(unq_dates)
rl_chk=df.groupby('Rolling_year')['DATE'].nunique()
df['Rolling_year']=df['Rolling_year'].apply(lambda ry: ry if (rl_chk[ry]==12) else "No Use")




#%%

# UI 

t1,t2=st.tabs(['Market Size','Market Metrics'])    

with t1:
    display_func_2(exp_geo_size,'Market Size')            
    
with t2:
    display_func_2(exp_geo_metrics,'Market Metrics')                           
    
down_df=excel_conv_format(exp_geo_size,exp_geo_metrics)
st.download_button(label="Click To Download Summary !",data=down_df,
                   file_name=str(product_m)+" Summary"+" .xlsx")

exp_trend_size=trend_file_creation("Market Size")
exp_trend_metrics=trend_file_creation("Market Metrics")    
exp_trend_df=excel_conv_trend(exp_trend_size,exp_trend_metrics)
st.download_button(label="Click To Download Trend !",data=exp_trend_df,
                   file_name=str(product_m)+" Trend"+" .xlsx")






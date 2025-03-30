import pandas as pd
import numpy as np
import os
from datetime import datetime

uat = int(os.getenv('UAT'))
# uat = int(1117)

def min_mc_weight(row):
    # Use 'average FC' if 'Min_weight' is NaN (blank)
    min_weight = row['Min_weight'] if pd.notna(row['Min_weight']) else row['average FC']
    return (min_weight * 1219) / row['width']

def derive_mc_weight(row):
    # Use 'average FC' if 'Min_weight' is NaN (blank)
    return (row['weight'] * 1219) / row['width']

# FINISH GOODS - add forecast + stockend (needcut > 0) + coil center priority
finish_csv = pd.read_csv(f"data/SlitRequestSRO-{uat}.csv")
month = finish_csv['Month'][0]
finish_csv['3rd Priority'] = finish_csv['3rd Priority'].astype("object")
finish_csv['1st Priority'] = finish_csv['1st Priority'].astype("object")
finish_csv['2nd Priority'] = finish_csv['2nd Priority'].astype("object")

finish_csv.drop(columns=['Customer'],inplace=True)
finish_csv.rename(columns={'Customer Short Name':"Customer"},inplace=True)

finish_csv['Length'] = finish_csv['Length'].str.upper()
finish_csv["MC Code"] = finish_csv.apply(lambda row: f"{row['Maker']} {row['Spec']} {row['Thickness']:.2f}", axis=1)
# finish_csv = finish_csv[(finish_csv['Length'] == 'C') |(finish_csv['CUT COIL'] =="yes")]

# Fix standard
finish_csv['Standard'] = finish_csv['Standard'].str.lower()
finish_csv["Customer_FG"] = (finish_csv['Customer'].astype(object)+"-"+finish_csv['FG Code'].astype(object))
finish_csv['Stockend Qt'] = finish_csv['Stockend Qt'].astype(float)

# STOCK END
stock_end = pd.read_excel(f"data/Stock End Details_{month}.xlsx")
stock_end.drop(columns=['(Do Not Modify) Row Checksum','(Do Not Modify) Modified On','(Do Not Modify) StockEndDetail',
'(Do Not Modify) Modified On','(Do Not Modify) Row Checksum','BP Code'],inplace=True)
stock_end.rename(columns={"Customer Short Name (BP Code) (Customer Master)":"Customer",
                        'Width (FG Code) (Material Code Master (FG codes))':"Width",
                        "Length (FG Code) (Material Code Master (FG codes))":"Length",
                        "Stock End Qt":"Stockend Qt"},inplace=True)

stock_end = stock_end[(stock_end['Length'] == 'C') |(stock_end['Length'] == '1219')]
stock_end["Customer_FG"] = (stock_end['Customer'].astype(object)+"-"+stock_end['FG Code'].astype(object))
total_stock_end = stock_end.groupby('Customer_FG')['Stockend Qt'].sum().reset_index()
substock_end = stock_end[['FG Code','Customer','Customer_FG']]

total_stock_end['Stockend Qt'] = total_stock_end['Stockend Qt'].astype(float)
total_stock_end['Need Cut'] = total_stock_end['Stockend Qt']

merged_stock_end = pd.merge(total_stock_end, substock_end, on="Customer_FG",how="left")
merged_stock_end_filtered = merged_stock_end[~merged_stock_end['Customer_FG'].isin(finish_csv['Customer_FG'])]

concatenated_po = pd.concat([finish_csv, merged_stock_end_filtered], ignore_index=True)

# COIL CENTER PRIORITY
coil_center_prio = pd.read_excel("data/master/Coil Center Priorities.xlsx")
coil_center_prio['3rd Priority'] = coil_center_prio['3rd Priority'].astype("object")
coil_center_prio['1st Priority'] = coil_center_prio['1st Priority'].astype("object")
coil_center_prio['2nd Priority'] = coil_center_prio['2nd Priority'].astype("object")

# Drop Standard and take Calculate Standard
coil_center_prio.drop(columns=['Standard'], inplace = True)
coil_center_prio['Standard'] = coil_center_prio['Calculate Standard'].str.lower()
coil_center_prio.drop(columns=['Calculate Standard'], inplace = True)
coil_center_prio["Customer_FG"] = (coil_center_prio['Customer'].astype(object)+"-"+coil_center_prio['FG Code'].astype(object))
# coil_center_prio.drop(columns=['BP code'],inplace=True)
coil_center_prio = coil_center_prio.apply(lambda col: col.fillna('') if col.dtype == 'object' else col.fillna(np.nan))

# Merge the DataFrames on the 'Key' column
merged_concatenated_po = pd.merge(concatenated_po, coil_center_prio, on='Customer_FG', how='left',suffixes=('_po', '_cc'))
# Fill NaN values in df1 columns with corresponding values from df2 columns
for col in coil_center_prio.columns:
    if col != 'Customer_FG':
        merged_concatenated_po[col + '_po'] = merged_concatenated_po[col + '_po'].combine_first(merged_concatenated_po[col + '_cc'])
        merged_concatenated_po.drop(columns=[col + '_cc'], inplace=True)

# Rename columns to remove suffixes
merged_concatenated_po.columns = [col.replace('_po', '') for col in merged_concatenated_po.columns]

# FORECAST
forecast = pd.read_excel(f"data/Forecast from Client_{month}.xlsx")
forecast.drop(columns=['(Do Not Modify) FCClient','(Do Not Modify) Row Checksum','(Do Not Modify) Modified On',
'FCID','(Do Not Modify) Modified On','(Do Not Modify) FCClient','(Do Not Modify) Row Checksum','BP Code'],inplace=True)
forecast.rename(columns={"Customer Short Name (BP Code) (Customer Master)":"Customer"},inplace=True)
forecast['Target Month'] = pd.to_datetime(forecast['Target Month'], format='%Y-%m')
forecast["Customer_FG"] = (forecast['Customer'].astype(object)+"-"+forecast['FG Code'].astype(object))

# Assume that current month = 1, fc1==2, fc2==3, fc3==4, 
forecast_fc1 = forecast[forecast['Forecast date'].dt.month == 2]
latest_indices_fc1 = forecast_fc1.groupby('Customer_FG')['Target Month'].idxmax()
latest_records_fc1 = forecast_fc1.loc[latest_indices_fc1,['Customer_FG','Forecast Qty']]
latest_records_fc1.rename(columns={'Forecast Qty':"fc1"},inplace=True)

forecast_fc2 = forecast[forecast['Forecast date'].dt.month == 3]
latest_indices_fc2 = forecast_fc2.groupby('Customer_FG')['Target Month'].idxmax()
latest_records_fc2 = forecast_fc2.loc[latest_indices_fc2,['Customer_FG','Forecast Qty']]
latest_records_fc2.rename(columns={'Forecast Qty':"fc2"},inplace=True)

forecast_fc3 = forecast[forecast['Forecast date'].dt.month == 4]
latest_indices_fc3 = forecast_fc3.groupby('Customer_FG')['Target Month'].idxmax()
latest_records_fc3 = forecast_fc3.loc[latest_indices_fc3,['Customer_FG','Forecast Qty','Maker','Spec','Thickness','Width','Length']]
latest_records_fc3.rename(columns={'Forecast Qty':"fc3"},inplace=True)

merged_forecast = latest_records_fc1.merge(latest_records_fc2, on='Customer_FG', how='outer').merge(latest_records_fc3, on='Customer_FG', how='outer')
merged_forecast['average FC'] = merged_forecast[['fc1', 'fc2', 'fc3']].mean(axis=1)

merged_po_forecast = pd.merge(merged_concatenated_po, merged_forecast, on='Customer_FG', how='left',suffixes=('_po', '_fc'))
# Fill NaN values in df1 columns with corresponding values from df2 columns
for col in merged_forecast.columns:
    if col not in ['Customer_FG','fc1','fc2','fc3','average FC']:
        merged_po_forecast[col + '_po'] = merged_po_forecast[col + '_po'].combine_first(merged_po_forecast[col + '_fc'])
        merged_po_forecast.drop(columns=[col + '_fc'], inplace=True)
# Rename columns to remove suffixes
merged_po_forecast.columns = [col.replace('_po', '') for col in merged_po_forecast.columns]

# DROP IF STANDARD OR 1ST EMPTY
merged_po_forecast = merged_po_forecast.dropna(subset=['Standard', '1st Priority'])

# SAVE
merged_po_forecast.drop_duplicates(inplace=True)
merged_po_forecast.rename(columns={"Customer":'customer_name',
                                    "FG Code":"fg_codes",
                                    "Maker":'maker',
                                    "Spec":'spec',
                                    "Thickness":'thickness',
                                    "Length":'length',
                                    "Width":'width',
                                    "Need Cut":'need_cut',
                                    "Standard":'standard',
                                    "Min":'Min_weight',
                                    "Max": "Max_weight",}, inplace = True)
merged_po_forecast['average FC'] = merged_po_forecast['average FC'].fillna(0)

merged_po_forecast.reset_index(inplace=True)
merged_po_forecast.rename(columns={'index': 'order_id'}, inplace=True)
merged_po_forecast["Min_MC_weight"] = merged_po_forecast.apply(min_mc_weight, axis=1)

# FIX STANDARD AFTER MERGING ALL TABLES
# Define conditions
conditions = [
    (merged_po_forecast['Min_MC_weight'] <= 3500) & (merged_po_forecast['standard'].isin(['na',pd.NA,''])),
    (merged_po_forecast['Min_MC_weight'] > 3500) & (merged_po_forecast['Min_MC_weight'] <= 7000) & (merged_po_forecast['standard'].isin(['na',pd.NA,'']))
]

# Define corresponding outputs
choices = ['small', 'medium']

# Apply the conditions
merged_po_forecast['standard'] = np.select(conditions, choices, default=merged_po_forecast['standard'])

#Fix for STEEL SHEET
merged_po_forecast.loc[(merged_po_forecast['length'] != '1219') & (merged_po_forecast['standard'] == 'na'), 'standard'] = 'small'

finish_order = ['order_id', 'customer_name', 'fg_codes', 'maker', 'spec', 'thickness',
       'width', 'length', 'PO Qt', 'Stockend Qt', 'need_cut', 'standard',
       'Min_weight', 'Max_weight', '1st Priority', '2nd Priority',
       '3rd Priority', 'MC Code', 'Customer_FG', 'fc1', 'fc2', 'fc3',
       'average FC', 'Min_MC_weight','Month']

merged_po_forecast = merged_po_forecast[finish_order]
merged_po_forecast.to_excel(f'data/finish_uat_{uat}.xlsx',index=False)

# MOTHER COIL - filter status + overlap MC code
mother_coil = pd.read_excel(f"data/MC Inventories_{month}.xlsx")
mother_coil.drop(columns=['(Do Not Modify) MC_Inventory','(Do Not Modify) Row Checksum','(Do Not Modify) Modified On',
'(Do Not Modify) Modified On','Inspection No','Grade','HTV Comment','Prod. Date','Customer Short Name (BP Code) (Customer Master)','Ngày tạo'],inplace=True)
mother_coil.rename(columns={
                            "Inventory ID":'inventory_id',
                            'Coil Center':'warehouse',
                            "Maker":'maker',
                            "Spec":'spec',
                            "Thickness":'thickness',
                            "Length":'length',
                            "Width":'width',
                            "Status":'status',
                            "Remark":'remark'},inplace=True)

mother_coil['weight'] = mother_coil["Weight"] - mother_coil["Usage Weight"].fillna(0)
mother_coil['weight_1219'] = mother_coil.apply(derive_mc_weight, axis=1)
mother_coil["MC Code"] = mother_coil.apply(lambda row: f"{row['maker']} {row['spec']} {row['thickness']:.2f}", axis=1)
mother_coil['receiving_date'] = pd.to_datetime(mother_coil['Receiving Date'])

mother_coil['receiving_date'] = mother_coil['receiving_date'].apply(lambda x: x.isoformat())
mother_coil.drop(columns=['Receiving Date'],inplace=True)

filtered_mother_coil = mother_coil[mother_coil['status'].isin(['M:RAW MATERIAL', 'Z:SEMI MCOIL', 'R:REWIND'])]
filtered_mother_coil=filtered_mother_coil.reset_index(drop=True)
filtered_mother_coil = filtered_mother_coil.drop_duplicates(subset='inventory_id', keep='first')
column_order = ['inventory_id', 'warehouse', 'FG Code', 'maker', 'spec', 'thickness',
       'length', 'width', 'Weight', 'Qty', 'Usage Status', 'Usage Weight',
       'status', 'remark',  'weight', 'weight_1219',
       'MC Code', 'receiving_date']
filtered_mother_coil = filtered_mother_coil[column_order]
filtered_mother_coil.to_excel(f"data/mother_coil_uat_{month}.xlsx", index= False)
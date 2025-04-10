# GET OVERLAP MATERIAL_PROPERTIES
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import math
import copy
import os
import re
import logging

global stock_ratio_default
global low_count
global overloaded_total_need_cut
global overloaded_total_fg_codes

from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# PARAMETERS
added_stock_ratio = float(os.getenv('ADDED_STOCK_RATIO', '0.5'))
stock_ratio_default = float(os.getenv('STOCK_RATIO_DEFAULT', '0.5'))

low_count = int(os.getenv('LOW_COUNT', '4'))
overloaded_total_need_cut = float(os.getenv('OVERLOADED_TOTAL_NEEDCUT', '29000'))
overloaded_total_fg_codes = float(os.getenv('OVERLOADED_TOTAL_FG_CODES', '15'))
sub_overloaded_total_fg_codes = float(os.getenv('SUB_OVERLOADED_TOTAL_FG_CODES', '9'))

# SETUP
finish_key = 'order_id'
finish_columns = ["customer_name", "fg_codes",
                "width", "need_cut", "standard", 'cut_standard',
                "fc1", "fc2", "fc3", "average FC",
                "1st Priority", "2nd Priority", "3rd Priority",'coil_center_priority',
                "Min_weight", "Max_weight", "Min_MC_weight"]
forecast_columns = ["fc1", "fc2", "fc3", "average FC"]

stock_key = "inventory_id"
stock_columns = ['receiving_date',"width", "weight","weight_1219",'warehouse',"status",'remark']

def find_spec_type(spec, spec_type_df):
    try:
        type = spec_type_df[spec_type_df['spec']==spec]['type'].values[0]
    except IndexError:
        type = "All"
    return type

def find_materialprops_and_jobs(fin_file_path,mc_file_path):
    """_Find overlapped material code with need cut < 0_

    Args:
        fin_file_path (_type_): _description_
        mc_file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # LOAD DATA
    fin_df = pd.read_excel(fin_file_path)
    mc_df = pd.read_excel(mc_file_path)

    has_need_cut_df = fin_df[fin_df['need_cut'] < -10]
    has_need_cut_df['materialprops'] = has_need_cut_df.apply(
        lambda row: f"{row['maker']}+{row['spec']}+{row['thickness']}", axis=1
    )

    fin_materialprops = has_need_cut_df['materialprops'].unique()
    mc_df.loc[:, 'materialprops'] = (mc_df['maker'] + "+" + mc_df['spec']+ "+" + mc_df['thickness'].astype(str))
    mc_materialprops = mc_df['materialprops'].unique()

    # Find the intersection (overlapping values)
    overlap = set(mc_materialprops) & set(fin_materialprops)

    # Convert the result back to a list if needed
    overlapped_materialprops = sorted(list(overlap))
    n_jobs = len(overlapped_materialprops)
    
    need_cut_df_filtered = has_need_cut_df[has_need_cut_df['materialprops'].isin(overlapped_materialprops)]
    need_cut_df_grouped = need_cut_df_filtered.groupby('materialprops')['need_cut'].sum().reset_index().sort_values(by='need_cut', ascending=False) #needcut am
    materialprops = need_cut_df_grouped['materialprops'].unique()
    
    return materialprops, n_jobs

def merge_standards_with_low_count(mdf,logger):
    """_Rules_

    """
    # Group by the 'standard' column and count instances
    standard_counts = mdf.groupby("standard").size().reset_index(name="count")
    logger.info(standard_counts)
    # standard_counts_dict = standard_counts.set_index('standard')['count'].to_dict()
    all_standards = standard_counts['standard'].to_list()
    standards_with_low_count = standard_counts[standard_counts["count"] <= low_count]['standard'].to_list()
    
    if len(all_standards) == 2 and ("small" in all_standards and "big" in all_standards):
        pass #ko merge 2 group nay duoc
    elif len(all_standards) == 2 and "medium" in all_standards:
        if standards_with_low_count and "small" in all_standards:
            mdf.loc[:, 'cut_standard'] = mdf['cut_standard'].replace({"small": "medium"})
            logger.warning("merge small to medium")
        elif standards_with_low_count and "big" in all_standards:
            mdf.loc[:, 'cut_standard'] = mdf['cut_standard'].replace({"medium": "big"})
            logger.warning("merge med to big")
        else: pass # no group low count
        
    if len(all_standards) == 3:
        if standards_with_low_count and "big" not in standards_with_low_count:
            mdf.loc[:, 'cut_standard'] = mdf['cut_standard'].replace({"small": "medium"})
            logger.warning("merge small to medium")
        elif standards_with_low_count and "big" in standards_with_low_count:
            if "small" in standards_with_low_count:
                logger.warning('div medi them merge')
                mdf.loc[(mdf['cut_standard'] == 'medium') & (mdf['Min_MC_weight'] >= 5000), 'cut_standard'] = 'big'
                mdf.loc[(mdf['cut_standard'] == 'medium') & (mdf['Min_MC_weight'] < 5000), 'cut_standard'] = 'small'
            else: # ko co small trong low-count
                mdf.loc[:, 'cut_standard'] = mdf['cut_standard'].replace({"medium": "big"})
                logger.warning("merge med to big")
    
    return mdf

def find_coil_center_with_standard(df):
    """ DF only have need-cut < 0
    """
    all_coil_centers = df[["1st Priority", "2nd Priority", "3rd Priority"]].values.flatten()
        
    # Find unique coil center
    unique_strings = sorted(set(x for x in all_coil_centers if isinstance(x, str) and not x.replace('.', '', 1).isdigit()))
       
    return unique_strings
    
def div(numerator, denominator):
    def division_operation(row):
        if row[denominator] == 0:
            if row[numerator] > 0:
                return np.inf
            elif row[numerator] < 0:
                return -np.inf
            else:
                return np.nan  # Handle division by zero with numerator equal to zero
        else:
            return float(row[numerator] / row[denominator])
    return division_operation

def create_finish_dict(finish_df):
    
    # Width - Decreasing// need_cut - Descreasing // Average FC - Increasing
    sorted_df = finish_df.sort_values(by=['need_cut','width'], ascending=[True,False]) # need cut van dang am

    sorted_df[["Min_weight", "Max_weight"]] = sorted_df[["Min_weight", "Max_weight"]].fillna("")
    
    # Fill NaN values in specific columns with the average, ignoring NaN
    sorted_df[forecast_columns] = sorted_df[forecast_columns].fillna(0)

    # Initialize result dictionary - take time if the list long
    finish = {f"F{int(row[finish_key])}": {column: row[column] for 
                                          column in finish_columns} for 
                                          _, row in sorted_df.iterrows()}
    return finish

def create_stocks_dict(stock_df):
    # Sort data according to the priority of FIFO
    # stock_df['receiving_date'] = stock_df['receiving_date'].apply(lambda x: x.strftime('%m-%Y-%d'))
    sorted_df = stock_df.sort_values(by=['warehouse','receiving_date','weight'], ascending=[True,True, True])
    
    # Set the index of the DataFrame to 'inventory_id'
    sorted_df.set_index(stock_key, inplace=True)
    
    # Convert DataFrame to dictionary
    sorted_df[stock_columns] = sorted_df[stock_columns].fillna("")
    stocks = sorted_df[stock_columns].to_dict(orient='index')
   
    return stocks

def filter_by_materialprops(file_path, materialprops):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    df = df.reset_index(drop=True)
    filtered_df = df[
                    (df["spec"] == materialprops["spec"]) & 
                    (df["thickness"] == materialprops["thickness"]) &
                    (df["maker"] == materialprops["maker"])
                    ]
    return filtered_df

def filter_finish_by_stock_ratio(file_path, materialprops):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    filtered_df = df[
                    # (df["customer_name"] == materialprops["customer"]) & 
                    (df["spec"] == materialprops["spec"]) & 
                    (df["thickness"] == materialprops["thickness"]) &
                    (df["maker"] == materialprops["maker"])
                    ]
    # Ensure filtered_df is a copy if it's a slice
    filtered_df = filtered_df.copy()

    # Assign the calculated 'stock_ratio' column
    filtered_df['stock_ratio'] = filtered_df.apply(lambda row: row['need_cut'] / (row['average FC']+1), axis=1)

    df = filtered_df[filtered_df['stock_ratio'] < float(stock_ratio_default)] # chon ca nhung need cut ko am
    
    return df

# INPUT
# uat = int(os.getenv('UAT'))
uat = int(1149)
uat_list = [1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161]
for uat in uat_list:
    print(f"<< Process UAT {uat} >> ")
    # DATA
    fin_file_path = f"data/finish_uat_{uat}.xlsx"
    df = pd.read_excel(fin_file_path)
    month = df['Month'][0]
    mc_file_path = f"data/mother_coil_uat_{month}.xlsx"
    spec_type_df = pd.read_csv((os.getenv('MASTER_SPECTYPE')))


    ### START DATA PROCESSING 
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'log/dataprocessing-{uat}.log', level=logging.INFO, 
                        format='%(levelname)s - %(message)s')

    logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.info('*** DATA PROCESSING ***')
    logger.info(f'*** DATE {datetime.today().strftime('%Y-%m-%d')} ***')

    materialprops, n_jobs = find_materialprops_and_jobs(fin_file_path =fin_file_path, mc_file_path = mc_file_path)
    logger.info(f"Total Job:{n_jobs}")

    job_list = {
        'slit_request_id': uat,
        'number of job': n_jobs,
        'jobs':[]
    }

    finish_list = {
        'slit_request_id': uat,
        'materialprop_finish':{}
    }
    stocks_list = {
        'slit_request_id': uat,
        'materialprop_stock':{}
    }
            
    for i, materialprop in enumerate(materialprops):
        logger.info(f"MC CODE {i} {materialprop}")
        # LOAD JOB
        materialprop_split = materialprop.split("+")
        maker = materialprop_split[0]
        spec = materialprop_split[1]
        thickness = round(float(materialprop_split[2]),2)
        MC_code = maker + " " + spec  + " " + str(thickness)
        MATERIALPROPS = {
            "spec" : spec,
            "thickness" : thickness,
            "maker"     : maker,
            "type"      : find_spec_type(spec,spec_type_df),
            "code"      : MC_code
        }
        
        finish_list['materialprop_finish'][materialprop] = {'materialprop': MATERIALPROPS, 'group':[]}
        
        # Filter FINISH by MATERIAL_PROPERTIES 
        materialprop_finish_df = filter_finish_by_stock_ratio(fin_file_path, MATERIALPROPS)
        
        # Take unique list with need-cut < 0
        materialprop_finish_df['cut_standard'] = materialprop_finish_df["standard"].copy()
        neg_finish_df = materialprop_finish_df[materialprop_finish_df['need_cut'] < 0]
    
        # Add need-cut>0 to defined-std group as above
        materialprop_finish_df.loc[:, 'stock_ratio'] = materialprop_finish_df.apply(div('need_cut', 'average FC'), axis=1)
        added_pos_finish_df = materialprop_finish_df[
            (materialprop_finish_df['need_cut'] >= 0) & 
            (materialprop_finish_df['stock_ratio'] <= float(added_stock_ratio))
        ]# chon ca nhung need cut ko am
        
        #new DF with neg and pos needcut and new cut group
        finish_df= pd.concat([neg_finish_df, added_pos_finish_df], ignore_index=True)
        finish_df['coil_center_priority'] = (
                                            finish_df['1st Priority'].astype(object).fillna('') + "-" + 
                                            finish_df['2nd Priority'].astype(object).fillna('') + "-" + 
                                            finish_df['3rd Priority'].astype(object).fillna('')
                                            )
        # Find how many group of
        st1_coil_center_priority_list = finish_df['1st Priority'].unique().tolist()
        merged_standard_by_1stpriority_df = pd.DataFrame(columns=finish_df.columns)
        for coil_center in st1_coil_center_priority_list:
            logger.info(f"IN COIL CENTER {coil_center}")
            fin_df = finish_df[finish_df['1st Priority'] == coil_center]
            merged_df = merge_standards_with_low_count(fin_df,logger) #'cut-standard la cai duoc merge
            merged_standard_by_1stpriority_df = pd.concat([merged_standard_by_1stpriority_df, merged_df], axis=0, ignore_index=True)
        
        # RE ORDER BY PREFERENCE
        custom_order = ["big","small","medium"]
        unique_standard = merged_standard_by_1stpriority_df['cut_standard'].unique()
        defined_standard = sorted(set(x for x in unique_standard if isinstance(x, str)), key=lambda x: custom_order.index(x))
        
        finish_list['materialprop_finish'][materialprop] = {'materialprop': MATERIALPROPS, 'group':[]}
        
        # Create FINISH list with customer group
        for cust_gr in defined_standard:
            filtered_finish_df = merged_standard_by_1stpriority_df[merged_standard_by_1stpriority_df['cut_standard']==cust_gr]
            unique_coil_center = find_coil_center_with_standard(filtered_finish_df)
            finish = create_finish_dict(filtered_finish_df)
            
            finish_list['materialprop_finish'][materialprop]['group'].append(
                {cust_gr:{
                        'coil_center_order': unique_coil_center,
                        'FG_set':finish}})
        
        # Filter STOCKS by MATERIAL_PROPERTIES
        materialprop_stocks_df = filter_by_materialprops(mc_file_path, MATERIALPROPS)
        stocks = create_stocks_dict(materialprop_stocks_df)
        # Add to stocks lists
        stocks_list['materialprop_stock'][materialprop] = {'materialprop': MATERIALPROPS, 'stocks': stocks}
        
        # Add to JOB lists
        job_list['jobs'].append({"job": i,'materialprop': materialprop,"code": MC_code, 'available_stock_qty': {} ,'tasks':{}})
        current_job = job_list['jobs'][i]
        warehouse_qty = {}
        
        # SUM STOCK BY WAREHOUSE
        wh_list = materialprop_stocks_df['warehouse'].unique().tolist()
        for wh in wh_list:
            wh_stock = materialprop_stocks_df[materialprop_stocks_df['warehouse'] == wh]
            sum_wh_stock = wh_stock['weight'].sum()
            warehouse_qty[wh] = float(sum_wh_stock)
            
        current_job['available_stock_qty'] = dict(sorted(warehouse_qty.items(), key=lambda item: item[1], reverse=False))
            
        sub_job_operator = {}
        total_need_cut = 0
        
        # Finished goods list by cut group
        for std_group in defined_standard:
            cust_df = merged_standard_by_1stpriority_df[(merged_standard_by_1stpriority_df['cut_standard']==std_group)&(merged_standard_by_1stpriority_df['need_cut']<0)]
            len_fg_code = len(cust_df['fg_codes'])
            
            needcut = sum(cust_df['need_cut']) * -1
            sub_job_operator[std_group] = {"total_need_cut": float(needcut),
                                        "len_fg_codes": len_fg_code}
            total_need_cut += needcut
            
        current_job['tasks'] = copy.deepcopy(sub_job_operator)
        current_job['total_need_cut'] = total_need_cut

    with open(f'jobs/finish-list-uat-{uat}.json', 'w') as json_file:
        json.dump(finish_list, json_file, indent=2)
    with open(f'jobs/job-list-uat-{uat}.json', 'w') as json_file:
        json.dump(job_list, json_file, indent=2)
    with open(f'jobs/stocks-list-uat-{uat}.json', 'w') as stocks_file:
        json.dump(stocks_list, stocks_file, indent=2)
    print("<< DONE >> ")

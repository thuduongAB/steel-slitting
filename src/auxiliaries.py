# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import copy
import math
import re
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# LOAD ENV PARAMETERS & DATA
global max_bound, mc_ratio, stock_ratio, bound_step
global added_stock_ratio_avg_fc
global stop_stock_ratio, stop_needcut_wg
global overloaded_total_need_cut, overloaded_total_fg_codes, max_coil_weight

max_coil_weight         = float(os.getenv('MAX_WEIGHT_MC_DIV_2', '7000'))
max_bound               = float(os.getenv('MAX_BOUND', '3.0'))
added_stock_ratio_avg_fc = [3000, 2000, 1000, 500 ,100]
bound_step = 0.5

# uat = int(1131)
low_count = int(os.getenv('LOW_COUNT', '5'))
stop_stock_ratio = float(os.getenv('ST_RATIO', '-0.03'))
stop_needcut_wg         = -float(os.getenv('STOP_NEEDCUT_WG', '90'))
overloaded_total_need_cut = float(os.getenv('OVERLOADED_TOTAL_NEEDCUT', '29000'))
overloaded_total_fg_codes = float(os.getenv('OVERLOADED_TOTAL_FG_CODES', '15'))
sub_overloaded_total_fg_codes = float(os.getenv('SUB_OVERLOADED_TOTAL_FG_CODES', '9'))

# HELPERS
def create_warehouse_order(finish):
    finish_dict_value = dict(list(finish.values())[0])
    coil_center_1st = finish_dict_value['1st Priority']
    if finish_dict_value['2nd Priority'] == 'x':
        coil_center_2nd = None
    else:
        coil_center_2nd = finish_dict_value['2nd Priority']
    
    if finish_dict_value['3rd Priority'] == 'x':
        coil_center_3rd = None
    else:
        coil_center_3rd = finish_dict_value['3rd Priority']
    
    warehouse_order = [coil_center_1st,coil_center_2nd,coil_center_3rd]
    return warehouse_order

def filter_stocks_by_wh(stock, warehouse):
    # Filter and order the dictionary
    return {k: v for k, v in stock.items() if v['warehouse'] == warehouse}

def filter_stocks_by_group_standard(finish, stocks):
    # Find coil-standard group
    first_item = list(finish.items())[0]
    customer_gr = first_item[1]['cut_standard']
    print(f"customer std {customer_gr}")
    if customer_gr == "medium":
        # Filter out stock < min MC weight
        mc_weights = [round(v['Min_MC_weight']) if v is not None and not math.isnan(v['Min_MC_weight']) else 0 for _, v in finish.items()]
        # Convert to a NumPy array and filter out NaNs
        mc_weights = np.array(mc_weights, dtype=float)
        mc_weights = mc_weights[~np.isnan(mc_weights)]  # Remove NaN values
        # Check if the list is empty after removing NaNs
        if np.percentile(mc_weights, 50) == 0:
            min_mc_weight = 3500
        else:
            min_mc_weight = np.percentile(mc_weights, 50)
            if min_mc_weight > 5000:
                min_mc_weight = 5000
            else: pass
        filtered_stocks = {k: v for k, v in stocks.items() if v['weight_1219'] >= min_mc_weight}
    elif customer_gr == "big" and len(list(finish.items())) >2:
        # Filter out stock < 7000
        filtered = {k: v for k, v in stocks.items() if v['weight_1219'] >= max_coil_weight}
        if len(filtered) == 0:
            filtered_stocks = {k: v for k, v in stocks.items() if v['weight_1219'] >= 4000}
        else:
            filtered_stocks = {k: v for k, v in stocks.items() if v['weight_1219'] >= max_coil_weight}
    else:
        # No filter for small group/or too few FG
        filtered_stocks = copy.deepcopy(stocks)
    return filtered_stocks

def partite_stock(stocks, allowed_cut_weight, group, medium_mc_weight):
    """Take amount of stock equals 2.5 to 3.0 need cut
    Args:
        stocks (dict): _description_
        allowed_cut_weight (float): _description_
    Returns:
        sub_stocks: dict
    """
    try:
        max_mc_wg = max([v['weight_1219'] for _,v in stocks.items()])
    except ValueError: #empty
        max_mc_wg = 5000
    
    if "big" in group:
        filtered_stocks = {k: v for k, v in stocks.items() if v['weight_1219'] >= max_coil_weight}
        if len(filtered_stocks) < 2:
            filtered_stocks = {k: v for k, v in stocks.items() if v['weight_1219'] >= 6000}
        else: pass
        # Sort weight ascending and receiving_date ascending
        filtered_stocks = dict(sorted(filtered_stocks.items(), key=lambda x: (datetime.fromisoformat(x[1]['receiving_date']).timestamp(),x[1]['weight'])))
    elif ("medium" in group) and (medium_mc_weight*2 > max_mc_wg):
        # Sort weight ascending and receiving_date ascending
        filtered_stocks = dict(sorted(stocks.items(), key=lambda x: (datetime.fromisoformat(x[1]['receiving_date']).timestamp(),x[1]['weight'])))
    elif ("medium" in group) and (medium_mc_weight*2 <= max_mc_wg):
        # Sort weight descending and receiving_date ascending
        filtered_stocks = dict(sorted(stocks.items(), key=lambda x: (-datetime.fromisoformat(x[1]['receiving_date']).timestamp(),x[1]['weight']), reverse= True))
    else: 
        # Sort weight descending and receiving_date ascending
        filtered_stocks = dict(sorted(stocks.items(), key=lambda x: (-datetime.fromisoformat(x[1]['receiving_date']).timestamp(),x[1]['weight']), reverse= True))
    
    sub_stocks = {}
    accumulated_weight = 0
    
    for s, sinfo in filtered_stocks.items():
        accumulated_weight += sinfo['weight']
        if allowed_cut_weight * 1.25 <= accumulated_weight:
            if len(sub_stocks) <= 1:
                sub_stocks[s] = {**sinfo}
            else:
                break
        else:
            sub_stocks[s] = {**sinfo}
    res = dict(sorted(sub_stocks.items(), key=lambda x: (x[1]['weight']), reverse=True))
 
    return res

def partite_finish(finish, stock_ratio):
    """_Select more FG codes (finish) below the indicated stock ratio to reduce trim loss _

    Args:
        finish (_type_): _description_
        stock_ratio (_type_): _description_

    Returns:
        sub_finish: the proportion of finish has the stock ratio as required
    """
    partial_pos_finish = {}
    for avg_fc in added_stock_ratio_avg_fc:
        print(f"range forecast {avg_fc}")
        for f, finfo in finish.items():
            average_fc = max(finfo.get('average FC', 0), 1)
            fg_ratio = finfo['need_cut'] / average_fc
            if (0 <= fg_ratio <= stock_ratio and round(finfo['average FC']) >= avg_fc):
                partial_pos_finish[f] = finfo.copy()
        
        if len(partial_pos_finish) >= 1:
            default_avg_fc = avg_fc
            break
        else:
            continue
    
    sub_finish = {}
    for f, finfo in finish.items():
        # Ensure 'average FC' is treated as 1 if it's 0
        average_fc = max(finfo.get('average FC', 0), 1)
        fg_ratio = finfo['need_cut'] / average_fc
        
        # Safely retrieve values with default fallback
        default_avg_fc = default_avg_fc if 'default_avg_fc' in locals() else 500
    
        # Check conditions for partial finishes
        if (
            fg_ratio < 0 
            or (0 <= fg_ratio <= stock_ratio and round(average_fc) >= default_avg_fc)
        ):
            sub_finish[f] = finfo.copy()

    res = dict(sorted(sub_finish.items(), key=lambda x: (x[1]['need_cut'],x[1]['width']))) # need_cut van dang am sort ascending 
    return res

def move_finish(finish, over_cut):
    """Only take the FG with overcut """
    # Update need cut
    for f in over_cut.keys(): # neu f khong co trong over_cut thi tuc la finish[f] chua duoc xu ly
        try: # finish stock ratio < -2% removed in previous run, still in overcut
            finish[f]['need_cut'] = over_cut[f] # finish need cut am
        except KeyError:
            pass
            
    # Take only finish with negative need_cut
    mo_finish = {k: v for k, v in finish.items() 
                 if v['need_cut']/(v['average FC']+1) < stop_stock_ratio 
                 or v['need_cut']< stop_needcut_wg}
    return mo_finish
    
def refresh_stocks(taken_stocks, taken_stocks_dict, stocks):
    if not taken_stocks:
        remained_stocks = stocks
    else:
        taken_div_og = []
        taken_merge_og = []
        for s in taken_stocks:
            if str(s).__contains__("-Di"):
                og_s = str(s).split("-Di")[0]
                taken_div_og.append(og_s)
            elif str(s).__contains__("-Me"):
                og_s = str(s).split("-Me")[0]
                taken_merge_og.append(og_s)
        taken_stocks = taken_stocks + taken_div_og + taken_merge_og
    
        # UPDATE stocks
        div_stock_list = list(set(taken_div_og))
        for stock_key in div_stock_list:
            for i in range(3):
                try:
                    wg = taken_stocks_dict[f'{stock_key}-Di{i+1}']
                    stocks[f'{stock_key}-Di{i+1}'] = stocks[stock_key]
                    stocks[f'{stock_key}-Di{i+1}'].update({'weight': wg})
                    stocks[f'{stock_key}-Di{i+1}'].update({'status':"R:REWIND"})
                except KeyError: # already update in someround - the stock ID is the remained
                    pass
        
        meg_stock_list = list(set(taken_merge_og))
        for stock_key in meg_stock_list:
            try:
                taken_wg = taken_stocks_dict[f"{stock_key}-Me"]
                original_wg = stocks[stock_key]['weight']
                # taken MC
                stocks[f'{stock_key}-Me'] = stocks[stock_key]
                stocks[f'{stock_key}-Me'].update({'weight': taken_wg})
                stocks[f'{stock_key}-Me'].update({'status':"R:REWIND"})
                if original_wg - taken_wg > 0:
                    # remaining MC
                    stocks[f'{stock_key}-Le'] = stocks[stock_key]
                    stocks[f'{stock_key}-Le'].update({'weight': original_wg-taken_wg}) #left weight co the tien ve 0
                    stocks[f'{stock_key}-Le'].update({'status':"R:REWIND"})
                else: pass
            except KeyError: # already update in someround - the stock ID is the remained
                pass 
                 
        remained_stocks = {
                s: {**s_info}
                for s, s_info in stocks.items()
                if s not in taken_stocks
            }
    return remained_stocks

def check_subgroup_fg(df, sum_need_cut, standard):
    """ DF only have need-cut < 0
    """
    # Create new empty df to save split group
    appended_df = pd.DataFrame(columns=df.columns)
    std_dict = {}
    
    # Sort 2 sides
    sorted_desc = df.sort_values(by="width", ascending=False)  # Largest first
    sorted_asc = df.sort_values(by="width", ascending=True)   # Smallest first
    count = df.shape[0]

    # Interleave rows
    interleaved_rows = []
        
    # chia thanh nhieu nhom neu co qua nhieu fg code
    if count == 0:
        n = 0
        pass

    elif 0 < count <= overloaded_total_fg_codes and float(sum_need_cut) < overloaded_total_need_cut:
        n = 1
        if not df.empty and not df.isna().all().all():
            appended_df = pd.concat([appended_df, df], ignore_index=True)
    elif float(sum_need_cut) >= overloaded_total_need_cut:
        if round(count/2) >= low_count: no_each_group = round(count/2)
        else:  no_each_group = sub_overloaded_total_fg_codes
        
        n = math.ceil(count/no_each_group) #n group
        num_it = int(count//n) # each group have num_item
    else:
        no_each_group = overloaded_total_fg_codes
        n = math.ceil(count/no_each_group) #n group
        num_it = int(count//n) # each group have num_item
       
    # Create interleaved df
    gr2 = int(count//2)
    for m in range(gr2):
        interleaved_rows.append(sorted_desc.iloc[m])  # Add largest
        interleaved_rows.append(sorted_asc.iloc[m])  # Add smallest
    # Handle odd-length DataFrame (optional, here for completeness)
    if count % 2 != 0:
        interleaved_rows.append(sorted_desc.iloc[gr2])
    # Convert interleaved rows to a DataFrame
    interleaved_df = pd.DataFrame(columns=df.columns,data=interleaved_rows)
    # Split into two DataFrames
    interleaved_df = interleaved_df.reset_index(drop=True)
    
    if n > 1:
        for i in range(n):
            if i != n-1:
                interleaved_df.loc[num_it*(i):num_it*(i+1),'cut_standard'] = f"{standard}{str((i+1))}"    # Rows from `num` onwards
            else:
                interleaved_df.loc[num_it*(i):,'cut_standard'] = f"{standard}{str((i+1))}"
                std_dict[standard] = i+1
    else: pass
    if not interleaved_df.empty and not interleaved_df.isna().all().all():
        appended_df = pd.concat([appended_df, interleaved_df], ignore_index=True)
    
    return appended_df, std_dict

def dividing_to_subgroup(coilcenter_finish_df,standard_group,MATERIALPROPS):
    # OVERLOAD-CHECK
    total_need_cut = coilcenter_finish_df[coilcenter_finish_df['need_cut'] < 0]['need_cut'].sum()
    coilcenter_subgroup_df, std_dict = check_subgroup_fg(coilcenter_finish_df, total_need_cut, standard_group)
    defined_standard = coilcenter_subgroup_df['cut_standard'].unique().tolist()
    
    try: n_group = std_dict[standard_group] # for empty std_dict - no subgroup
    except KeyError: n_group = 0
    
    sub_finish_list = {'materialprop': MATERIALPROPS, 'subgroup':[]}
    
    if n_group > 1:
        desired_order = ["small","small1","small2","small3",'small4','small5',
                        "big","big1","big2",'big3','big4','big5',
                        "medium","medium1","medium2",'medium3','medium4','medium5']
        
        # Reorder dynamically based on the desired order
        ordered_standard = sorted(defined_standard, key=lambda x: desired_order.index(x))
        
        # Create FINISH list with customer group
        for cust_gr in ordered_standard:
            coilcenter_sub_finish_df = coilcenter_subgroup_df[coilcenter_subgroup_df['cut_standard']==cust_gr]
            coilcenter_sub_finish = create_finish_dict(coilcenter_sub_finish_df)
            sub_finish_list['subgroup'].append({cust_gr: coilcenter_sub_finish})
    else: 
        coilcenter_sub_finish = create_finish_dict(coilcenter_finish_df)
        sub_finish_list['subgroup'].append({standard_group: coilcenter_sub_finish})
    
    return sub_finish_list                 

def divide_to_subgroup(df, defined_std, std_dict): 
    # DF by material code
    df = df.copy()
    custom_order =["small","medium","big"]
    df["standard"] = pd.Categorical(df["standard"], categories=custom_order, ordered=True)
    df['cut_standard'] = df["standard"]
    
    # Step 1:
    sub_group = [item for item in defined_std if re.search(r'\d', item)]
    re_group = [item for item in defined_std if not re.search(r'\d', item)] #remained
    
    # Step 2: Remove numbers from the filtered items
    cleaned_list = [re.sub(r'\d', '', item) for item in sub_group]

    # Step 3: Create a unique list
    unique_sub_list = list(set(cleaned_list))
    
    if not unique_sub_list: # new ko co sub group
        appended_df = copy.deepcopy(df)
    else:
        if len(re_group) == 0:
            # Create new empty df to save split group
            appended_df = pd.DataFrame(columns=df.columns)
        else:
            appended_df = copy.deepcopy(df[df['cut_standard']==str(re_group[0])])
            
        for std in unique_sub_list:
            # default: chia doi list
            filtered_df = df[df['cut_standard'] == std]
            sorted_desc = filtered_df.sort_values(by="width", ascending=False)  # Largest first
            sorted_asc = filtered_df.sort_values(by="width", ascending=True)   # Smallest first
            count = filtered_df.shape[0]
            
            # Interleave rows
            interleaved_rows = []
            
            # chia thanh nhieu nhom neu co qua nhieu fg code
            num = int(count//2)
            
            for r in range(num):
                interleaved_rows.append(sorted_desc.iloc[r])  # Add largest
                interleaved_rows.append(sorted_asc.iloc[r])  # Add smallest
            # Handle odd-length DataFrame (optional, here for completeness)
            if count % 2 != 0:
                interleaved_rows.append(sorted_desc.iloc[num])
                
            # Convert interleaved rows to a DataFrame
            interleaved_df = pd.DataFrame(columns=df.columns,data=interleaved_rows)
            
            n= std_dict[std] +1 # number of group
            num_it = int(count//n) # item per group
            
            # Split into two DataFrames
            interleaved_df = interleaved_df.reset_index(drop=True)
            for i in range(n):
                if i != n-1:
                    interleaved_df.loc[num_it*(i):num_it*(i+1),'cut_standard'] = f"{std}{str((i+1))}"    # Rows from `num` onwards
                else:
                    interleaved_df.loc[num_it*(i):,'cut_standard'] = f"{std}{str((i+1))}"
                    
            if not interleaved_df.empty and not interleaved_df.isna().all().all():
                appended_df = pd.concat([appended_df, interleaved_df], ignore_index=True)
        
    return appended_df

def create_finish_dict(finish_df):
    # SETUP
    finish_key = 'order_id'
    finish_columns = [
                    "customer_name", "fg_codes",
                    "width", "need_cut", "standard", 'cut_standard',
                    "fc1", "fc2", "fc3", "average FC",
                    "1st Priority", "2nd Priority", "3rd Priority",
                    'coil_center_priority',
                    "Min_weight", "Max_weight", "Min_MC_weight"
                    ]
    forecast_columns = ["fc1", "fc2", "fc3", "average FC"]
    
    # Width - Decreasing// need_cut - Descreasing // Average FC - Increasing
    sorted_df = finish_df.sort_values(by=['need_cut','width'], ascending=[True,False]) # need cut van dang am

    sorted_df[["Min_weight", "Max_weight"]] = sorted_df[["Min_weight", "Max_weight"]].fillna("")
    
    # Fill NaN values in specific columns with the average, ignoring NaN
    # sorted_df[forecast_columns] = sorted_df[forecast_columns].apply(lambda x: x.fillna(x.mean()), axis=1)
    sorted_df[forecast_columns] = sorted_df[forecast_columns].fillna(0)

    # Initialize result dictionary - take time if the list long
    try:
        finish = {f"{str(row[finish_key])}": {column: row[column] for 
                                            column in finish_columns} for 
                                            _, row in sorted_df.iterrows()}
    except TypeError or KeyError:
        finish = {str(row[finish_key]): {column: row[column] for 
                                            column in finish_columns} for 
                                            _, row in sorted_df.iterrows()}
        
    return finish

def compare_and_add_coilcenter(list1, list2):
    # Iterate over items in list2
    for item in list2:
        # If item is not in list1, add it to the beginning of list1
        if item not in list1:
            list1.insert(0, item)
    return list1

# SAVE FILE
def save_to_json(filename, data):
    with open(filename, 'w') as solution_file:
        json.dump(data, solution_file, indent=2)

def flattern_data(data):
    # Flattening the nested structure
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['cuts']}
        for cut, line in item['cuts'].items():
            flattened_item = {**common_data, 
                                    'cuts': cut, 
                                    'lines': line,
                                    'Customer': item['Customer'][cut],
                                    'fg_code':item['fg_code'][cut],
                                    'FG width': item['FG width'][cut],
                                    'FG Weight': item['FG Weight'][cut],
                                    'standard': item['standard'][cut],
                                    'Min weight': item['Min weight'][cut],
                                    'Max weight':item['Max weight'][cut],
                                    'explanation': item['explanation'][cut],
                                    'remarks': item['remarks'][cut],
                                    'average_fc': item['average_fc'][cut],
                                    '1st Priority':item['1st Priority'][cut]
                                    }
            flattened_data.append(flattened_item)
    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def transform_to_df(data):
    # Flatten the data
    flattened_data = []
    for item in data:
        common_data = {k: v for k, v in item.items() if k not in ['count','count_cut','cuts']}
        for cut, line in item['cuts'].items():
            if line > 0:
                try:
                    flattened_item = {**common_data, 
                                    'cuts': cut, 
                                    'lines': line,
                                    'Customer': item['Customer'][cut],
                                    'fg_code':item['fg_code'][cut],
                                    'FG width': item['FG width'][cut],
                                    'FG Weight': item['FG Weight'][cut],
                                    'standard': item['standard'][cut],
                                    'Min weight': item['Min weight'][cut],
                                    'Max weight':item['Max weight'][cut],
                                    'remarks': item['remarks'][cut],
                                    'average_fc': item['average_fc'][cut],
                                    '1st Priority':item['1st Priority'][cut]
                                    }
                except TypeError:
                    flattened_item = {**common_data, 
                                    'cuts': cut, 
                                    'lines': line,
                                    'Customer': item['Customer'][cut],
                                    'fg_code':item['fg_code'][cut],
                                    'FG width': item['FG width'][cut],
                                    'FG Weight': item['FG Weight'][cut],
                                    'standard': item['standard'][cut],
                                    'Min weight': item['Min weight'][cut],
                                    'Max weight':item['Max weight'][cut],
                                    'remarks': "",
                                    'average_fc': item['average_fc'][cut],
                                    '1st Priority':item['1st Priority'][cut]
                                    }
                flattened_data.append(flattened_item)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    return df

def clean_filename(filename):
    # Define a regular expression pattern for allowed characters (letters, numbers, dots, underscores, and dashes)
    # Replace any character not in this set with an underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)

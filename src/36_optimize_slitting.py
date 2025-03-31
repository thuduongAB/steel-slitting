# USER FOR BOUND <= 3 - DUALITY, LINEAR, REWIND CASES
import pandas as pd
import numpy as np
import traceback, logging
from datetime import datetime
import time
import json
import copy
import math
import re
import os
import warnings

# MODEL
from model import CuttingStocks
from model import SemiProb
from model import LinearProblem, CuttingOneStock
from auxiliaries import *
warnings.filterwarnings('ignore', category=FutureWarning)

# STAGE1: LOAD ENV PARAMETERS & DATA
global max_bound, mc_ratio, stock_ratio, bound_step
global added_stock_ratio_avg_fc
global starting_bound
global stop_stock_ratio, stop_needcut_wg
global overloaded_total_need_cut, overloaded_total_fg_codes, max_coil_weight

### --- PARAMETER SETTING ---
uat = int(os.getenv('UAT', '0000'))
mc_ratio = float(os.getenv('MC_RATIO', '2.5'))
added_stock_ratio = int(os.getenv('ADDED_STOCK_RATIO', '10'))/100

### --- DEFAULT PARAMETER  ---
added_stock_ratio_avg_fc = [3000, 2000, 1000, 500 ,100]
stop_stock_ratio        = -float(os.getenv('ST_RATIO', '0.03'))
stop_needcut_wg         = -float(os.getenv('STOP_NEEDCUT_WG', '90'))
max_coil_weight         = float(os.getenv('MAX_WEIGHT_MC_DIV_2', '7000'))
max_bound               = float(os.getenv('MAX_BOUND', '3.0'))
bound_step              = float(os.getenv('BOUND_STEP', '0.5'))
overloaded_total_need_cut = float(os.getenv('OVERLOADED_TOTAL_NEEDCUT', '29000'))
overloaded_total_fg_codes = float(os.getenv('OVERLOADED_TOTAL_FG_CODES', '15'))
sub_overloaded_total_fg_codes = float(os.getenv('SUB_OVERLOADED_TOTAL_FG_CODES', '9'))

def find_starting_bound(test_finish, test_stocks):
    """ Find best step, which usually take more stocks in the first round"""
    # If Med-Big, take the least heavy, Small take the heaviest (sorted in partite_stocks)
    st_key = list(test_stocks.keys())[0]
    stock_weight = test_stocks[st_key]['weight']
    stock_width = test_stocks[st_key]['width']
    
    bound_finish = [round((stock_weight/stock_width * v['width'] + v['need_cut'])/(v['average FC']+1)) for _, v in test_finish.items()]
    # Check if any number in the list is >= 2
    bound_2 = np.percentile(sorted(bound_finish,reverse=True), 30)
    if bound_2 >= 2 :
        starting_bound = 2
    else:
        starting_bound = 1
    return starting_bound

def onestock_cut(logger, finish, stocks, MATERIALPROPS, margin_df):
    taken_stocks = []
    bound = starting_bound
    # SET UP
    oneSteel = CuttingOneStock(finish, stocks, MATERIALPROPS)
    oneSteel.update(bound = starting_bound, margin_df = margin_df)
    
    # SET PROB AND SOLVE 
    oneSteel.set_prob()
    stt, final_solution_patterns, over_cut = oneSteel.solve_prob("flow")

    if stt == "Solved":
        stock_key = final_solution_patterns[0]['stock']
        taken_stocks.append(stock_key)
        taken_stocks_dict = {stock_key: stocks[stock_key]['weight']}
        over_cut_rate = {k: round(over_cut[k]/(finish[k]['average FC']+1), 4) for k in over_cut.keys()}
        logger.warning(f"!!! Status {stt}")
        logger.info(f">>> Take stock {stock_key}")
        logger.info(f">>> Overcut amount {over_cut}")
        logger.info(f">>> Overcut ratio: {over_cut_rate}")
        # logger
        logger.info(f">>> Total {len(final_solution_patterns)} Stocks are used, weighting {stocks[stock_key]['weight']}")
        logger.info(f">>> with trim loss pct {final_solution_patterns[0]['trim_loss_pct']} as {final_solution_patterns[0]['trim_loss']} mm")
        return final_solution_patterns, over_cut, taken_stocks, taken_stocks_dict
    else:
        return final_solution_patterns, over_cut, taken_stocks , {} #over-cut aam

def multistocks_cut(retry_inner_count, logger, finish, stocks, MATERIALPROPS, margin_df, prob_type):
    """
    to cut all possible stock with finish demand upto upperbound
    """
    bound = starting_bound
    # SET UP
    steel = CuttingStocks(finish, stocks, MATERIALPROPS)
    steel.update(bound = starting_bound, margin_df = margin_df)
    steel.filter_stocks_by_group_standard()
    steel.check_division_stocks()
    sorted_keys = sorted([k for k in steel.filtered_stocks.keys()])
    steel.filtered_stocks = copy.deepcopy({k: steel.filtered_stocks[k] for k in sorted_keys})

    # st = sorted({k for k in steel.filtered_stocks.keys()})
    logger.info(f"> After dividing stocks: {sorted_keys}")
    
    cond = steel.set_prob(prob_type) 
    final_solution_patterns = []
    
    while cond == True: 
        print("CUTTING DUAL... ")
        len_last_sol = copy.deepcopy(len(final_solution_patterns))
        stt, final_solution_patterns, over_cut = steel.solve_prob("CBC", retry_inner_count) 
        
        check_solution_patterns = (not final_solution_patterns or len(final_solution_patterns) == len_last_sol)
  
        if check_solution_patterns and bound == max_bound:
            logger.info(f"Empty solution/or limited optimals, max bound")
            cond = False
            break
        elif check_solution_patterns and bound < max_bound:
            bound += bound_step
            try: 
                steel.refresh_stocks()
            except AttributeError: 
                pass
            
            finish_k = {k: v['need_cut'] for k, v in steel.prob.dual_finish.items() if v['need_cut'] > 0}
            logger.info(f" No optimal solution for needcut {finish_k}, increase to {bound} bound")
            steel.update_upperbound(bound)
            cond = True 
        else:
            over_cut_rate = {k: round(over_cut[k]/(finish[k]['average FC']+1), 4) for k in over_cut.keys()}
            steel.refresh_stocks()
            steel.refresh_finish()
            
            logger.warning(f"!!! Status {stt}")
            logger.info(f">>> Take stock {[p['stock'] for p in final_solution_patterns]}")
            logger.info(f">>> Overcut amount {over_cut}")
            logger.info(f">>> Overcut ratio: {over_cut_rate}")
            
            empty_fg = (not steel.prob.dual_finish)
            if empty_fg:
                logger.info(f"!!! FINISH CUTTING")
            else:
                st_list = {k: v['need_cut'] for k, v in steel.prob.dual_finish.items()}
                logger.info(f">>> FG continue to cut {st_list}")
            
            re_stocks = [k for k in steel.prob.dual_stocks.keys()]
            empty_stocks = (not re_stocks) 
            if empty_stocks: logger.info(f"!!! Out of stocks")
            else: logger.info(f">>> Remained Stocks {re_stocks}")
            
            cond = (not empty_stocks) & (not empty_fg)
    
    if not final_solution_patterns and bound == max_bound:
        taken_stocks = []
        logger.warning("!!! No Dual Solution ")
        raise NoDualSolution("Final_solution_patterns is empty, and reach MAX bound")
    else:
        taken_stocks    = [p['stock'] for p in final_solution_patterns]
        trimloss        = [p['trim_loss_pct'] for p in final_solution_patterns]
        stock_weight    = [p['stock_weight'] for p in final_solution_patterns]
        taken_stocks_dict = {
                            p['stock']: p['stock_weight'] 
                            for p in final_solution_patterns 
                            if p['stock'] and p['stock_weight']
                        }
        logger.info(f">>> Total {len(final_solution_patterns)} Stocks are used, weighting {sum(stock_weight)}, average trim loss {round(np.mean(trimloss),3) if len(trimloss) > 0 else np.nan}")
        logger.info(f">>> with trim loss each MC {trimloss}")

    try:
        over_cut_rate = {k: round(over_cut[k]/(finish[k]['average FC']+1), 4) for k in over_cut.keys()}
        logger.info(f">>>> TOTAL STOCK RATIO (OVER CUT): {over_cut_rate}")
    except UnboundLocalError: # NO solution
        over_cut = {}

    # REWIND CASE
    if prob_type == "Rewind":
        steel.prob.create_new_stocks_set()
        print("CUTTING REWIND... ")
        if steel.prob.probstt == "Solved":
            print("solved")
            steel.prob.run()
            steel.calculate_finish_after_cut()
        
            final_solution_patterns = copy.deepcopy(steel.prob.final_solution_patterns)
            over_cut = steel.over_cut
            logger.info(f">>> Rewind overcut amount {over_cut}")
            logger.info(f">>> with trim loss each MC {steel.prob.final_solution_patterns[0]['trim_loss_pct']}")
            taken_stocks = [p['stock'] for p in final_solution_patterns]
        else:
            final_solution_patterns = []
            over_cut = {}
            taken_stocks = []
        remained_stocks = {
                    s: {**s_info}
                    for s, s_info in steel.prob.start_stocks.items()
                    if s not in taken_stocks
            }
        return final_solution_patterns, over_cut, taken_stocks, remained_stocks  
    else:
        # RETURN RESULTS
        return final_solution_patterns, over_cut, taken_stocks, taken_stocks_dict

class ContinueSubFinish(Exception):
    pass  

class OutOfStocks(Exception):
    pass

class NoDualSolution(Exception):
    pass

########################## CLEAR FILES ##############################
# Set the directory containing the Excel files
folder_path = 'results'
# Get a list of all Excel files in the folder
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f.startswith(f"UATresult-{uat}")]

for file in excel_files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)
    print(f"Deleted file: {file_path}")

########################## START ##############################
today = datetime.today()
formatted_date = today.strftime("%y-%m-%d")

# START LOGGER
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'log/UATbatch-{uat}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
# LOAD MASTER DATA
margin_df = pd.read_csv((os.getenv('MASTER_MIN_MARGIN')))
spec_type =  pd.read_csv((os.getenv('MASTER_SPECTYPE')))

# LOAD JOB-LIST
logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
logger.info('*** LOAD JOB LIST ***')

with open(f'jobs/job-list-uat-{uat}.json', 'r') as file:
    job_list = json.load(file)

with open(f'jobs/stocks-list-uat-{uat}.json', 'r') as stocks_file:
    stocks_list =json.load(stocks_file)

with open(f'jobs/finish-list-uat-{uat}.json', 'r') as finish_file:
    finish_list = json.load(finish_file)

n_jobs = job_list['number of job']

for i in range(n_jobs):
    # LOAD JOB INFO -- 1/2 LAYER
    tasks = job_list['jobs'][i]['tasks']                    
    materialprop_set = job_list['jobs'][i]['materialprop']
    available_stock_qty = job_list['jobs'][i]['available_stock_qty']
    MATERIALPROPS = stocks_list['materialprop_stock'][materialprop_set]['materialprop']
    batch = MATERIALPROPS['code']

    # LOAD STOCKS
    stocks_to_use = stocks_list['materialprop_stock'][materialprop_set]['stocks'] # Original stocks available 
    total_taken_stocks = []
    taken_stocks =[]

    # START JOB
    logger.info("------------------------------------------------")
    logger.info(f'*** START processing JOB {i} MATERIALPROPS:{batch} ***')

    ## Loop FINISH - each TASK (by CUSTOMER) in JOB - P1.0
    moved_standard_finish = dict()
    for finish_item in finish_list['materialprop_finish'][materialprop_set]['group']:# STANDARD GROUP - LAYER 1
        try:
            # SETUP
            over_cut = dict()
            final_solution_patterns = []
            
            ## GET STANDARD-NAME 
            standard_group = list(finish_item.keys())[0]
            logger.info(f"### STANDARD: {standard_group}")
            
            # Get BEGINNING FINISH -COIL CENTER list
            all_finish = finish_item[standard_group]['FG_set']  # original finish with stock ratio < 0.1 (need cut can be positive)
            if standard_group == "medium":
                all_finish.update(moved_standard_finish)

            avalaible_coil_center = [k for k, _ in available_stock_qty.items()]
            logger.info(f"### COIL CENTER HAVING STOCK TO CHOOSE: {avalaible_coil_center}")
            
            if added_stock_ratio <= stop_stock_ratio:
                filtered_finish = {k: v for k, v in all_finish.items() if v['need_cut']/(v['average FC']+1) < stop_stock_ratio}
                if len(filtered_finish) < 3:
                    finish = copy.deepcopy(all_finish)
                else:
                    finish = copy.deepcopy(filtered_finish)
            else: finish = partite_finish(all_finish, added_stock_ratio)

            if not finish:
                logger.warning("NO FG to cut")
                pass
            else:
                #STOCK BY STANDARD
                stocks_by_standard = copy.deepcopy(filter_stocks_by_group_standard(finish, stocks_to_use))
                logger.info(f"All finished good to cut {len(finish)} by standard with {len(stocks_by_standard)} stocks ")
                
                # CONVERT FG-DICT to DF:
                finish_df= pd.DataFrame.from_dict(finish, orient='index').reset_index() 
                finish_df.rename(columns={'index': 'order_id'}, inplace=True)
                unique_cc_1st = finish_df['1st Priority'].unique()
                avalaible_coil_center = compare_and_add_coilcenter(avalaible_coil_center, unique_cc_1st)
                # LOOP TO START COIL CENTER PRIORITY - LAYER 2
                j = 0
                outer_coil_center_priority_cond = True
                while j <= len(avalaible_coil_center)-1 and outer_coil_center_priority_cond: #NO.2
                    coil_center = avalaible_coil_center[j] # Di tu coil center co it coil nhat
                    logger.info(f"<<< START IN {coil_center} >>>")
                    
                    # Filter rows where '1st_priority' contains [coil_center]
                    coilcenter_finish_df = finish_df[finish_df['1st Priority'] == coil_center]
                    neg_need_cut_df = coilcenter_finish_df[coilcenter_finish_df['need_cut']< 0]
                    sum_need_cut = sum(neg_need_cut_df['need_cut'])
                    
                    if coilcenter_finish_df.shape[0] > 0 and sum_need_cut < 0: 
                        coilcenter_finish_df_sorted = coilcenter_finish_df.loc[coilcenter_finish_df['coil_center_priority'].apply(len).sort_values(ascending=False).index]
                        priority_columns = ['1st Priority', '2nd Priority', '3rd Priority']
                        coilcenter_finish_df_sorted[priority_columns] = coilcenter_finish_df_sorted[priority_columns].astype('object')
                        
                        inner_coil_center_order = coilcenter_finish_df_sorted[['1st Priority','2nd Priority','3rd Priority']].iloc[0].astype(str).tolist()
                        logger.info(f"Inner coil center order {inner_coil_center_order}")
                        
                        icc = 0
                        inner_coil_center_priority_cond = True
                        retry_inner_count = 0
                        add_new_fg_in_new_coil_center = pd.DataFrame(columns=coilcenter_finish_df.columns)
                        while icc <= len(inner_coil_center_order)-1 and inner_coil_center_priority_cond: #NO.3
                            logger.info(f"Round {retry_inner_count}")
                            # CHECK STOCK
                            filtered_stocks_by_wh = copy.deepcopy(filter_stocks_by_wh(stocks_by_standard, coil_center))
                            # Try add new FG in new coil center
                            coilcenter_finish_df = pd.concat([coilcenter_finish_df, add_new_fg_in_new_coil_center], ignore_index=True)
                            # OVERLOAD-CHECK
                            sub_finish_list = dividing_to_subgroup(coilcenter_finish_df, standard_group, MATERIALPROPS)
                            moved_subfinish = {}
                            for subfinish_item in sub_finish_list['subgroup']:
                                group_name = list(subfinish_item.keys())[0] 
                                logger.info(f" SUB_GROUP {group_name}")                
                                # Get SUBFINISH list
                                subfinish = subfinish_item[group_name]
                                subfinish.update(moved_subfinish)
                                
                                try: # GO TO NEXT WH IF STOCK EMPTY WHEN START-CUTTING NEW SUBGROUP 
                                    # + ADDED NEW FG IN NEW HOUSE (AS 1ST PRIORITY)
                                    if len(filtered_stocks_by_wh) == 0:
                                        logger.warning(f'Out of stock for {coil_center}, find more stock in priority order {inner_coil_center_order}')
                                        while icc < len(inner_coil_center_order) - 1:
                                            icc += 1
                                            coil_center = inner_coil_center_order[icc]
                                            stocks_by_next_wh = filter_stocks_by_wh(stocks_by_standard, coil_center)
                                            if len(stocks_by_next_wh) > 0:
                                                logger.info(f">>> FOUND STOCKS IN {coil_center}, REDIRECT TO {coil_center}")
                                                avalaible_coil_center = avalaible_coil_center[avalaible_coil_center != coil_center]
                                                filtered_stocks_by_wh = copy.deepcopy(stocks_by_next_wh)
                                                added_finish_df = finish_df[finish_df['1st Priority']== coil_center]
                                                
                                                if added_finish_df.shape[0]> 0:
                                                    added_finish_df_sorted = added_finish_df.loc[added_finish_df['coil_center_priority'].apply(len).sort_values(ascending=False).index]

                                                    inner_coil_center_order = added_finish_df_sorted[['1st Priority','2nd Priority','3rd Priority']].iloc[0].astype(str).tolist()
                                                    inner_subfinish= pd.DataFrame.from_dict(subfinish, orient='index').reset_index()
                                                    inner_subfinish.rename(columns={'index': 'order_id'}, inplace=True)

                                                    new_coilcenter_finish_df = pd.concat([inner_subfinish, added_finish_df], axis=0, ignore_index=True)
                                                else: 
                                                    new_coilcenter_finish_df = pd.DataFrame.from_dict(subfinish, orient='index').reset_index()
                                                    new_coilcenter_finish_df.rename(columns={'index': 'order_id'}, inplace=True)

                                                # OVERLOAD-CHECK
                                                sub_finish_list = dividing_to_subgroup(new_coilcenter_finish_df, standard_group, MATERIALPROPS)
                                                
                                                for subfinish_item in sub_finish_list['subgroup']: 
                                                    group_name = list(subfinish_item.keys())[0]
                                                    logger.info(f" SUB_GROUP {group_name}")
                                                    
                                                    # Get SUBFINISH list
                                                    subfinish = subfinish_item[group_name]
                                                    raise ContinueSubFinish
                                            else:
                                                logger.warning(f'>>>> Out of stocks for WH {coil_center} as {icc+1} priority')
                                                if icc+1 == 3:
                                                    raise OutOfStocks
                                                else: pass
                                    else: pass
                                except ContinueSubFinish: pass
                                
                                medium_mc_weight = np.percentile(sorted([v['Min_MC_weight'] for _, v in subfinish.items()]),50)
                                partial_f_list = {k for k in subfinish.keys()} 
                                total_need_cut_by_cust_gr = -sum(item["need_cut"] for item in subfinish.values() if item["need_cut"] < 0)
                                logger.info(f"> Finished Goods key {partial_f_list}")
                                logger.info(f"> Total Need Cut: {total_need_cut_by_cust_gr}")

                                # SELECT STOCKS by need cut
                                if len(subfinish) <= 2 and total_need_cut_by_cust_gr < 4000 :
                                    filtered_st = dict(sorted(filtered_stocks_by_wh.items(), key=lambda x: (x[1]['weight'])))
                                    try:
                                        selected_stock = next((key, value) for key, value in filtered_st.items() if value["weight"] >= total_need_cut_by_cust_gr * 1.3)
                                        partial_stocks = copy.deepcopy({selected_stock[0]: selected_stock[1]})
                                    except StopIteration:
                                        partial_stocks = dict(sorted(filtered_stocks_by_wh.items(), key=lambda x: (x[1]['weight'])))
                                else: partial_stocks = partite_stock(filtered_stocks_by_wh, total_need_cut_by_cust_gr * mc_ratio, group_name, medium_mc_weight)

                                # Check unqualified stocks:
                                unqualified_stocks = {key: filtered_stocks_by_wh[key] for key in filtered_stocks_by_wh if key not in partial_stocks}
                                
                                # CHECK partial stock
                                if len(partial_stocks.keys()) > 0:
                                    st = {k for k in partial_stocks.keys()}
                                    logger.info(f"> Number of stocks in {coil_center} : {st}")
                                    # ********** NOW CUT WITH SELECTED FG AND STOCK  ***************
                                    logger.info(f">> CUT for: {len(subfinish.keys())} FINISH  w {len(partial_stocks.keys())} MCs")
                                    args_dict = {
                                                'logger': logger,
                                                'finish': subfinish,
                                                'stocks': partial_stocks,
                                                'MATERIALPROPS': MATERIALPROPS,
                                                'margin_df': margin_df,
                                                }
                                    
                                    # FIND STARTING BOUND
                                    starting_bound = find_starting_bound(subfinish, partial_stocks)
                                    logger.info(f"Starting bound: {starting_bound}")

                                    try: # START OPTIMIZATION
                                        start_time = time.time()
                                        
                                        if len(partial_stocks) == 1 and standard_group == 'big':
                                            logger.info("*** NORMAL ONE STOCK Case ***")
                                            final_solution_patterns, over_cut, taken_stocks, taken_stocks_dict = onestock_cut(**args_dict)
                                            
                                        else:
                                            logger.info("*** NORMAL DUAL MULTI Case ***")
                                            final_solution_patterns, over_cut, taken_stocks, taken_stocks_dict = multistocks_cut(retry_inner_count,**args_dict, prob_type ="Dual")

                                        ### Exclude taken_stocks out of stock_to_use only for dividing MC
                                        stocks_to_use = refresh_stocks(taken_stocks, taken_stocks_dict, stocks_to_use)
                                        stocks_by_standard = refresh_stocks(taken_stocks, taken_stocks_dict, stocks_by_standard)
                                        filtered_stocks_by_wh = refresh_stocks(taken_stocks, taken_stocks_dict, filtered_stocks_by_wh)

                                    except NoDualSolution: 
                                        neg_subfinish = {k: v for k, v in finish.items() if v['need_cut'] < 0}
                                        logger.info(f"---- TRY CUT REWIND OR SEMI for {len(neg_subfinish.keys())} FG and {len(partial_stocks.keys())} ----")
                                        if len(partial_stocks.keys()) == 1 and len(neg_subfinish.keys()) == 1:
                                            logger.info('*** SEMI CASE *** 1 FG vs 1 Stock')
                                            steel = SemiProb(partial_stocks, subfinish, MATERIALPROPS)
                                            steel.update(margin_df)
                                            steel.cut_n_create_new_stock_set()
                                            
                                            ## Update lai stock by wh
                                            filtered_stocks_by_wh.pop(list(partial_stocks.keys())[0])   # truong hop ko cat duoc 
                                            filtered_stocks_by_wh.update(steel.remained_stocks)         # thi 2 dong nay bu tru nhau
                                            
                                            stocks_by_standard.pop(list(partial_stocks.keys())[0])
                                            stocks_by_standard.update(steel.remained_stocks)

                                            stocks_to_use.pop(list(partial_stocks.keys())[0])
                                            stocks_to_use.update(steel.remained_stocks)
                                            try:
                                                taken_stock_key = list(steel.taken_stocks.keys())[0]
                                                trim_loss_semi_pct=(steel.og_stock_width - sum([finish[f]["width"] * steel.cuts_dict[f] for f in steel.cuts_dict.keys()]))/steel.og_stock_width
                                                if trim_loss_semi_pct > 0.04:
                                                    explain = "keep Semi"
                                                else: explain = ""
                                                weight_dict = {f: round(steel.cuts_dict[f] * finish[f]['width'] * steel.taken_stocks[taken_stock_key]['weight']/steel.taken_stocks[taken_stock_key]['width'],3) for f in steel.cuts_dict.keys()}
                                                over_cut = {k: float(v + finish[k]["need_cut"]) for k, v in weight_dict.items()} #need_cut am
                                                logger.info(f"Semi Solution : {taken_stock_key}, {steel.cuts_dict} weight {weight_dict}, over_cut {over_cut}")
                                                fk = list(steel.cuts_dict.keys())[0]
                                                if subfinish[fk]['Max_weight'] !=0 and subfinish[fk]['Max_weight'] !="":
                                                    div_ratio = round((subfinish[fk]['width'] * steel.taken_stocks[taken_stock_key]['weight'])/(steel.taken_stocks[taken_stock_key]['width']* subfinish[fk]['Max_weight']))
                                                else:
                                                    div_ratio=0
                                                if div_ratio <= 1: 
                                                    rmark_note = ""
                                                else: 
                                                    rmark_note = f"chat {div_ratio} phan"
                                                semi_pattern = {"stock": taken_stock_key,
                                                                            "inventory_id": re.sub(r"-Se\d+", "", taken_stock_key),
                                                                            "stock_width":  steel.taken_stocks[taken_stock_key]['width'],
                                                                            "stock_weight": steel.taken_stocks[taken_stock_key]['weight'],
                                                                            "fg_code":{f: subfinish[f]['fg_codes'] for f in steel.cuts_dict.keys()},
                                                                            "Customer": {f: subfinish[f]['customer_name'] for f in steel.cuts_dict.keys()},
                                                                            'cuts': steel.cuts_dict,
                                                                            "FG Weight": weight_dict,
                                                                            "FG width": {f: subfinish[f]['width'] for f in steel.cuts_dict.keys()},
                                                                            "standard": {f: subfinish[f]['standard'] for f in steel.cuts_dict.keys()},
                                                                            "Min weight":{f: subfinish[f]['Min_weight'] for f in steel.cuts_dict.keys()},
                                                                            "Max weight":{f: subfinish[f]['Max_weight'] for f in steel.cuts_dict.keys()},
                                                                            'average_fc':{f: subfinish[f]['average FC'] for f in steel.cuts_dict.keys()},
                                                                            '1st Priority':{f: subfinish[f]['1st Priority'] for f in steel.cuts_dict.keys()},
                                                                            "explanation": explain,
                                                                            "remarks": {f: rmark_note for f in steel.cuts_dict.keys()},
                                                                            "cutting_date":"",
                                                                            "trim_loss":"" , 
                                                                            "trim_loss_pct": round(trim_loss_semi_pct*100,3)
                                                                        }
                                                final_solution_patterns.append(semi_pattern)
                                                taken_stocks_dict = {p['stock']: p['stock_weight'] for p in final_solution_patterns}
                                            except IndexError:
                                                fk = list(subfinish.keys())[0]
                                                sk = list(partial_stocks.keys())[0]
                                                weight_oneline = round(subfinish[fk]['width'] * partial_stocks[sk]['weight']/partial_stocks[sk]['width'],3)
                                                logger.warning(f"one line cut {weight_oneline}-kgs for MC {sk} weight {partial_stocks[sk]['weight']}")
                                                final_solution_patterns = []

                                        elif len(neg_subfinish.keys()) <= 3:
                                            logger.info("*** REWIND Case ***")
                                            # Find the stock item with the highest weight
                                            highest_weight_item = max(partial_stocks.items(), key=lambda x: x[1]["weight"])
                                            # Extracting the item key and details
                                            item_key, item_details = highest_weight_item
                                            rewind_stocks ={item_key: item_details}
                                            rewind_args_dict = {
                                                'logger': logger,
                                                'finish': subfinish,
                                                'stocks': rewind_stocks,
                                                'MATERIALPROPS': MATERIALPROPS,
                                                'margin_df': margin_df,
                                                }
                                            try:
                                                final_solution_patterns, over_cut, taken_stocks, remained_stocks = multistocks_cut(retry_inner_count,**rewind_args_dict, prob_type="Rewind")
                                                taken_stocks_dict = {p['stock']: p['stock_weight'] for p in final_solution_patterns}
                                                logger.info(f"REMAINED stocks: {[remained_stocks.keys()]}")
                                                filtered_stocks_by_wh.pop(list(partial_stocks.keys())[0]) # truong hop ko cat duoc 
                                                filtered_stocks_by_wh.update(remained_stocks)     # thi 2 dong nay bu tru nhau
                                                
                                                stocks_by_standard.pop(list(partial_stocks.keys())[0]) 
                                                stocks_by_standard.update(remained_stocks)   

                                                stocks_to_use.pop(list(partial_stocks.keys())[0]) 
                                                stocks_to_use.update(remained_stocks)    
                                            except TypeError: pass 
                                        else: pass
                                else: logger.warning(f'>>>> Out of SUITABLE stocks for {group_name} in {coil_center}')
                                
                                # MOVE NOT-FULLY CUT FG IN THIS SUBGROUP TO NEXT SUBGROUP
                                moved_subfinish = move_finish(subfinish, over_cut)

                                # COMPLETE CUTTING for sub-finish
                                if not final_solution_patterns:
                                    logger.warning(f"!!! NO solution/NO cutting at {coil_center}")
                                    sum_need_cut = -sum([subfinish[f]['need_cut'] for f in subfinish.keys()])
                                    no_solution = [{
                                        "stock": coil_center,
                                        "inventory_id": "",
                                        "stock_width":  "",
                                        "stock_weight": "",
                                        "cutting_date":"",
                                        "trim_loss":"" , 
                                        "trim_loss_pct": "",
                                        'cuts': {f: 0 for f in subfinish.keys()},
                                        "fg_code":{f: subfinish[f]['fg_codes'] for f in subfinish.keys()},
                                        "Customer":{f: subfinish[f]['customer_name'] for f in subfinish.keys()},
                                        "FG Weight": {f: 0 for f in subfinish.keys()},
                                        "FG width": {f: subfinish[f]['width'] for f in subfinish.keys()},
                                        "standard": {f: subfinish[f]['standard'] for f in subfinish.keys()},
                                        "Min weight":  {f: subfinish[f]['Min_weight'] for f in subfinish.keys()},
                                        "Max weight":  {f: subfinish[f]['Max_weight'] for f in subfinish.keys()},
                                        "average_fc": {f: subfinish[f]['average FC'] for f in subfinish.keys()},
                                        '1st Priority':{f: subfinish[f]['1st Priority'] for f in subfinish.keys()},
                                        "explanation": {f: "No optimal solution for trim loss <4%" if sum_need_cut > 200 else "Minor need-cut" for f in subfinish.keys()},
                                        "remarks": {f: "" for f in subfinish.keys()}
                                        }]
                                    # --- SAVE DF to EXCEL ---
                                    end_time = time.time()
                                    cleaned_materialprop_set = clean_filename(materialprop_set)
                                    filename = f"results/UATresult-{uat}-job{i}-{group_name}-nosolution.xlsx"
                                    df = flattern_data(no_solution)
                                    df['time'] = end_time - start_time
                                    if os.path.isfile(filename):
                                        with pd.ExcelWriter(filename, engine='openpyxl', mode='a',if_sheet_exists='overlay') as writer:
                                            df.to_excel(writer, sheet_name='Sheet1', index=False)
                                    else: df.to_excel(filename, index=False)
                                else: 
                                    # UPDATE USED STOCK AND REMAINING NEEDCUT
                                    total_taken_stocks.append(taken_stocks)
                                    stocks_by_standard = refresh_stocks(taken_stocks, taken_stocks_dict ,stocks_by_standard)
                                    stocks_to_use = refresh_stocks(taken_stocks,taken_stocks_dict, stocks_to_use)
                                    # --- SAVE DF to EXCEL ---
                                    end_time = time.time()
                                    cleaned_materialprop_set = clean_filename(materialprop_set)
                                    filename = f"results/UATresult-{uat}-job{i}-{group_name}-{coil_center}.xlsx"
                                    df = transform_to_df(final_solution_patterns)
                                    df['time'] = end_time - start_time
                                    if os.path.isfile(filename):
                                        with pd.ExcelWriter(filename, engine='openpyxl', mode='a',if_sheet_exists='overlay') as writer:
                                            df.to_excel(writer, sheet_name='Sheet1', index=False)
                                    else: df.to_excel(filename, index=False)

                                    logger.info(f">>> SOLUTION {materialprop_set} for {group_name} at {coil_center} saved  EXCEL file")
                                    # Refresh
                                    over_cut = {}
                                    final_solution_patterns = []

                            # GO TO next INNER COIL CENTER priority to find more suitable coils?
                            while icc < len(inner_coil_center_order) - 1: # NO.4
                                next_warehouse_stocks = filter_stocks_by_wh(stocks_by_standard, inner_coil_center_order[icc+1])
                                current_total_warehouse_stocks =filter_stocks_by_wh(stocks_by_standard, inner_coil_center_order[icc])
                                current_inner_coil_center_priority_cond = False
                                next_inner_coil_center_priority_cond = False
                                for st in unqualified_stocks.keys():
                                    try:
                                        current_total_warehouse_stocks.pop(st)
                                    except KeyError: pass
                                
                                logger.info(f"After refreshing subfinish {moved_subfinish.keys()}")
                                hasnt_negative_over_cut = (not moved_subfinish) # empty moved_subfinish = True
                                if len(current_total_warehouse_stocks) == 0:
                                    icc += 1
                                    logger.info(f"-- Go to inner wh order {inner_coil_center_order[icc]}")
                                    next_inner_coil_center_priority_cond = (not hasnt_negative_over_cut and (len(next_warehouse_stocks)!=0))
                                    logger.info(f"??? Go to next inner warehouse: {next_inner_coil_center_priority_cond} as neg need cut {(not hasnt_negative_over_cut)} and next warehouse {(len(next_warehouse_stocks)!=0)} ")
                                else:
                                    current_inner_coil_center_priority_cond = (not hasnt_negative_over_cut and (len(current_total_warehouse_stocks)!=0))
                                    logger.info(f"??? Continue to cut {current_inner_coil_center_priority_cond}: in {inner_coil_center_order[icc]} warehouse as neg need cut {(not hasnt_negative_over_cut)} and current warehouse stocks {(len(current_total_warehouse_stocks))} ")
                                
                                inner_coil_center_priority_cond = (next_inner_coil_center_priority_cond or current_inner_coil_center_priority_cond)
                                if not inner_coil_center_priority_cond:
                                    break
                                if current_inner_coil_center_priority_cond:
                                    coilcenter_finish_df= pd.DataFrame.from_dict(moved_subfinish, orient='index').reset_index()
                                    coilcenter_finish_df.rename(columns={'index': 'order_id'}, inplace=True)
                                    retry_inner_count +=1
                                    if retry_inner_count < 2:
                                        logger.info(f"Stay in {inner_coil_center_order[icc]} to find other coil {retry_inner_count} times")
                                        break
                                    else: raise OutOfStocks
                                    
                                if next_inner_coil_center_priority_cond:
                                    coil_center = inner_coil_center_order[icc]
                                    logger.info(f">>> REDIRECT TO {coil_center}")
                                    moved_subfinish_df= pd.DataFrame.from_dict(moved_subfinish, orient='index').reset_index()
                                    if icc == 1:
                                        priority_2nd = coilcenter_finish_df['2nd Priority'].unique()
                                        if len(priority_2nd) >1 : # khac nhau 2nd prio
                                            check_leftover_2nd_prio = True
                                            left_over_coil_center_2nd_prio = priority_2nd[priority_2nd != coil_center]
                                            leftover_2nd_prio_subfinish_df = moved_subfinish_df[moved_subfinish_df['2nd Priority']== left_over_coil_center_2nd_prio[0]]

                                        coilcenter_finish_df = moved_subfinish_df[moved_subfinish_df['2nd Priority'] == coil_center]
                                    elif icc == 2 and (left_over_coil_center_2nd_prio==coil_center):
                                        coilcenter_finish_df = moved_subfinish_df[moved_subfinish_df['3rd Priority'] == coil_center]
                                        coilcenter_finish_df = pd.concat([coilcenter_finish_df,leftover_2nd_prio_subfinish_df],axis=0, ignore_index=True)
                                    elif icc == 2:
                                        coilcenter_finish_df = moved_subfinish_df[moved_subfinish_df['3rd Priority'] == coil_center]
                                    coilcenter_finish_df.rename(columns={'index': 'order_id'}, inplace=True)
                                    # Add more FG if any
                                    add_new_fg_in_new_coil_center = finish_df[finish_df['1st Priority'] == coil_center]
                                    avalaible_coil_center = avalaible_coil_center[avalaible_coil_center != coil_center]
                                    break                           
                    else: 
                        logger.warning(f'>>>> DONT HAVE FG w NEEDCUT {sum_need_cut} WITH 1ST PRIORITY AT {coil_center}. Move to next coil center')
                    
                    # GO TO next OUTER COIL CENTER priority if there still 1st priority group
                    if j < len(avalaible_coil_center) - 1:
                        j +=1
                        logging.info(f"TRY go to {avalaible_coil_center[j]}")
                        next_warehouse_stocks = filter_stocks_by_wh(stocks_by_standard, avalaible_coil_center[j])
                        next1st_coilcenter_finish_df = finish_df[finish_df['1st Priority']== avalaible_coil_center[j]]
                        if next1st_coilcenter_finish_df.shape[0] > 0 and len(next_warehouse_stocks):
                            outer_coil_center_priority_cond = True
                        else:
                            outer_coil_center_priority_cond = False      
                    else: 
                        outer_coil_center_priority_cond = False
                    logger.info(f"??? Go to next warehouse 1priority: {outer_coil_center_priority_cond}")

                # MOVE NOT-FULLY CUT FG IN THIS SUBGROUP TO MEDIUM
                moved_standard_finish.update(moved_subfinish)

        except OutOfStocks: 
                continue
                        
        except Exception as e:       
            logger.warning(f"Error with Customer {standard_group}: {type(e)} {e}")
            logger.info(f"Occured on line {traceback.extract_tb(e.__traceback__)[-1]}")
            continue
            
logger.info('**** TEST JOB ENDED **** ')
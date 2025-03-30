import numpy as np
import pandas as pd
import copy
from datetime import datetime
import math
import os
from model import FinishObjects, StockObjects
from model import RewindProb
from model import DualProblem
# from .O41_dual_solver import testDualProblem

global max_coil_weight3
global max_coil_weight2
global customer_group 
global stop_stock_ratio
global stop_needcut_wg

customer_group = ['small','small-medium']
stop_stock_ratio = float(os.getenv('STOP_STOCK_RATIO', '-0.03'))
stop_needcut_wg = float(os.getenv('STOP_NEEDCUT_WG', '-90'))
max_coil_weight2 = float(os.getenv('MAX_WEIGHT_MC_DIV_2', '7000'))
max_coil_weight3 = float(os.getenv('MAX_WEIGHT_MC_DIV_3', '10000'))

# DEFINE PROBLEM
class CuttingStocks:
    def __init__(self, finish, stocks, MATERIALPROPS):
        """
        -- Input --
        MATERIALPROPS: {
            "spec_name": "JSH270C-PO",
            "type": "Carbon",
            "thickness": 3.0,
            "maker": "CSVC",
            "code": "CSVC JSH270C-PO 3.0"
        stocks = {"HTV0269/24-D1": {   "receiving_date": 44961,   "width": 1158, 
                                        "weight": 3936,   "warehouse": "NQS"},
                   "HTV0269/24-D2": {   "receiving_date": 44961,   "width": 1219, 
                                        "weight": 3936,   "warehouse": "NQS"}   
                                        }                  
        finish = {"F288": {   "customer_name": "VPIC1",   "width": 143.5,  "need_cut": -1300.0,  
                            "fc1": 2724.48,   "fc2": 1776.9600000000003,   "fc3": 1752.9600000000003,  
                            "1st Priority": "HSC",   "2nd Priority": "x",   "3rd Priority": "x",  
                            "Min_weight": 0.0,   "Max_weight": 0.0},
                  "F290": {...}
                  }
        """
        self.S = copy.deepcopy(StockObjects(stocks, MATERIALPROPS))
        self.F = copy.deepcopy(FinishObjects(finish, MATERIALPROPS))
        self.over_cut = {}
        self.over_cut_ratio = {}

    def update(self, bound, margin_df):
        self.S.update_min_margin(margin_df)
        self.F.reverse_need_cut_sign() #from negative to positive
        self.F.update_bound(bound)
     
    def _stock_weight_threshold(self, min_w, max_w):
        """_Create a diction of min and weight of coil that will produce same div of FG_

        Args:
            min_w (_int_): Min Weight of Cut Coil by Customer x FG Codes
            max_w (_int_): Max Weight of Cut Coil by Customer x FG Codes
        Result:
        {1080: {'min': 1588.235294117647,
                'max': [2842.1052631578946,
                        5684.210526315789,
                        8526.315789473683,
                        11368.421052631578]},
        1219: {'min': 1792.6470588235295,
               'max': [3207.8947368421054,
                        6415.789473684211,
                        9623.684210526317,
                        12831.578947368422]}}
        """
        
        self.stock_width =list(set([sv['width'] for _ ,sv in self.S.stocks.items()]))
        max_f_width = max([fv['width'] for _, fv in self.F.finish.items()])
        min_f_width = min([fv['width'] for _, fv in self.F.finish.items()])
        self.mm_stock_weight = {}
        for c_width in self.stock_width:
            # find weight range with min-max weight
            # min
            if min_w == 0.0:
                min_coil_weight = 0
            else:
                min_coil_weight = min_w * c_width/min_f_width
            # max
            if max_w == 0.0:
                max_coil_wg = float('inf')
            else:
                max_coil_wg = max_w * c_width/ max_f_width
            
            # check if min_coil_weight << max_coil_weight

            max_threshold = []
            for i in range(4):
                max_t = max_coil_wg * (i+1)
                max_threshold.append(max_t)
            self.mm_stock_weight[c_width] = {"min": min_coil_weight, "max": max_threshold}

    def _filter_min_stock(self):
        "Remove MC that too small, 1 line cut [weight] < Min Weight "
        self.filtered_stocks ={}
        for s_width, sv in self.mm_stock_weight.items():
            min_w = sv['min']
            filtered_min_stocks = {k: v for k, v in self.S.stocks.items() if (v['width'] == s_width and v['weight'] >= min_w)}
            self.filtered_stocks.update(filtered_min_stocks)
    
    def filter_stocks_by_group_standard(self):
        # Find coil-standard group
        first_item = list(self.F.finish.items())[0]
        customer_gr = first_item[1]['cut_standard']
        if  "medium" in customer_gr:
            # Filter out stock < min MC weight
            mc_weights = [round(v['Min_MC_weight']) if v is not None and not math.isnan(v['Min_MC_weight']) else 0 for _, v in self.F.finish.items()]
            # Convert to a NumPy array and filter out NaNs
            mc_weights = np.array(mc_weights, dtype=float)
            mc_weights = mc_weights[~np.isnan(mc_weights)]  # Remove NaN values
            # Check if the list is empty after removing NaNs
            if np.percentile(mc_weights, 50) == 0:
                self.min_mc_weight = 3500
            else:
                min_mc_weight = np.percentile(mc_weights, 50)
                if min_mc_weight > 5000:
                    self.min_mc_weight = 5000
                else: self.min_mc_weight = min_mc_weight
            self.filtered_stocks = {k: v for k, v in self.S.stocks.items() if v['weight_1219'] >= self.min_mc_weight}
        elif "big" in customer_gr  and len(list(self.F.finish.items())) >2:
            # Filter out stock < 7000
            filtered = {k: v for k, v in self.S.stocks.items() if v['weight_1219'] >= max_coil_weight2}
            if len(filtered) == 0:
                self.filtered_stocks = {k: v for k, v in self.S.stocks.items() if v['weight_1219'] >= 4000}
            else:
                self.filtered_stocks = {k: v for k, v in self.S.stocks.items() if v['weight_1219'] >= max_coil_weight2}
        else:
            # No filter for small group/or too few FG
            self.filtered_stocks = copy.deepcopy(self.S.stocks)

    def _div_to_superbig_stocks(self):
        pop_stock_key = []
        stock_list = copy.deepcopy(self.filtered_stocks)
        # min_mc_weight  = min([self.F.finish[f]['Min_MC_weight'] for f in ]
        for k, v in stock_list.items():
            if v['weight'] >= (max_coil_weight2+1000) * 2:
                pop_stock_key.append(k)
                half_wg = v['weight'] * 0.5
                for i in range(2):
                    self.filtered_stocks[f'{k}-Di{i+1}'] = v
                    self.filtered_stocks[f'{k}-Di{i+1}'].update({'weight': half_wg})
                    self.filtered_stocks[f'{k}-Di{i+1}'].update({'status':"R:REWIND"})
            else:
                pass
        for s in pop_stock_key:
            #### Update lai stock
                self.filtered_stocks.pop(s) #remove original stock
    
    def _div_to_medi_stocks(self):
        pop_stock_key = []
        stock_list = copy.deepcopy(self.filtered_stocks)
        # min_mc_weight  = min([self.F.finish[f]['Min_MC_weight'] for f in ]
        for k, v in stock_list.items():
            if v['weight'] >= self.min_mc_weight * 2:
                pop_stock_key.append(k)
                half_wg = v['weight'] * 0.5
                for i in range(2):
                    self.filtered_stocks[f'{k}-Di{i+1}'] = v
                    self.filtered_stocks[f'{k}-Di{i+1}'].update({'weight': half_wg})
                    self.filtered_stocks[f'{k}-Di{i+1}'].update({'status':"R:REWIND"})
            else:
                pass
        for s in pop_stock_key:
            #### Update lai stock
                self.filtered_stocks.pop(s) #remove original stock
    
    def _div_to_small_stocks(self):
        pop_stock_key = []
        for p, v in enumerate(self.check_stock_pos):
            if v == 1:
                stock_item = list(self.filtered_stocks.items())[p]
                stock_key = stock_item[0]
                pop_stock_key.append(stock_key)
                if self.filtered_stocks[stock_key]['weight'] < max_coil_weight3:
                    half_wg = self.filtered_stocks[stock_key]['weight']*0.5
                    for i in range(2):
                        self.filtered_stocks[f'{stock_key}-Di{i+1}'] = self.filtered_stocks[stock_key]
                        self.filtered_stocks[f'{stock_key}-Di{i+1}'].update({'weight': half_wg})
                        self.filtered_stocks[f'{stock_key}-Di{i+1}'].update({'status':"R:REWIND"})
                else:
                    third_wg = self.filtered_stocks[stock_key]['weight']/3
                    for i in range(3):
                        self.filtered_stocks[f'{stock_key}-Di{i+1}'] = self.filtered_stocks[stock_key]
                        self.filtered_stocks[f'{stock_key}-Di{i+1}'].update({'weight': third_wg})
                        self.filtered_stocks[f'{stock_key}-Di{i+1}'].update({'status':"R:REWIND"})
            else:
                pass
            
        for s in pop_stock_key:
            #### Update lai stock
                self.filtered_stocks.pop(s) #remove original stock
               
    def check_division_stocks(self):
        # Check customer gr
        first_item = list(self.F.finish.items())[0]
        customer_gr = first_item[1]['cut_standard']
        # Check stocks need to be divided for small and small-medium
        self.check_stock_pos = [1 if v['weight']>= max_coil_weight2 else 0 for _, v in self.filtered_stocks.items()]
        if "medium" in customer_gr:
            self._div_to_medi_stocks()
        elif "small" in customer_gr and np.sum(self.check_stock_pos)>=1:
            self._div_to_small_stocks()
        else: pass
        # Sort weight descending and receiving_date ascending
        self.filtered_stocks = dict(sorted(self.filtered_stocks.items(), key=lambda x: (-datetime.fromisoformat(x[1]['receiving_date']).timestamp(),x[1]['weight']), reverse= True))
    
    def filter_stocks_min_max(self, min_weight = None, max_weight = None):
        if min_weight == None and max_weight == None: 
            # flow khong consider min va max_weight
            self.filtered_stocks = copy.deepcopy(self.S.stocks)
        else:
            self._stock_weight_threshold(min_weight, max_weight)
            self._filter_min_stock()

    def set_prob(self, prob_type):
        if len(self.filtered_stocks) > 0 and prob_type == "Dual":
            # Initiate problem  => thanh self.prob.dual_finish va self.prob.dual_stocks
            self.prob = DualProblem(self.F.finish, self.filtered_stocks)
            return True # continue to run solve prob
        elif len(self.filtered_stocks) > 0 : 
            self.prob = RewindProb(self.F.finish, self.filtered_stocks)
            return False
        else: #ko co stock de cat
            return False

    # Phase 5: Evaluation Over-cut / Stock Ratio
    def _sum_weight(self):
        # Initialize an empty dictionary for the total sums
        sums_weight = {}
        # Loop through each dictionary in the list
        for entry in self.prob.final_solution_patterns:
            # count = entry['count']
            cuts = entry['FG Weight']
            # Loop through each key in the cuts dictionary
            for key, value in cuts.items():
                # If the key is not already in the total_sums dictionary, initialize it to 0
                if key not in sums_weight:
                    sums_weight[key] = 0
                # Add the product of count and value to the corresponding key in the total_sums dictionary
                sums_weight[key] += round(value,3)
                
        return sums_weight

    def _calculate_div_ratio(self,s):
        """
        {1080: {'min': 1588.235294117647,
                'max': [2842.1052631578946,
                        5684.210526315789,
                        8526.315789473683,
                        11368.421052631578]},
        1219: {'min': 1792.6470588235295,
               'max': [3207.8947368421054, 3 2
                        6415.789473684211, 2 3
                        9623.684210526317, 1 4
                        12831.578947368422]}} 0 5
        """
        # doi chieu stock weight voi range max in mm_stock_weight by stock_width
        stocks_width = self.prob.start_stocks[s]['width']
        to_verify_range = sorted(self.mm_stock_weight[stocks_width]['max'], reverse=True)
        self.div_ratio = 0
        for i, w in enumerate(to_verify_range):
            if self.prob.start_stocks[s]['weight'] >= w:
                self.div_ratio =  (3-i) + 2
                break

    def _calculate_finish_after_cut_by_mm_weight(self):
        # for all orginal finish, not only dual
        # ap dung TH chat 1 kieu cho tat ca cac FG code 
        if self.prob.probstt == "Solved":
            for i, sol in enumerate(self.prob.final_solution_patterns):
                s = self.prob.final_solution_patterns[i]['stock'] # stock cut
                self._calculate_div_ratio(s)
                cuts_dict = self.prob.final_solution_patterns[i]['cuts']
                weight_dict = {f: round(cuts_dict[f] * self.F.finish[f]['width'] * self.prob.start_stocks[s]['weight']/self.prob.start_stocks[s]['width'],3) for f in cuts_dict.keys()}
                if self.div_ratio <= 1: 
                    rmark_note = "" 
                else: rmark_note = f"chat {self.div_ratio} phan"
                self.prob.final_solution_patterns[i] = {**sol,
                                                        "fg_code":{f: self.F.finish[f]['fg_codes'] for f in cuts_dict.keys()},
                                                        "Customer":{f: self.F.finish[f]['customer_name'] for f in cuts_dict.keys()},
                                                        "FG Weight": weight_dict,
                                                        "FG width": {f: self.F.finish[f]['width'] for f in cuts_dict.keys()},
                                                        "standard": {f: self.F.finish[f]['standard'] for f in cuts_dict.keys()},
                                                        "Min weight": {f: self.F.finish[f]['Min_weight'] for f in cuts_dict.keys()},
                                                        "Max weight": {f: self.F.finish[f]['Max_weight'] for f in cuts_dict.keys()},
                                                        "average_fc":{f: self.F.finish[f]['average FC'] for f in cuts_dict.keys()},
                                                        '1st Priority':{f: self.F.finish[f]['1st Priority'] for f in cuts_dict.keys()}
                                                        }
                self.prob.final_solution_patterns[i].update({"remarks": rmark_note})
            # Total Cuts
            total_sums = self._count_weight()
            for k in self.F.finish.keys():
                try:
                    self.over_cut[k] = round(total_sums[k] - self.F.finish[k]['need_cut'],3)
                except KeyError:
                    self.over_cut[k] = - round(self.F.finish[k]['need_cut'],3) 
                    # can tinh overcut cho moi finish, du ko duoc cat trong list dual finish
                self.over_cut_ratio[k] = round(self.over_cut[k]/(self.F.finish[k]['average FC']+1),3)
        else: pass # ko co nghiem trong lan chay truoc do
    
    def _remark_div_ratio_by_mm_weight(self, wg, line, min, max):
        """_summary_
        Args:
            w (_type_): total cut weight of f in F
            c (_type_): line cut of f in F
            min (_type_): min weight of f in F
            max (_type_): max weight of f in F
        Returns:
            remark_note : 
        """
        min_value = float(min) if min else 0.0
        max_value = float(max) if max else 0.0
        if line == 0 or max_value == 0.0:
            rnote = ""
        elif float(wg)/float(line) < 0.85 * min_value:
            rnote = "nho hon min"
        elif float(wg)/float(line) <= 1.15 * max_value:
            rnote = ""
        else:
            div_ratio = round(wg/(line * max_value)) # cho phep sai so
            if div_ratio <= 1: 
                rnote = "" 
            else: rnote = f"chat {div_ratio} phan"
        return rnote
    
    def _compute_weight_and_remarks(self,unit_weight,cuts_dict):
        weight_dict = {f: round(cuts_dict[f] * self.F.finish[f]['width'] * unit_weight,3) for f in cuts_dict.keys()}
        rmark_dict = {f: self._remark_div_ratio_by_mm_weight(weight_dict[f],cuts_dict[f],
                                                                        self.F.finish[f]['Min_weight'],
                                                                        self.F.finish[f]['Max_weight']) for f in cuts_dict.keys()}
        return weight_dict, rmark_dict
    
    def _generate_fg_data(self, cuts_dict, weight_dict, rmark_dict):
        return {
                "fg_code":{f: self.F.finish[f]['fg_codes'] for f in cuts_dict.keys()},
                "Customer":{f: self.F.finish[f]['customer_name'] for f in cuts_dict.keys()},
                "FG Weight": weight_dict,
                "FG width": {f: self.F.finish[f]['width'] for f in cuts_dict.keys()},
                "standard": {f: self.F.finish[f]['standard'] for f in cuts_dict.keys()},
                "Min weight":  {f: self.F.finish[f]['Min_weight'] for f in cuts_dict.keys()},
                "Max weight":  {f: self.F.finish[f]['Max_weight'] for f in cuts_dict.keys()},
                "average_fc": {f: self.F.finish[f]['average FC'] for f in cuts_dict.keys()},
                '1st Priority':{f: self.F.finish[f]['1st Priority'] for f in cuts_dict.keys()},
                "remarks": rmark_dict
                }
    
    def calculate_finish_after_cut(self):
        # for all orginal finish, not only dual
        duplicated_patterns = []
        self.duplicated_stock_id = []
        if self.prob.probstt == "Solved":
            self.prob.final_solution_patterns = sorted(self.prob.final_solution_patterns, key=lambda x: (x['stock'], x.get('trim_loss_pct', float('inf'))))
            last_inventory_id=""
            last_cuts_dict={}
            for i, sol in enumerate(self.prob.final_solution_patterns):
                inventory_id = self.prob.final_solution_patterns[i]['inventory_id'] # orginal name stock taken
                cuts_dict = self.prob.final_solution_patterns[i]['cuts']
                
                if i != 0 and (inventory_id == last_inventory_id) and (cuts_dict == last_cuts_dict):
                    # merge value cut dict and stock weight
                    merged_weight = self.prob.final_solution_patterns[i]['stock_weight']+self.prob.final_solution_patterns[i-1]['stock_weight']
                    unit_weight = merged_weight/self.prob.final_solution_patterns[i]['stock_width']

                    weight_dict, rmark_dict = self._compute_weight_and_remarks(unit_weight,cuts_dict)
                    self.prob.final_solution_patterns[i] = {**sol, **self._generate_fg_data(cuts_dict, weight_dict, rmark_dict)}
                    # new stock name
                    neu_stock_id = f"{inventory_id}-Me"
                    stock_name_div = copy.deepcopy(self.prob.final_solution_patterns[i]['stock'])
                    # update id and weight
                    self.prob.final_solution_patterns[i].update({"stock_weight": merged_weight})
                    self.prob.final_solution_patterns[i].update({"stock": neu_stock_id})
                    duplicated_patterns.append(i-1)
                    # Refresh prob.start_stocks accordingly
                    last_stock_name_div = self.prob.final_solution_patterns[i-1]['stock']
                
                    self.prob.start_stocks[neu_stock_id] = self.prob.start_stocks[last_stock_name_div].copy()
                    self.prob.start_stocks[neu_stock_id].update({"stock_weight": merged_weight})
                    self.prob.start_stocks.pop(last_stock_name_div)
                    self.prob.start_stocks.pop(stock_name_div)
                else:
                    last_inventory_id = inventory_id
                    last_cuts_dict = cuts_dict

                    unit_weight = self.prob.final_solution_patterns[i]['stock_weight']/self.prob.final_solution_patterns[i]['stock_width']
                    weight_dict, rmark_dict = self._compute_weight_and_remarks(unit_weight,cuts_dict)
                    self.prob.final_solution_patterns[i] = {**sol, **self._generate_fg_data(cuts_dict, weight_dict, rmark_dict)}

            # Sort in reverse to avoid indexing errors when popping
            if not duplicated_patterns:
                pass
            else:
                for j in sorted(duplicated_patterns, reverse=True):
                    self.prob.final_solution_patterns.pop(j)

            # Total Cuts
            sums_weight = self._sum_weight()
            for k in self.F.finish.keys():
                try:
                    self.over_cut[k] = round(sums_weight[k] - self.F.finish[k]['need_cut'],3) 
                except KeyError:
                    self.over_cut[k] = - round(self.F.finish[k]['need_cut'],3)
                self.over_cut_ratio[k] = round(self.over_cut[k]/(self.F.finish[k]['average FC']+1),3)
        else: pass # ko co nghiem
            
    def solve_prob(self, solver):
        # Run and calculate results
        self.prob.run(solver)
        self.calculate_finish_after_cut()
        return self.prob.probstt, self.prob.final_solution_patterns, self.over_cut

    def _check_remain_stocks(self):
        # Extract stocks from final_solution_patterns
        if not self.prob.final_solution_patterns:
            taken_stocks = {}
        else :
            taken_stocks = {p['stock'] for p in self.prob.final_solution_patterns}
        # Find remained_stocks dictionary - 
        self.remained_stocks = {
            s: {**s_info}
            for s, s_info in self.prob.start_stocks.items()
            if s not in taken_stocks
        }
        self.remained_stocks = dict(sorted(self.remained_stocks.items(), key=lambda x: (-datetime.fromisoformat(x[1]['receiving_date']).timestamp(),x[1]['weight']), reverse= True))
        
    def check_status(self):
        self._check_remain_stocks()
        # CHECK FOR ANOTHER ROUND
        if not self.prob.overused_list or not self.remained_stocks: # No overused list or remained_stocks empty
            print("FINISH CUTTING")
            return False
        else:
            print("CONTINUE CUTTING")
            return True
        
    def refresh_stocks(self):
        # Update remained stock after cut
        self._check_remain_stocks()
        self.prob.dual_stocks = copy.deepcopy(self.remained_stocks)

    def refresh_finish(self):
        # Take only finish with negative need_cut or stock ratio low
        re_finish = {k: v for k, v in self.prob.dual_finish.items() 
                        if self.over_cut_ratio[k] < stop_stock_ratio 
                        or self.over_cut[k] < stop_needcut_wg}
        if len(re_finish) == 0 or len(re_finish) > 3: 
            pass
        elif len(re_finish) <=3 and sum([self.over_cut[k] for k, _ in re_finish.items()]) >= -200:
            re_finish = {} # con it need cut -> ko cat nua
        else:
            for stop_rate in [0.0, 0.1, 0.3, 0.5]:
                len_remained_finish = [k for k, v in self.over_cut_ratio.items() if v <stop_rate]
                if len(len_remained_finish) > 3:
                    re_finish = copy.deepcopy({k: self.prob.start_finish[k] for k in len_remained_finish})
                    break
                
        # update finish and turn need cut to positive sign
        try:
            for f, f_info in re_finish.items():
                f_info['need_cut'] = copy.deepcopy(self.over_cut[f])
                f_info['upper_bound'] += -self.over_cut[f]
                if f_info['need_cut']  < 0:
                    f_info['need_cut'] *= -1
                else: 
                    f_info['need_cut'] = 0
        except KeyError: #empty refinish
            pass
        self.prob.dual_finish = copy.deepcopy(re_finish)
    
    def update_upperbound(self, bound):
        # update bound for Objects already is reversed sign of needcut
        self.prob.dual_finish = {f: {**f_info, 
                                     "upper_bound": f_info['need_cut'] + f_info['average FC']* bound} 
                                 for f, f_info in self.prob.dual_finish.items()
                                 }
          
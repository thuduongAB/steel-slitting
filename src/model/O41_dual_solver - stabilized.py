import pandas as pd
import numpy as np
import re
import copy
import os

# from model import FinishObjects, StockObjects
from pulp import LpMaximize, LpMinimize, LpProblem,LpAffineExpression, LpVariable, lpSum, PULP_CBC_CMD, GLPK ,GLPK_CMD, LpBinary, value
# from ortools.linear_solver import pywraplp

global stop_stock_ratio
global stop_needcut_wg

stop_stock_ratio = float(os.getenv('STOP_STOCK_RATIO', '-0.03'))
stop_needcut_wg = -100

# DEFINE PROBLEM
class DualProblem:
    
    def __init__(self, finish, stocks):
        self.len_stocks = len(stocks)
        self.dual_stocks = copy.deepcopy(stocks)
        self.start_stocks = copy.deepcopy(stocks)
        self.dual_finish = copy.deepcopy(finish)
        self.start_finish = copy.deepcopy(finish)
        self.final_solution_patterns = []

    # PHASE 1: Naive/ Dual Pattern Generation
    def _make_naive_patterns(self):
        """
        Generates patterns of feasible cuts from stock width to meet specified finish widths.
        """
        self.patterns = []
        # stock_check = {k for k in self.dual_stocks.keys()}
        # print(f"stock check bef nai patterns {stock_check}")
        for f in self.dual_finish:
            feasible = False
            for s in self.dual_stocks:
                # max number of f that fit on s, bat buoc phai round down vi ko cat qua width duoc
                num_cuts_by_width = int((self.dual_stocks[s]["width"]-self.dual_stocks[s]["min_margin"]) / self.dual_finish[f]["width"])
                # max number of f that satisfied the need cut WEIGHT BOUND
                num_cuts_by_weight = round((self.dual_finish[f]["upper_bound"] * self.dual_stocks[s]["width"] ) / (self.dual_finish[f]["width"] * self.dual_stocks[s]['weight']))
                # min of two max will satisfies both
                num_cuts = min(num_cuts_by_width, num_cuts_by_weight)

                # make pattern and add to list of patterns
                if num_cuts > 0:
                    feasible = True
                    cuts_dict = {key: 0 for key in self.dual_finish.keys()}
                    cuts_dict[f] = num_cuts
                    trim_loss = self.dual_stocks[s]['width'] - sum([self.dual_finish[f]["width"] * cuts_dict[f] for f in self.dual_finish.keys()])
                    trim_loss_pct = round(trim_loss/self.dual_stocks[s]['width'] * 100, 3)
                    self.patterns.append(
                        {"stock":s,
                         "inventory_id": re.sub(r"-Di\d+", "", s),
                        'trim_loss':trim_loss, 
                        "trim_loss_pct": trim_loss_pct ,
                        "explanation":"",'remark':"","cutting_date":"",
                        'coil_status':self.dual_stocks[s]["status"],
                        "stock_weight": self.dual_stocks[s]['weight'], 
                        'stock_width':self.dual_stocks[s]['width'],
                        "cuts": cuts_dict
                        }
                    )
                    
            if not feasible:
                continue

    def create_finish_demand_by_line_w_naive_pattern(self):
        self._make_naive_patterns()
        self.dual_finish = {f: {**f_info
                                ,"upper_demand_line": max([self.patterns[i]['cuts'][f] for i,_ in enumerate(self.patterns)], default=1)
                                ,"demand_line": min([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns) if self.patterns[i]['cuts'][f] != 0], default= 0)
                            } for f, f_info in self.dual_finish.items()
        }
        
    # PHASE 2: Pattern Duality
    def _filter_out_overlapped_stock(self):
        if self.max_key is not None:
            del self.dual_stocks[self.max_key]
            # Refresh patterns      
            self.create_finish_demand_by_line_w_naive_pattern() #bo refresh thi pattern se it hon
        else: pass
        
    def _count_pattern(self,patterns):
        """
        Count each stock is used how many times
        """
        stock_counts = {}
        # Iterate through the list and count occurrences of each stock
        for item in patterns:
            stock = item['stock']
            count = 1
            if stock in stock_counts:
                stock_counts[stock] += count
            else:
                stock_counts[stock] = count

        return stock_counts

    def _stabilize_duals(self, dual_value, stabilization_factor=0.1):
        return max(0, dual_value - stabilization_factor)

    def _new_pattern_problem(self, width_s, ap_upper_bound, demand_duals, MIN_MARGIN):
        prob = LpProblem("NewPatternProblem", LpMaximize)

        # Decision variables - Pattern
        ap = {f: LpVariable(f"ap_{f}", lowBound=0, upBound = ap_upper_bound[f], cat="Integer") for f in self.dual_finish.keys()}

        # Objective function
        # maximize marginal_cut:
        prob += lpSum(ap[f] * demand_duals[f] for f in self.dual_finish.keys()), "MarginalCut"

        # Constraints - subject to stock_width - MIN MARGIN
        prob += lpSum(ap[f] * self.dual_finish[f]["width"] for f in self.dual_finish.keys()) <= width_s - MIN_MARGIN, "StockWidth_MinMargin"
        
        # Constraints - subject to trim loss 4% 
        prob += lpSum(ap[f] * self.dual_finish[f]["width"] for f in self.dual_finish.keys()) >= 0.96 * width_s , "StockWidth"

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs'],timeLimit=60))

        # marg_cost = value(prob.objective)
        marg_cost = self._stabilize_duals(value(prob.objective))
        pattern = {f: int(ap[f].varValue) for f in self.dual_finish.keys()}
        
        return marg_cost, pattern

    def _generate_dual_pattern(self):
        # Stock nao do toi uu hon stock khac o width thi new pattern luon bi chon cho stock do #FIX
        prob = LpProblem("GeneratePatternDual", LpMinimize)

        # Sets
        F = list(self.dual_finish.keys())
        P = list(range(len(self.patterns)))

        # Parameters
        s = {p: self.patterns[p]["stock"] for p in range(len(self.patterns))}
        a = {(f, p): self.patterns[p]["cuts"][f] for p in P for f in F}
        demand_finish = {f: self.dual_finish[f]["demand_line"] for f in F}
        upper_demand_finish = {f: self.dual_finish[f]["upper_demand_line"] for f in F}

        # Decision variables #relaxed integrality
        x = {p: LpVariable(f"x_{p}", lowBound=0, upBound=20, cat="Continuous") for p in P}

        # OBJECTIVE function minimize stock used:
        prob += lpSum(x[p] for p in P), "Cost"

        # Constraints
        for f in F:
            prob += (
                lpSum(a[f, p] * x[p] for p in P) >= demand_finish[f], 
                f"Demand_{f}"
            )
            prob += (lpSum(a[f, p] * x[p] for p in P) <= upper_demand_finish[f], 
                     f"UpperDemand_{f}" 
            )

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))

        # Extract dual values
        dual_values = {f: prob.constraints[f"Demand_{f}"].pi for f in F}

        ap_upper_bound = {
            f: max([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns)], default=0) 
            for f in self.dual_finish.keys()
        }
        demand_duals = {f: dual_values[f] for f in F}

        marginal_values = {}
        pattern = {}
        for s in self.dual_stocks.keys():
            marginal_values[s], pattern[s] = self._new_pattern_problem( #new pattern by line cut (trimloss), ignore weight
                self.dual_stocks[s]["width"], ap_upper_bound, demand_duals, self.dual_stocks[s]["min_margin"]
            )
        try:
            s = max(marginal_values, key=marginal_values.get) # pick the first stock if having same width
            cuts_dict =pattern[s]
            new_pattern = {"stock":s, "inventory_id": re.sub(r"-Di\d+", "", s),
                           'stock_weight': self.dual_stocks[s]['weight'], 
                           'stock_width': self.dual_stocks[s]["width"],
                           "cuts": cuts_dict,
                            "explanation":"",'remark':"","cutting_date":""
                           }
            # print(f"test cut dict {cuts_dict}")
        except ValueError or TypeError:
            new_pattern = None
        return new_pattern
    
    # GENERATION FLOW
    def generate_patterns(self):
        n = 1
        remove_stock = True
        self.max_key = None
        self.total_dual_pat = []
        while remove_stock == True:
            self._filter_out_overlapped_stock()
            new_pattern = self._generate_dual_pattern() 
            dual_pat = [] # check stock lap
            while (new_pattern not in dual_pat) and (new_pattern is not None):
                dual_pat.append(new_pattern) 
                self.total_dual_pat.append(new_pattern)
                self.patterns.append(new_pattern)
                new_pattern = self._generate_dual_pattern()
            # Filter stock having too many patterns
            if not dual_pat:
                remove_stock = False
            else:
                ls = self._count_pattern(dual_pat)
                self.max_key = max(ls, key=ls.get) 
                max_count = ls[self.max_key]
                if max_count >= 1 and n < self.len_stocks: #remove until only 1 stock
                    remove_stock = True
                    n +=1
                else: 
                    remove_stock = False          
        # Refresh naive patterns
        self._make_naive_patterns()
        
    # PHASE 3: Filter Patterns
    def _count_fg_cut(self, dict):
        # Count bao nhieu FG duoc cat trong pattern nay
        count = sum(1 for value in dict.values() if value > 0)
        return count
    
    def filter_patterns_and_stocks_by_constr(self):
        """_summary_
        {"stock":s, 
            "inventory_id": s,
            'stock_weight': 4000, 
            'stock_width': 1219,
            "cuts": pattern[s],
            }
        """
        # Initiate list
        self.filtered_patterns = []
        print(f"leng DUAL PATTERNS {len(self.total_dual_pat)}")
        # Filter patterns
        final_patterns = [*self.patterns, *self.total_dual_pat]
        for pattern in final_patterns:
            cuts_dict= pattern['cuts']
            count_fg_cut = self._count_fg_cut(cuts_dict)
            width_s = self.start_stocks[pattern['stock']]['width']
            trim_loss = width_s - sum([self.start_finish[f]["width"] * cuts_dict[f] for f in cuts_dict.keys()])
            trim_loss_pct = round(trim_loss/width_s * 100, 3)
            if (count_fg_cut == 1 and trim_loss_pct <= 3.5) or (count_fg_cut > 1 and trim_loss_pct < 4.0):
                # Filter out pattern only cut for 1 FG and trim loss too high
                pattern.update({'trim_loss': trim_loss, 
                                "trim_loss_pct": trim_loss_pct,
                                'coil_status':self.start_stocks[pattern['stock']]['status'],
                                "count_cut": count_fg_cut})
                self.filtered_patterns.append(pattern)
                
        # Sort pattern
        status_order = {"S:SEMI MCOIL": 0, "R:REWIND": 1, "M:RAW MATERIAL": 2}

        self.filtered_patterns = sorted(
            self.filtered_patterns,
            key=lambda x: (
                x['trim_loss_pct'],
                -x['count_cut'],
                status_order.get(x['coil_status'], float('inf'))  # Default to 'inf' for unknown statuses
            )
        )
        # Initiate chosen stocks dict
        self.chosen_stocks = {}

        # Filter stocks
        filtered_stocks = copy.deepcopy([self.filtered_patterns[i]['stock'] for i in range(len(self.filtered_patterns))])
        for stock_name, stock_info in self.start_stocks.items():
            if stock_name in filtered_stocks and stock_name not in self.chosen_stocks.keys():
                self.chosen_stocks[stock_name]= {**stock_info}
    
    # PHASE 4: Optimize WEIGHT Patterns    
    def optimize_cut(self):
        # Sets
        S = list(self.chosen_stocks.keys())
        P = list(range(len(self.filtered_patterns)))

        # PARAMETER - unit weight of stock
        c = {p: self.chosen_stocks[pattern["stock"]]["weight"]/self.chosen_stocks[pattern["stock"]]["width"] for p, pattern in enumerate(self.filtered_patterns)}
        s = {p: self.chosen_stocks[pattern["stock"]]["weight"] for p, pattern in enumerate(self.filtered_patterns)}
        sp = {(s, p): 1 if self.filtered_patterns[p]["stock"]== s else 0 for p in P for s in S}
        
        # Define VARIABLES
        x = {p: LpVariable(f"X_{p}",lowBound=0,upBound=1,cat='Integer') for p in P}
        
        # Create a LP minimization problem
        prob = LpProblem("PatternCuttingProblem", LpMinimize)
        
        # Objective function: MINIMIZE total stock use
        # prob += lpSum(x[p] for p in P), "TotalStockUse"
        prob += LpAffineExpression([
            (x[p], s[p]) for p in P
        ])

        # Constraints: meet demand for each finished part
        for f in self.dual_finish:
            prob += (
                lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p] * x[p]
                          for p in range(len(self.filtered_patterns))) >= self.dual_finish[f]['need_cut'] - 0.019*self.dual_finish[f]['average FC'], 
                f"DemandWeight{f}"
            )
            prob += (
                lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p] * x[p]
                          for p in range(len(self.filtered_patterns))) <= self.dual_finish[f]['upper_bound'], 
                f"UpperDemandWeight{f}"
            )
        
        # Contraint: for each stock choose one only pattern
        for s in S:
            prob += (
                lpSum(x[p] * sp[s,p] for p in range(len(self.filtered_patterns))) <= 1,
                f"PatternStock{s}"
            )
        
        # Solve the problem GLPL - IBM solver
        prob.solve(GLPK_CMD(msg=False, path="/opt/homebrew/bin/glpsol",timeLimit=60))
        print("solver GLPK")

        try: # Extract results
            solution = [
                        round(x[p].varValue) 
                        for p in range(len(self.filtered_patterns))
                    ]      
            self.solution_list = []
            for i, pattern_info in enumerate(self.filtered_patterns):
                count = solution[i]
                if count > 0:
                    self.solution_list.append({"count": count, **pattern_info})
                    
            # self.probstt = "Solved"
            if prob.status == 1:
                self.probstt = "Solved"
            else: 
                self.probstt = "Infeasible"
        except KeyError: self.probstt = "Infeasible" # khong co nghiem
        
    def cbc_optimize_cut(self):
        # Sets
        # S = list(self.chosen_stocks.keys())
        P = list(range(len(self.filtered_patterns)))

        # PARAMETER - unit weight of stock
        c = {p: self.chosen_stocks[pattern["stock"]]["weight"]/self.chosen_stocks[pattern["stock"]]["width"] for p, pattern in enumerate(self.filtered_patterns)}
        s = {p: self.chosen_stocks[pattern["stock"]]["weight"] for p, pattern in enumerate(self.filtered_patterns)}
        # sp = {(s, p): 1 if self.filtered_patterns[p]["stock"]== s else 0 for p in P for s in S}
        
        # Define VARIABLES
        x = {p: LpVariable(f"X_{p}",lowBound=0, upBound=1, cat='Integer') for p in P}
        
        # Create a LP minimization problem
        prob = LpProblem("PatternCuttingProblem", LpMinimize)
        
        # Objective function: MINIMIZE total stock use considering stock weight    
        prob += LpAffineExpression([(x[p], s[p]) for p in P])

        # Constraints: meet demand for each finished part
        for f in self.dual_finish:
            prob += (
                lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p] * x[p]
                          for p in range(len(self.filtered_patterns))) >= self.dual_finish[f]['need_cut'], 
                f"DemandWeight{f}"
            )
            prob += (
                lpSum(self.filtered_patterns[p]['cuts'][f] * self.dual_finish[f]['width'] * c[p] * x[p]
                          for p in range(len(self.filtered_patterns))) <= self.dual_finish[f]['upper_bound'], 
                f"UpperDemandWeight{f}"
            )
        
        # Solve the problem CBC - Default solver
        prob.solve(PULP_CBC_CMD(msg=False, options=['--solver', 'highs']))
        
        try: # Extract results        
            # Take ALL SUB-OPTIMAL solutions
            for var in [0.5, 0.2, 0.1, 0.01]:
                solution = [1 if x[i].varValue >= var else 0 for i in range(len(self.filtered_patterns))] #trimloss co the rat cao voi varValue lon/stock ratio cao voi varValue thap
                if sum(solution) > 0:
                    break
                else: print(f"found no solution at {var}")
            
            self.solution_list = []
            for i, pattern_info in enumerate(self.filtered_patterns):
                # count = solution[i]
                if solution[i] > 0:
                    self.solution_list.append({"count": solution[i], **pattern_info})
            if sum(solution) > 0:
                self.probstt = "Solved"
            else: self.probstt = "Infeasible"
        except KeyError: self.probstt = "Infeasible" # No result
    
    def _expand_solution_list(self):
        # List to hold the result
        self.expanded_solution_list = []
        # Process each item in the inventory
        for item in self.solution_list:
            count = item['count']
            # Replicate the entry 'count' times >1 to 'count' set to 1
            for _ in range(count):
                new_item = item.copy()  # Copy the item
                new_item['count'] = 1  # Set the count to 1
                self.expanded_solution_list.append(new_item)   
        # print(f"EXPANDED SOLUTION LIST {len(self.expanded_solution_list)}")
    
    def find_final_solution_patterns(self):
        """ 
        Args:
            patterns [
                    {"stock":,'TP238H002948-1', # ID by model, can have "Di1" "Di2"
                    'inventory_id': 'TP238H002948-1', original ID
                    'stock_weight': 4000, 'stock_width': 1219 'trim_loss': 48.0, 'trim_loss_pct': 3.938,
                    'explanation':, "cutting_date': , 'remark':,
                    'cuts': {'F200': 0, 'F198': 3, 'F197': 0, 'F196': 1, 'F190': 4, 'F511': 2, 'F203': 0}
                    }, 
                    {"stock":,'TP238H002948-2',
                    },
                ]
        """
        # self._expand_solution_list()
        sorted_solution_list = sorted(self.solution_list, key=lambda x: (x['stock'], x.get('trim_loss_pct', float('inf')), -x['count_cut']))
        # sorted_solution_list = sorted(self.expanded_solution_list, key=lambda x: (x['stock'], x.get('trim_loss_pct', float('inf')), -x['count_cut']))
        
        self.overused_list = []
        take_stock = None
        for solution_pattern in sorted_solution_list:
            current_stock = solution_pattern['stock']
            if current_stock == take_stock:
                # stock duoc dung truoc do
                self.overused_list.append(solution_pattern)
            else:
                take_stock = current_stock
                self.final_solution_patterns.append(solution_pattern)
    
    # RUN FLOW 4 PHASES    
    def run(self, solver):
        #Phase 1
        self.create_finish_demand_by_line_w_naive_pattern()
        
        #Phase 2
        self.generate_patterns()

        #Phase 3
        self.filter_patterns_and_stocks_by_constr()
        
        #Phase 4
        if solver == "CBC":
            self.cbc_optimize_cut()
        else:
            self.optimize_cut()
        
        if self.probstt == 'Solved':
            self.find_final_solution_patterns()

if __name__ == "__main__":
    finish = {
              "F2": {
                "customer_name": "INT",
                "fg_codes": "CSVC JSH440W-PO 2.30X167XC",
                "width": 167.0,
                "need_cut": -15418.0,
                "standard": "small",
                "cut_standard": "small",
                "fc1": 0.0,
                "fc2": 14117.0,
                "fc3": 9589.82,
                "average FC": 7902.273333333334,
                "1st Priority": "HSC",
                "2nd Priority": "x",
                "3rd Priority": "x",
                "coil_center_priority": "HSC--",
                "Min_weight": 412.0,
                "Max_weight": 1235.0,
                "Min_MC_weight": 3007.353293413174
              },
              "F68": {
                "customer_name": "VPIC1",
                "fg_codes": "CSVC JSH440W-PO 2.30X138XC",
                "width": 138.0,
                "need_cut": -4886.0,
                "standard": "small",
                "cut_standard": "small",
                "fc1": 6712.41,
                "fc2": 7353.18,
                "fc3": 8643.82,
                "average FC": 7569.803333333333,
                "1st Priority": "HSC",
                "2nd Priority":"x",
                "3rd Priority": "x",
                "coil_center_priority": "HSC--",
                "Min_weight": 475.74,
                "Max_weight": 1218.32,
                "Min_MC_weight": 4202.370000000001
              },
              "F75": {
                "customer_name": "VPIC1",
                "fg_codes": "CSVC JSH440W-PO 2.30X150XC",
                "width": 150.0,
                "need_cut": -3049.0,
                "standard": "small",
                "cut_standard": "small",
                "fc1": 5122.99,
                "fc2": 5129.63,
                "fc3": 4897.37,
                "average FC": 5049.996666666666,
                "1st Priority": "HSC",
                "2nd Priority": "x",
                "3rd Priority": "x",
                "coil_center_priority": "HSC--",
                "Min_weight": 517.11,
                "Max_weight": 1324.26,
                "Min_MC_weight": 4202.3806
              },
              "F4": {
                "customer_name": "INT",
                "fg_codes": "CSVC JSH440W-PO 2.30X252XC",
                "width": 252.0,
                "need_cut": -1100.0,
                "standard": "small",
                "cut_standard": "small",
                "fc1": 933.57,
                "fc2": 1106.75,
                "fc3": 752.43,
                "average FC": 930.9166666666666,
                "1st Priority": "HSC",
                "2nd Priority":"x",
                "3rd Priority": "x",
                "coil_center_priority": "HSC--",
                "Min_weight": 621.0,
                "Max_weight": 1400.0,
                "Min_MC_weight": 3003.964285714286
              },
              "F59": {
                "customer_name": "TMW",
                "fg_codes": "CSVC JSH440W-PO 2.30X110XC",
                "width": 110.0,
                "need_cut": -206.0,
                "standard": "small",
                "cut_standard": "small",
                "fc1": 1528.69,
                "fc2": 1050.69,
                "fc3": 1529.33,
                "average FC": 1369.57,
                "1st Priority": "HSC",
                "2nd Priority":"x",
                "3rd Priority": "x",
                "coil_center_priority": "HSC--",
                "Min_weight": "",
                "Max_weight": "",
                "Min_MC_weight": 2282.8545454545456
              }
            }
    stocks = {
        "TP24CH002060": {
          "receiving_date": "2024-10-01T07:00:00",
          "width": 1219.0,
          "weight": 8784,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "R:REWIND",
          "remark": " "
        },
        "TP24AH004786": {
          "receiving_date": "2024-11-04T07:00:00",
          "width": 1219.0,
          "weight": 11175,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004788": {
          "receiving_date": "2024-11-05T07:00:00",
          "width": 1190.0,
          "weight": 11025,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004790": {
          "receiving_date": "2024-11-05T07:00:00",
          "width": 1190.0,
          "weight": 11090,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004800": {
          "receiving_date": "2024-11-06T07:00:00",
          "width": 1105.0,
          "weight": 10575,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004799": {
          "receiving_date": "2024-11-06T07:00:00",
          "width": 1105.0,
          "weight": 10645,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004789": {
          "receiving_date": "2024-11-06T07:00:00",
          "width": 1190.0,
          "weight": 11230,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004787": {
          "receiving_date": "2024-11-08T07:00:00",
          "width": 1190.0,
          "weight": 11185,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24AH004785": {
          "receiving_date": "2024-11-08T07:00:00",
          "width": 1219.0,
          "weight": 11455,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-553"
        },
        "TP24BH000578": {
          "receiving_date": "2024-11-13T07:00:00",
          "width": 1148.0,
          "weight": 10860,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-566"
        },
        "TP24BH000577": {
          "receiving_date": "2024-11-13T07:00:00",
          "width": 1148.0,
          "weight": 10965,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-566"
        },
        "TP24BH000576": {
          "receiving_date": "2024-11-14T07:00:00",
          "width": 1148.0,
          "weight": 9535,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-566"
        },
        "TP24BH000573": {
          "receiving_date": "2024-11-14T07:00:00",
          "width": 1219.0,
          "weight": 10720,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-566"
        },
        "TP24BH000572": {
          "receiving_date": "2024-11-14T07:00:00",
          "width": 1219.0,
          "weight": 10905,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-566"
        },
        "TP24BH000574": {
          "receiving_date": "2024-11-14T07:00:00",
          "width": 1190.0,
          "weight": 10940,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-566"
        },
        "TP24BH002406": {
          "receiving_date": "2024-11-25T07:00:00",
          "width": 1105.0,
          "weight": 9245,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-601"
        },
        "TP24BH002407": {
          "receiving_date": "2024-11-26T07:00:00",
          "width": 1105.0,
          "weight": 9145,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-601"
        },
        "TP24BH002405": {
          "receiving_date": "2024-11-26T07:00:00",
          "width": 1190.0,
          "weight": 11205,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-601"
        },
        "TP24BH002403": {
          "receiving_date": "2024-11-26T07:00:00",
          "width": 1219.0,
          "weight": 11370,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-601"
        },
        "TP24CH000253": {
          "receiving_date": "2024-12-10T07:00:00",
          "width": 1148.0,
          "weight": 11130,
          "min_margin": 6,
          "warehouse": "HSC",
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-633"
        },
        "TP24CH000251": {
          "receiving_date": "2024-12-13T07:00:00",
          "width": 1148.0,
          "weight": 9300,
          "warehouse": "HSC",
          "status": "M:RAW MATERIAL",
          "min_margin": 6,
          "remark": "1C24TNS-633"
        },
        "TP24CH000250": {
          "receiving_date": "2024-12-13T07:00:00",
          "width": 1148.0,
          "weight": 9420,
          "warehouse": "HSC",
          "min_margin": 6,
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-633"
        },
        "TP24CH000252": {
          "receiving_date": "2024-12-13T07:00:00",
          "width": 1148.0,
          "min_margin": 6,
          "weight": 11235,
          "warehouse": "HSC",
          "status": "M:RAW MATERIAL",
          "remark": "1C24TNS-633"
        }
      }
    finish = {f: {**f_info, "upper_bound": -f_info['need_cut'] + f_info['average FC']* 1.0} 
                for f, f_info in finish.items()}
    
    steel = DualProblem(finish, stocks)
    steel.create_finish_demand_by_line_w_naive_pattern()
    steel.generate_patterns()
    steel.filter_patterns_and_stocks_by_constr()
    shorted = [{'stock': p['stock'], 'trimloss': p['trim_loss'],'cuts': p['cuts']} for p in steel.total_dual_pat]
    print(shorted)
    
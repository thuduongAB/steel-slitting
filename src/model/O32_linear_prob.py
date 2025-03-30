import pandas as pd
import numpy as np
import copy

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, LpStatus
from model import FinishObjects, StockObjects

# DEFINE PROBLEM
class LinearProblem:
  # FOR CASE USE INDICATE EXACTLY ONE COIL TO TRY TRIMLOSS
  def __init__(self, stock, finish):
    self.stock = stock # single stock 
    self.finish = finish
    self.skey = list(stock.keys())[0]
    self.over_cut = None
    self.solution = {}
  
  def make_naive_patterns(self):
    """
    Generates patterns of feasible cuts from stock width to meet specified finish widths. not considering the weight contraint
    """
    self.patterns = []
    for f in self.finish:
        feasible = False
        # max number of f that fit on s, bat buoc phai round down vi ko cat qua width duoc
        num_cuts_by_width = int((self.stock[self.skey]["width"]-self.stock[self.skey]["min_margin"]) / self.finish[f]["width"])
        
        # make pattern and add to list of patterns
        if num_cuts_by_width > 0:
          feasible = True
          cuts_dict = {key: 0 for key in self.finish.keys()}
          cuts_dict[f] = num_cuts_by_width
          trim_loss = self.stock[self.skey]['width'] - sum([self.finish[f]["width"] * cuts_dict[f] for f in self.finish.keys()])
          trim_loss_pct = round(trim_loss/self.stock[self.skey]['width'] * 100, 3)
          self.patterns.append({"cuts": cuts_dict, 'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct })
            
        if not feasible :
            pass
            
  def optimize_cut(self):
    # Create the problem
    prob = LpProblem("CuttingOneStock", LpMaximize)

    # Data 
    F = list(self.finish.keys())
    width_s_min_margin = self.stock[self.skey]['width'] - self.stock[self.skey]['min_margin']
    width_f = {f: self.finish[f]["width"] for f in self.finish.keys()}
    # wu = self.stock[0]['weight'] / self.stock[0]['width']
    a_upper_bound = {f: max([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns)]) for f in self.finish.keys()}

    # Decision variables
    a = {f: LpVariable(f'a[{f}]', lowBound=0, upBound=a_upper_bound[f], cat='Integer') for f in F}

    # Objective function: maximize total width
    prob += lpSum(a[f] * width_f[f] for f in F), "TotalWidth"

    # Constraints
    # Feasible pattern min margin
    prob += lpSum(a[f] * width_f[f] for f in F) <= width_s_min_margin, "FeasiblePatternMinMargin"

    # Feasible pattern max margin
    prob += lpSum(a[f] * width_f[f] for f in F) >= 0.96 * self.stock[self.skey]['width'], "FeasiblePatternMaxMargin"
    
    # Solve the problem
    prob.solve()

    try:
      # Extract results
      self.solution = {f: a[f].varValue for f in F}
      self.probstt = "Solved"
    except KeyError: self.probstt = "Infeasible"
  
  def run(self):
    self.make_naive_patterns()
    self.optimize_cut()
  
class CuttingOneStock:
  
  def __init__(self, finish, stock, MATERIALPROPS):
      """
      -- Operator for type One Stock ONLY --
      MATERIALPROPS: {
          "spec_name": "JSH270C-PO",
          "type": "Carbon",
          "thickness": 3.0,
          "maker": "CSVC",
          "code": "CSVC JSH270C-PO 3.0"
      stocks = {"HTV0269/24-D1": {   "receiving_date": 44961,   "width": 1158, 
                                      "weight": 3936,   "warehouse": "NQS"}
      finish = {"F288": {   "customer_name": "VPIC1",   "width": 143.5,  "need_cut": -1300.0,  
                          "fc1": 2724.48,   "fc2": 1776.9600000000003,   "fc3": 1752.9600000000003,  
                          "1st Priority": "HSC",   "2nd Priority": "x",   "3rd Priority": "x",  
                          "Min_weight": 0.0,   "Max_weight": 0.0},
                "F290": {...}
                }
      """
      self.S = StockObjects(stock, MATERIALPROPS)
      self.F = FinishObjects(finish, MATERIALPROPS)
      # self.over_cut = {}
      self.final_solution_patterns = []

  def update(self, bound, margin_df):
    self.S.update_min_margin(margin_df)
    self.F.reverse_need_cut_sign() #from negative to positive
    self.F.update_bound(bound)

  def set_prob(self):
    self.prob = LinearProblem(self.S.stocks ,self.F.finish)
  
  def _calculate_finish_after_cut(self):
    # for all orginal finish, not only dual
    if self.prob.probstt == "Solved":
      cuts_dict = self.prob.solution
      self.weight_dict = {f: round(cuts_dict[f] * self.F.finish[f]['width'] * self.prob.stock[self.prob.skey]['weight']/self.prob.stock[self.prob.skey]['width'],3) for f in cuts_dict.keys()}
      self.over_cut = {k: round(self.weight_dict[k] - self.F.finish[k]['need_cut'],3) for k in self.weight_dict.keys()}
      trim_loss = self.prob.stock[self.prob.skey]['width'] - sum([self.F.finish[f]["width"] * cuts_dict[f] for f in self.F.finish.keys()])
      trim_loss_pct = round(trim_loss/self.prob.stock[self.prob.skey]['width'] * 100, 3)
      pattern = {"stock":self.prob.skey,
                                        'inventory_id': self.prob.skey, 
                                        'stock_weight': self.prob.stock[self.prob.skey]['weight'], 
                                        'stock_width': self.prob.stock[self.prob.skey]['width'], 
                                        'trim_loss': round(trim_loss,3), 
                                        'trim_loss_pct': round(trim_loss_pct,3),
                                          'cuts':cuts_dict,
                                          "fg_code":{f: self.F.finish[f]['fg_codes'] for f in cuts_dict.keys()},
                                          "Customer":{f: self.F.finish[f]['customer_name'] for f in cuts_dict.keys()},
                                          "FG Weight": self.weight_dict,
                                          "FG width": {f: self.F.finish[f]['width'] for f in cuts_dict.keys()},
                                          "standard": {f: self.F.finish[f]['standard'] for f in cuts_dict.keys()},
                                          "Min weight": {f: self.F.finish[f]['Min_weight'] for f in cuts_dict.keys()},
                                          "Max weight": {f: self.F.finish[f]['Max_weight'] for f in cuts_dict.keys()},
                                          "average_fc":{f: self.F.finish[f]['average FC'] for f in cuts_dict.keys()},
                                          '1st Priority':{f: self.F.finish[f]['1st Priority'] for f in cuts_dict.keys()},
                                          'explanation':"", 'cutting_date':"" , 'remarks':""
                                          }
      self.final_solution_patterns.append(pattern)
    else:
       self.weight_dict = {f: 0 for f in self.F.finish.keys()}
       self.over_cut = {f: round(0-self.F.finish[f]['need_cut'],3) for f in self.F.finish.keys()}
       
  def solve_prob(self,flow):
    # Run and calculate results
    self.prob.run()
    self._calculate_finish_after_cut()
    if flow == "change":
      return self.prob.probstt, self.prob.solution, self.weight_dict
    else:
       return self.prob.probstt, self.final_solution_patterns, self.over_cut

     
if __name__ == "__main__":
  margin_df = pd.read_csv('scr/model_config/min_margin.csv')
  spec_type = pd.read_csv('scr/model_config/spec_type.csv')
  
  MATERIALPROP={
            "spec_name": "JSH590R-PO",
            "type": "Carbon",
            "thickness": 2.0,
            "maker": "CSC",
            "code": "CSC JSH590R-PO 2.0"
         }
  stock = {"TP232H001075": { "receiving_date": 44945, "width": 1233, "weight": 9630.0, 
                            "warehouse": "HSC", "status": "M:RAW MATERIAL", "remark": "", 
                            # "min_margin": 10
            }
         }
  finish = {
      "F33": {
         "customer_name": "CIC",
         "width": 188.0,
         "need_cut": -30772.599709771595,
         "fc1": 30646.0820436,
         "fc2": 35762.3146452,
         "fc3": 34039.2591132,
         "average FC": 33482.551933999996,
         "1st Priority": "HSC",
         "2nd Priority": "x",
         "3rd Priority": "x",
         "Min_weight": 0.0,
         "Max_weight": 800.0
      },
      "F32": {
         "customer_name": "CIC",
         "width": 175.0,
         "need_cut": -28574.78588807786,
         "fc1": 26812.20409,
         "fc2": 31288.38713,
         "fc3": 29780.88883,
         "average FC": 29293.826683333333,
         "1st Priority": "HSC",
         "2nd Priority": "x",
         "3rd Priority": "x",
         "Min_weight": 0.0,
         "Max_weight": 800.0
      },
      "F31": {
         "customer_name": "CIC",
         "width": 155.0,
         "need_cut": -4401.8405357987585,
         "fc1": 4832.4321325,
         "fc2": 5639.1860525,
         "fc3": 5367.4857775,
         "average FC": 5279.701320833333,
         "1st Priority": "HSC",
         "2nd Priority": "x",
         "3rd Priority": "x",
         "Min_weight": 0.0,
         "Max_weight": 800.0
      },
      "F29": {
         "customer_name": "CIC",
         "width": 120.0,
         "need_cut": -1751.0,
         "fc1": 2585.511168,
         "fc2": 4319.793456,
         "fc3": 3797.778504,
         "average FC": 3567.694376,
         "1st Priority": "HSC",
         "2nd Priority": "x",
         "3rd Priority": "x",
         "Min_weight": 0.0,
         "Max_weight": 500.0
      }  
  }
  steel = CuttingOneStock(finish,stock,MATERIALPROP)
  steel.update(margin_df)
  # print(steel.S.stocks)
  steel.set_prob()
  print(f"Stock: {steel.prob.stock}")
  probstt, solution, weight_cuts = steel.solve_prob()
  # steel = LinearProb(stock,finish)
  # steel.run()
  if probstt == "Solved":
    print(solution)
    print(weight_cuts)
  
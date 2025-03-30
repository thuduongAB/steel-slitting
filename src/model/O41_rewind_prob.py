import numpy as np
import copy
import statistics
import re
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, LpStatus
# from model import DualProblem
from .O41_dual_solver import DualProblem
import math
    
class RewindProb(DualProblem):
  # have condition to decide proportion of orginal MC to cut # apply for many FG and 1 stock MC
  # then move to the linear problem -> to decide the trim loss only
  def __init__(self, finish, stock):
    self.dual_stocks = copy.deepcopy(stock)
    self.start_stocks = copy.deepcopy(stock)
    self.dual_finish = copy.deepcopy(finish)
    self.start_finish = copy.deepcopy(finish)
    self.final_solution_patterns = []
    self.stock_key = list(stock.keys())[0]
    self.og_width = stock[self.stock_key]['width']
    self.og_weight = stock[self.stock_key]['weight']
    self.stock = stock
  
  def _make_naive_patterns(self):
    """
    Generates patterns of feasible cuts from stock width to meet specified finish widths. not considering the weight contraint
    """
    self.patterns = []
    for f in self.dual_finish:
        feasible = False
        # max number of f that fit on s, bat buoc phai round down vi ko cat qua width duoc
        num_cuts_by_width = int((self.stock[self.stock_key]["width"]-self.stock[self.stock_key]["min_margin"]) / self.dual_finish[f]["width"])
        
        # make pattern and add to list of patterns
        if num_cuts_by_width > 0:
          feasible = True
          cuts_dict = {key: 0 for key in self.dual_finish.keys()}
          cuts_dict[f] = num_cuts_by_width
          trim_loss = self.stock[self.stock_key]['width'] - sum([self.dual_finish[f]["width"] * cuts_dict[f] for f in self.dual_finish.keys()])
          trim_loss_pct = round(trim_loss/self.stock[self.stock_key]['width'] * 100, 3)
          self.patterns.append({"cuts": cuts_dict, 'trim_loss':trim_loss, "trim_loss_pct": trim_loss_pct })
            
        if not feasible :
            pass
  
  def _optimize_cut(self):
    # Create the problem
    prob = LpProblem("CuttingOneStock", LpMaximize)

    # Data 
    F = list(self.dual_finish.keys())
    width_s_min_margin = self.stock[self.stock_key]['width'] - self.stock[self.stock_key]['min_margin']
    width_f = {f: self.dual_finish[f]["width"] for f in self.dual_finish.keys()}
    a_upper_bound = {}
    for f in self.dual_finish.keys():
        try:
            a_upper_bound[f] = max([self.patterns[i]['cuts'][f] for i, _ in enumerate(self.patterns)])
        except ValueError:
            a_upper_bound[f] = None  # Or any default value you prefer
          
    # Decision variables
    a = {f: LpVariable(f'a[{f}]', lowBound=0, upBound=a_upper_bound[f], cat='Integer') for f in F}

    # Objective function: maximize total width
    prob += lpSum(a[f] * width_f[f] for f in F), "TotalWidth"

    # Constraints
    # Feasible pattern min margin
    prob += lpSum(a[f] * width_f[f] for f in F) <= width_s_min_margin, "FeasiblePatternMinMargin"

    # Feasible pattern max margin
    prob += lpSum(a[f] * width_f[f] for f in F) >= 0.96 * self.stock[self.stock_key]['width'], "FeasiblePatternMaxMargin"
    
    # Solve the problem
    prob.solve()

    try:
      # Extract results
      trim_loss_pct = (self.stock[self.stock_key]['width'] - sum([self.dual_finish[f]["width"] * round(a[f].varValue) for f in F]))/self.stock[self.stock_key]['width']
      if 0 < trim_loss_pct <= 0.04:
        self.optimal_pattern = {f: round(a[f].varValue) for f in F}
        self.probstt = "Linear Solved"
        print(f"rewind solved pattern {self.optimal_pattern}")
      else: self.probstt = "Infeasible"
    except KeyError: self.probstt = "Infeasible"
    
  def _rewind_ratio(self):
    # xac dinh ratio da phai tinh den weight cat
    try:
      coil_weight_by_needcut = [self.dual_finish[f]["need_cut"] * self.stock[self.stock_key]['width'] /(self.dual_finish[f]["width"]*self.optimal_pattern[f]) for f in self.dual_finish.keys()]
      coil_weight_by_cut= [self.dual_finish[f]["width"] * self.optimal_pattern[f]*self.stock[self.stock_key]['weight']/self.stock[self.stock_key]['width'] for f in self.dual_finish.keys()]
      rewind_ratio_arr = [x / y for x, y in zip(coil_weight_by_needcut, coil_weight_by_cut)]
      self.rewind_ratio = np.percentile(rewind_ratio_arr, 75) # cho phep cat du 1 chut
      self.rewind_weight = self.rewind_ratio * self.og_weight
      if self.rewind_ratio < 1:
        pass
      else: 
        self.probstt = "Infeasible"
        print("Infeasible")
    except ZeroDivisionError: 
      self.probstt = "Infeasible"
      print("ZeroDivisionErrorInfeasible")

  
  def create_new_stocks_set(self):
    self._make_naive_patterns()
    self._optimize_cut()
    if self.probstt == "Linear Solved":
      self._rewind_ratio()
      if self.probstt == "Solved":
        remained_coil_weight = self.og_weight - self.rewind_weight
        #create new set stock if remained weight not to small
        if remained_coil_weight > 2500:
          for i in range(2):
            self.dual_stocks[f'{self.stock_key}-Re{i+1}'] = self.stock[self.stock_key]
            if i < 1: 
              self.dual_stocks[f'{self.stock_key}-Re{i+1}'].update({'weight': self.rewind_weight})
              # print(f"cut rewind weight {self.med_demand_weight}")
            else: 
              self.dual_stocks[f'{self.stock_key}-Re{i+1}'].update({'weight': remained_coil_weight}) # we have new set of stock
            self.dual_stocks[f'{self.stock_key}-Re{i+1}'].update({'status':"R:REWIND"})
      
          self.start_stocks = copy.deepcopy(self.dual_stocks)
          self.start_stocks.pop(self.stock_key)
        else: 
          print(f"rewind_coil too small{remained_coil_weight}")
    
  def run(self):
    if self.probstt == "Solved":
      trim_loss = self.dual_stocks[self.stock_key]['width'] - sum([self.dual_finish[f]["width"] * self.optimal_pattern[f] for f in self.dual_finish.keys()])
      trim_loss_pct = round(trim_loss/self.dual_stocks[self.stock_key]['width'] * 100, 3)
      self.final_solution_patterns = [{"stock":f'{self.stock_key}-Re{1}',
                                    "inventory_id": self.stock_key,
                                    'stock_weight': self.rewind_weight, 
                                    'stock_width': self.og_width,
                                    "cuts": self.optimal_pattern,
                                    "trim_loss":trim_loss,
                                    "trim_loss_pct":trim_loss_pct,
                                    "explanation":f"cut Rewind on original MC {self.og_weight}",
                                    "cutting_date":"",
                                    'remarks':"",
                                    }]
    
    

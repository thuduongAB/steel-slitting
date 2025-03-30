### LOAD FINISH AND MOTHER COIL BY PARAMS - CUSTOMER
import pandas as pd
import numpy as np
import math
import copy
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, LpStatus
from model import FinishObjects, StockObjects
# from O31_steel_objects import FinishObjects, StockObjects

# DEFINE PROBLEM
class SemiProb():
  def __init__(self, stocks, finish,MATERIALPROPS):
    self.S = StockObjects(stocks, MATERIALPROPS)
    self.F = FinishObjects(finish, MATERIALPROPS)
    self.skey = list(stocks.keys())[0]
    self.fkey = list(finish.keys())[0]
    self.taken_stocks ={}
    
  def _max_loss_margin_by_wh(self,margin_df):
    # xac dinh max margin cho phep voi loai Z:SEMI MCOIL
    wh_list = margin_df['Coil Center'].unique().tolist()
    self.max_margin_semi ={}
    for wh in wh_list:
      wh_df = margin_df[margin_df['Coil Center'] == wh]
      self.max_margin_semi[wh] = max(wh_df['Min Trim loss (mm)']) # ap dung cho th khong biet width coil goc 
      
  def _set_stock_finish(self):
    self.finish = self.F.finish
    self.stock = self.S.stocks
    self.og_stock_width = copy.deepcopy(self.stock[self.skey]['width'])
  
  def update(self, margin_df):
    self.F.reverse_need_cut_sign()
    self.F.update_bound(1.5) # take max bound = 3
    self.S.update_min_margin(margin_df)
    self._set_stock_finish()
    self.remained_stocks= self.stock
    self._max_loss_margin_by_wh(margin_df)
  
  def _cut_patterns(self):
    try:
      # Calculate the upper bound value
      upper_bound = self.finish[self.fkey]['upper_bound']
      print(f"test upper b {upper_bound}")

      # Calculate the weight factor
      weight_factor = self.stock[self.skey]['weight'] * self.finish[self.fkey]['width'] / self.stock[self.skey]['width']
      print(f"test semi line cut {round(upper_bound / weight_factor)}")
      
      # Calculate the number of cuts by weight
      if round(upper_bound / weight_factor)== 0:
        # print(f"test semi line cut {round(upper_bound / weight_factor)}")
        self.num_cuts_by_weight = 1
      elif round(upper_bound / weight_factor) > 5:
        self.num_cuts_by_weight = math.ceil(self.finish[self.fkey]['need_cut'] / weight_factor)
      else:
        self.num_cuts_by_weight = round(upper_bound / weight_factor)
      print(f"num cut by weight{self.num_cuts_by_weight}")
    except Exception:
      print(f"upper_bound {upper_bound} weigh_factor {weight_factor}")

    self.num_cuts_by_width = int(
          (self.stock[self.skey]["width"] - self.stock[self.skey]["min_margin"]) / 
          self.finish[self.fkey]["width"]
      )
  
  # chua margin bang 1/2 margin goc
  def _semi_cut_ratio(self):
    if self.stock[self.skey]['status'] == "Z:SEMI MCOIL": # cat tiep tu 1 coil semi, ap dung truong hop ko co remark do model generate tu truoc
      print("cat tu cuon SEMI") # allowed margin?
      # check margin con lai < max margin ko/// BUOC PHAI CAT HET ?
      margin = self.stock[self.skey]["width"] - self.num_cuts_by_width * self.finish[self.fkey]["width"]
      wh = self.stock[self.skey]['warehouse']
      if margin < (self.max_margin_semi[wh]*2):
        self.cuts_dict = {str(self.fkey): self.num_cuts_by_width}
        self.remained_stocks = {}
        self.taken_stocks = self.stock
      else: 
        self.cuts_dict = {str(self.fkey): 0}
        self.remained_stocks = self.stock
      
      cut_line = min(self.num_cuts_by_weight, self.num_cuts_by_width)
      self.cut_width = cut_line * self.finish[self.fkey]['width']
      self.remained_cuts = self.num_cuts_by_width - self.num_cuts_by_weight
      self.remain_width = self.stock[self.skey]['width'] - self.cut_width - (self.stock[self.skey]['min_margin']/2)
    elif self.stock[self.skey]['status'] == "M:RAW MATERIAL" or self.stock[self.skey]['status'] == "R:REWIND":  #hoac cat ra tu Mother coil phan biet bang status.
      print("cat tu cuon RAW MC/REWIND")
      cut_line = min(self.num_cuts_by_weight, self.num_cuts_by_width)
      self.cut_width = cut_line * self.finish[self.fkey]['width']
      self.remained_cuts = self.num_cuts_by_width - self.num_cuts_by_weight
      self.remain_width = self.stock[self.skey]['width'] - self.cut_width - (self.stock[self.skey]['min_margin']/2)  # chua bien 1 ben
    else: 
      pass
    
  def _check_remain_width(self):
    # case nay giong nhu check Z:SEMI MCOIL, ghi lai Remark cach cat nhu dict-cut
    # wh = self.stock[self.skey]['warehouse']
    return (self.remain_width > self.stock[self.skey]['min_margin'] )
  
  def cut_n_create_new_stock_set(self):
    self._cut_patterns()
    self._semi_cut_ratio() # cut_dict cho loai Z: SEMI
    if (self.stock[self.skey]['status'] == "M:RAW MATERIAL" or self.stock[self.skey]['status'] == "R:REWIND") and self._check_remain_width():
      cut_line = min(self.num_cuts_by_weight, self.num_cuts_by_width)
      self.cuts_dict = {str(self.fkey): cut_line}
      self.cut_weight = cut_line * self.finish[self.fkey]['width'] * self.stock[self.skey]['weight'] /self.stock[self.skey]['width']
      self.over_cut = {str(self.fkey): round(self.cut_weight - self.finish[self.fkey]['need_cut'],3)}
      self.taken_stocks = {f'{self.skey}-Se1':{"receiving_date": self.stock[self.skey]['receiving_date'],
                                                "width": cut_line * self.finish[self.fkey]['width'] + self.stock[self.skey]['min_margin']/2,
                                                "weight": (cut_line * self.finish[self.fkey]['width']+ self.stock[self.skey]['min_margin']/2)/self.stock[self.skey]['width']*self.stock[self.skey]['weight'],
                                                "warehouse": self.stock[self.skey]['warehouse'],
                                                'status': "Z:SEMI FINISHED",
                                                "remarks":""}}
    
      print(f"taken semi stock :{self.taken_stocks}") 
      
      self.remained_stocks = {f'{self.skey}-Se2':{"receiving_date": self.stock[self.skey]['receiving_date'],
                                                  "width": self.remain_width,
                                                  "weight": self.remain_width/self.stock[self.skey]['width']*self.stock[self.skey]['weight'],
                                                  "warehouse": self.stock[self.skey]['warehouse'],
                                                  'status': "Z:SEMI MCOIL",
                                                  "remarks":f"remained_cut: {self.fkey}:{self.remained_cuts}"}}
      
if __name__ == "__main__":
  MATERIALPROPS = {
            "spec_name": "JSC270C-SD",
            "type": "Carbon",
            "thickness": 2.0,
            "maker": "POSCOVN",
            "code": "POSCOVN JSC270C-SD 2.0"
         }
  stocks = {
            "HTV1766/24": {
               "receiving_date": 45514,
               "width": 938,
               "weight": 8325.0,
               "warehouse": "NQS",
               "status": "M:RAW MATERIAL",
               "remark": ""
            },
            "HTV1766": {
               "receiving_date": 45514,
               "width": 938,
               "weight": 5025.0,
               "warehouse": "NQS",
               "status": "M:RAW MATERIAL",
               "remark": ""
         }}
  finish = {"F524": {
                     "customer_name": " LEG ",
                     "width": 110.0,
                     "need_cut": -2000.0,
                     "fc1": 1693.624,
                     "fc2": 1320.9008000000001,
                     "fc3": 1507,
                     "average FC": 1507.2624,
                     "1st Priority": "NQS",
                     "2nd Priority": "HSC",
                     "3rd Priority": "HSC",
                     "Min_weight": 0,
                     "Max_weight": 0
                  }
               }
  margin_df = pd.read_csv('scr/model_config/min_margin.csv')
  spec_type = pd.read_csv('scr/model_config/spec_type.csv')
  
  steel = SemiProb(stocks, finish, MATERIALPROPS)
  steel.update(margin_df)
  steel.cut_n_create_new_stock_set()
  print(f"cuts: {steel.cuts_dict}")
  print(f"taken stocks: {steel.taken_stocks}")
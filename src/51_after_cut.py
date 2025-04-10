import pandas as pd
import os
from functools import reduce
import datetime

### --- PARAMETER SETTING ---
# uat = int(os.getenv('UAT', '0000'))
# uat = int(1140)
uat_list = [1135,1136,1137,1138,1139,1140,
            1141,1142,1143,1144,1145,1146,1147,1148,1149,
            1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161]
for uat in uat_list:
    merge = pd.read_excel(f"results/Merged_ResultUAT_{uat}.xlsx")
    finish_df = pd.read_excel(F"data/finish_uat_{uat}.xlsx")
    month = finish_df['Month'][0]

    all_mc = pd.read_excel(f"data/mother_coil_uat_{month}.xlsx")
    all_mc = all_mc[['inventory_id','MC Code','weight']]

    mothercoils = merge[['inventory_id','trim_loss','trim_loss_pct']].drop_duplicates()
    mothercoils.dropna(axis=0,inplace=True)

    re_mothercoils = pd.merge(mothercoils, all_mc, on="inventory_id", how='left')
    re_mothercoils['trim_loss_wg'] = re_mothercoils['weight'] * re_mothercoils['trim_loss_pct']/100

    final_mc = re_mothercoils.groupby('MC Code')[['trim_loss_wg','weight']].sum().reset_index()
    final_mc['trim loss pct'] = final_mc['trim_loss_wg']/final_mc['weight']*100


    cut_df = merge.groupby(['Customer','fg_code'])['FG Weight'].sum().reset_index()
    cut_df["Customer_FG"] = (cut_df['Customer'].astype(object) +"-"+ cut_df['fg_code'].astype(object))
    cut_df.drop(columns=['Customer','fg_code'],inplace=True)

    final_after_cut = pd.merge(finish_df, cut_df, on="Customer_FG", how='left')

    final_after_cut = final_after_cut[~(final_after_cut['need_cut']>=0 & final_after_cut['FG Weight'].isna())]
    final_after_cut['FG Weight'] = final_after_cut['FG Weight'].fillna(0)
    final_after_cut['Stock After Cut'] = final_after_cut['need_cut']+final_after_cut['FG Weight']
    final_after_cut['Stock Ratio After Cut'] = final_after_cut['Stock After Cut']/(final_after_cut['average FC']+1)
    final_after_cut['Month'] = month
    final_after_cut['Year'] = datetime.datetime.today().year

    # MC code 
    count_need_cut_by_mc = final_after_cut[final_after_cut['need_cut']<0].groupby('MC Code')['need_cut'].count().reset_index()
    count_need_cut_by_mc.rename(columns={"need_cut":"count_need_cut"},inplace=True)

    need_cut_by_mc = final_after_cut.groupby('MC Code')[['need_cut','average FC']].sum().reset_index()
    total_cut_by_mc = final_after_cut.groupby('MC Code')['FG Weight'].sum().reset_index()

    after_cut_positive_by_mc = final_after_cut[final_after_cut['Stock After Cut']>=0].groupby('MC Code')['Stock After Cut'].sum().reset_index()
    after_cut_positive_by_mc.rename(columns={'Stock After Cut':'pos_after_cut_wg'},inplace=True)

    after_cut_neg_by_mc = final_after_cut[final_after_cut['Stock After Cut']<0].groupby('MC Code')['Stock After Cut'].sum().reset_index()
    after_cut_neg_by_mc.rename(columns={'Stock After Cut':'neg_after_cut_wg'},inplace=True)

    dfs = [count_need_cut_by_mc,need_cut_by_mc, total_cut_by_mc, after_cut_positive_by_mc, after_cut_neg_by_mc, final_mc]

    # Merge DataFrames using reduce
    after_cut_by_mc = reduce(lambda left, right: pd.merge(left, right, on='MC Code', how='outer'), dfs)

    ####    SAVE EXCEL ###
    final_after_cut.rename(columns={"FG Weight":"Slit Qt",
                                    'average FC':"Average Forecast",
                                    "fg_codes":"FG code",
                                    'customer_name':"Customer",
                                    'need_cut':"Need Cut"},inplace=True)
    final_after_cut.to_excel(f'results/after_cut_by_fg_{uat}.xlsx',index=False)
    after_cut_by_mc.to_excel(f'results/after_cut_by_mc_{uat}.xlsx',index=False)
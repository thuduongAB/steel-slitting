import pandas as pd
import os

### --- PARAMETER SETTING ---
# uat = int(os.getenv('UAT', '0000'))
# uat = int(1141)
uat_list = [
            1135,1136,1137,1138,1139,1140,
            1141,1142,1143,1144,1145,1146,1147,1148,
            1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,
            1161
            ]
for uat in uat_list:
            
    # Set the directory containing the Excel files
    folder_path = 'results'
    # Get a list of all Excel files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and f.startswith(f"UATresult-{uat}")]
    # Initialize an empty list to hold dataframes
    df_list = []
    # Loop through each Excel file and read it into a dataframe
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)
        df_list.append(df)
    # Concatenate all dataframes into one
    merged_df = pd.concat(df_list, ignore_index=True)
    column_order = ['stock', 'inventory_id', 'stock_weight', 'stock_width',  
                    'receiving_date','explanation', 'remarks', 'cutting_date',
                    'trim_loss', 'trim_loss_pct', 'fg_code', 'Customer', 'FG Weight',
                    'FG width', 'standard', 'Min weight', 'Max weight', 'average_fc',
                    '1st Priority', 'cuts', 'lines', 'time']
    merged_df = merged_df[column_order]
    # Save the merged dataframe to a new Excel file
    merged_df.to_excel(os.path.join(folder_path, f'Merged_ResultUAT_{uat}.xlsx'), index=False)

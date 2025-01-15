

import os
import sys
import numpy as np
import pandas as pd

backtesting_path = r'C:\Users\vinayak\Desktop\Backtesting'
if backtesting_path not in sys.path:
    sys.path.append(backtesting_path)

from Modules import Data_Processing as dp

def merge_pnl_files(root_folder):
    summary_pnl_path = os.path.join(root_folder, "summary_pnl.csv")
    trades_margin_path = os.path.join(root_folder, "trades_margin.csv")
    all_pnl_data = []
    all_margin_data = []

    # Loop through all folders in the root directory
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            pnl_file_path = os.path.join(folder_path, "PNL.csv")
            margin_file_path = os.path.join(folder_path, 'margin.csv')

            # Check if PNL.csv exists in the folder
            if os.path.exists(pnl_file_path):
                try:
                    # Read the PNL.csv file
                    pnl_data = pd.read_csv(pnl_file_path, parse_dates=True, index_col=0)
                    all_pnl_data.append(pnl_data)
                except Exception as e:
                    print(f"Error reading {pnl_file_path}: {e}")

            if os.path.exists(margin_file_path):
                try:
                    margin_data = pd.read_csv(margin_file_path)
                    all_margin_data.append(margin_data)
                except Exception as e:
                    print(f"Error reading margin data {margin_file_path}: {e}")


    # Combine all PNL data into a single DataFrame
    if all_pnl_data:
        summary_pnl = pd.concat(all_pnl_data)

        # Save the summary to a new CSV file
        summary_pnl.to_csv(summary_pnl_path)
        print(f"Summary PNL saved to: {summary_pnl_path}")
    else:
        print("No PNL.csv files found!")

    if all_margin_data:
        total_margin = pd.concat(all_margin_data)
        total_margin.to_csv(trades_margin_path)
        print(f"All Trades Margin Saved to: {trades_margin_path}")
    else:
        print("No margin.csv file found")
    
    return summary_pnl_path

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python python_script.py <path> <fund_blocked> <risk_free_rate>")
        sys.exit(1)

    path = sys.argv[1]    
    merge_pnl_files(path)
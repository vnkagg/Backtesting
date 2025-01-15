

import os
import sys
import numpy as np
import pandas as pd
from Modules import Plot

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

def normalize(dff, col):
    df = dff.copy()
    running_pnl_agg = 0
    flag = True

    for idx in df.index:  # Iterate over the index (timestamps)
        if pd.isna(df.loc[idx, col]):  # Access row via timestamp
            if flag:
                prev_idx = df.index[df.index.get_loc(idx) - 1] if df.index.get_loc(idx) > 0 else None
                running_pnl_agg = df.loc[prev_idx, col] if prev_idx else 0
                flag = False
            df.loc[idx, col] = running_pnl_agg
        else:
            df.loc[idx, col] += running_pnl_agg
            flag = True
    return df


def get_metrics(daily_profit, fund_blocked, risk_free_rate):
    daily_profit.index = pd.to_datetime(daily_profit.index)
    metrics_df = {}
    metrics_df['Fund/Margin (rs)'] = fund_blocked
    metrics_df['Risk Free Rate'] = risk_free_rate

    # ret1 = (result_df['Running PNL (without expenses)'].iloc[-1] - result_df['Cumulative Expenses'].iloc[-1])/fund_blocked * 100
    
    ret = (daily_profit.sum())/fund_blocked * 100
    # print(f"Return: {np.round(ret1, 2)}%")
    print(f"Return: {np.round(ret, 2)}%")
    metrics_df['Return'] = ret

    print(f"Total Profit: {np.round(daily_profit.sum(), 2)} | Average Daily Profit: {np.round(daily_profit.mean(), 2)}")
    print(f"Risk Free profit: {risk_free_rate * fund_blocked} | Daily Risk Free Profit: {np.round(risk_free_rate * fund_blocked/ 365, 2)}")
    print(f"Standard Deviation of Daily Profits: {np.round(daily_profit.std(), 2)}")

    sharpe_value = (daily_profit.mean() - risk_free_rate*fund_blocked/365)/daily_profit.std() * np.sqrt(252)
    print(f"Sharpe: {np.round(sharpe_value, 2)}")
    metrics_df['Sharpe'] = sharpe_value

    sortino_value = (daily_profit.mean() - risk_free_rate*fund_blocked/365)/daily_profit[daily_profit < 0].std() * np.sqrt(252)
    print(f"Sortino: {np.round(sortino_value, 2)}")
    metrics_df['Sortino'] = sortino_value

    information_ratio = (daily_profit[daily_profit != 0].mean())/daily_profit[daily_profit != 0].std() * np.sqrt(252)

    print(f"Information Ratio: {np.round(information_ratio, 2)}")
    metrics_df['Information Ratio'] = information_ratio

    cum_daily_profit = daily_profit.cumsum()

    df_plt = cum_daily_profit.to_frame()
    df_plt.to_csv('sanity_cum_profit.csv')
    plt = Plot.plot_df(df_plt, *df_plt.columns)
    Plot.save_plot(plt, 'CummulativeProfitGraph.html')

    peaks = cum_daily_profit.cummax()
    peak = peaks.max()
    peaks = peaks[peaks >= peak]
    cum_daily_profit = cum_daily_profit[peaks.index]

    # cum_daily_profit = cum_daily_profit[peak_idx:]
    # peaks = peaks[peak_idx:]
    
    drawdowns = (cum_daily_profit - peaks)/(peaks) * 100
    max_drawdown = drawdowns.min()
    print(f"Drawdown: {np.round(max_drawdown, 2)}%")
    metrics_df['Drawdown'] = max_drawdown
    

    # monthly_data = daily_profit.resample('ME').sum()
    # monthly_profit = monthly_data[monthly_data > 0]
    # monthly_loss = monthly_data[monthly_data < 0]

    # if len(monthly_profit):
    #     monthly_profit = monthly_profit.sum()/monthly_profit.count()
    #     print("Average Monthly Profit:", np.round(monthly_profit, 2))
    #     metrics_df['Monthly Profit'] = monthly_profit
    # else:
    #     profit_loss_ratio = 0
    # if len(monthly_loss):
    #     monthly_loss = abs(monthly_loss.sum())/monthly_loss.count()
    #     print("Average Monthly Loss:", np.round(monthly_loss, 2))
    #     metrics_df['Monthly Loss'] = monthly_loss

    #     profit_loss_ratio = monthly_profit/monthly_loss
    # else:
    #     profit_loss_ratio = float('inf')
    
    # print("Monthly Profit/Loss Ratio:", np.round(profit_loss_ratio, 2))
    # metrics_df['Monthly Profit/ Monthly Loss'] = profit_loss_ratio

    metrics_df['Total Profit'] = daily_profit.sum()
    metrics_df['Risk Free Profit'] = len(daily_profit) * fund_blocked * risk_free_rate/365


    windays = (daily_profit > 0).sum()
    lossdays = (daily_profit < 0).sum()
    if (windays + lossdays) > 0:
        win_ratio = windays / (windays + lossdays)
    else:
        win_ratio = 0
    print(f"Win Ratio: {np.round(win_ratio, 2)*100}%")
    metrics_df['Win Days'] = windays
    metrics_df['Lose Days'] = lossdays
    metrics_df['Win Ratio'] = win_ratio
    
    print(f"Number of Traded Days: {len(daily_profit[daily_profit != 0])}")
    metrics_df['Number of Traded Days'] = len(daily_profit[daily_profit != 0])


def metrics(path, fund_blocked, risk_free_rate=0.1):
    # info_metrics = open(f'INFO_Metrics_for_{strategy_desc}.txt', 'w')
    # sys.stdout = info_metrics

    metrics_path = os.path.join(path, "metrics.xlsx")
    pnl_path = os.path.join(path, 'summary_pnl.csv')

    result_df = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
    result_df = result_df.groupby(result_df.index).agg({
        'Running PNL (without expenses)': 'sum',
        'Cumulative Expenses': 'sum',
        'Running PNL': 'sum'
        })
        
    

    _, ci = dp.get_continuous_excluding_market_holidays(result_df)
    result_df = result_df.reindex(ci)

    print(f"Fundblocked in rs: {fund_blocked}")
    


    result_df = normalize(result_df, 'Running PNL (without expenses)')
    result_df = normalize(result_df, 'Cumulative Expenses')
    
    result_df.index = pd.to_datetime(result_df.index)
    result_df['date'] = result_df.index.date  
    
    # daily_profit = result_df.groupby('date')['Running PNL (without expenses)'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    # daily_profit -= result_df.groupby('date')['Cumulative Expenses'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    running_pnl_last = result_df.groupby('date')['Running PNL (without expenses)'].last()
    cumulative_expenses_last = result_df.groupby('date')['Cumulative Expenses'].last()
    daily_profit = running_pnl_last - running_pnl_last.shift(1).fillna(0)
    daily_profit -= (cumulative_expenses_last - cumulative_expenses_last.shift(1).fillna(0))
    

    metrics_df = get_metrics(daily_profit, fund_blocked, risk_free_rate)

    metrics_df_monthly = {}
    # monthly_pnl_aggregated = {}
    # for last_date, monthly_pnl in daily_profit.resample("ME"):
    #     monthly_pnl_aggregated[last_date.strftime('%B')] = monthly_pnl.sum()

    for last_date, monthly_pnl in daily_profit.resample("ME"):
        month = last_date.strftime('%B')
        metrics_df_monthly[month] = get_metrics(monthly_pnl, fund_blocked, risk_free_rate)
        # metrics_df_monthly[month]['Sharpe'] = (monthly_pnl_aggregated[month] - fund_blocked * risk_free_rate/12)/pd.Series(monthly_pnl_aggregated.values()).std()
        # metrics_df_monthly[month]['Information Ratio'] = (monthly_pnl_aggregated[month])

    metrics_output = pd.DataFrame(metrics_df.items(), columns=['Metric', 'Value'])

    with pd.ExcelWriter(metrics_path) as writer:
        metrics_output.to_excel(writer, sheet_name='Metrics', index=False)
        result_df.to_excel(writer, sheet_name='Normalized Data')
        daily_profit.to_frame(name='Daily Profit').to_excel(writer, sheet_name='Daily Profit')
        running_pnl_last.to_frame(name='Running PNL (Daily)').to_excel(writer, sheet_name='Running PNL')
        cumulative_expenses_last.to_frame(name='Cumulative Expenses (Daily)').to_excel(writer, sheet_name='Cumulative Expenses')
    print(f"Metrics saved to: {metrics_path}")





if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python python_script.py <path> <fund_blocked> <risk_free_rate>")
        sys.exit(1)

    path = sys.argv[1]
    fund = int(sys.argv[2])
    rf = float(sys.argv[3]) / 100
    
    # merge_pnl_files(path)
    metrics(path, fund, rf)
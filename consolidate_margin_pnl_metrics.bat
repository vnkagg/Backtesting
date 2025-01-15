@echo off

REM Get the current working directory (PWD)
set "pwd=%cd%"

python C:\Users\vinayak\Desktop\Backtesting\get_margin_file.py %pwd%

REM Prompt user for fund and store the input
set /p fund="Enter Fund/Margin: "

REM Prompt user for risk-free rate and store the input
set /p risk_free="Enter Risk-Free Rate (percentage): "

REM Run the Python script and pass the inputs
REM Replace `python_script.py` with the actual name of your Python file
python C:\Users\vinayak\Desktop\Backtesting\get_pnl_metrics.py %pwd% %fund% %risk_free%

pause

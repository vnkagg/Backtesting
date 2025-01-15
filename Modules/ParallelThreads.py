
import subprocess
from concurrent.futures import ThreadPoolExecutor
class ThreadsForScript:
    def __init__(self, path, max_threads=5):
        self.file = path
        self.max_threads = max_threads
        # r"C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\CheckTradable\check_performance.py"

    # Function to simulate calling check_performance.py
    def run_script(self, *parameters):
        try:
            # Call the check_performance.py script with inputs
            process = subprocess.run(
                ["python", self.file],
                input = "\n".join(parameters) ,
                text=True,  # Ensure inputs are passed as text
                capture_output=True,  # Capture stdout and stderr
                check=False  # Do not raise an exception for non-zero exit codes
            )
            
            # Handle successful execution
            if process.returncode == 0:
                print(f"Task completed successfully for {", ".join(parameters)}")
                print(process.stdout)
            else:
                # Handle errors from the subprocess
                print(f"Error occurred for {"\n".join(parameters)}")
                print(f"Return code: {process.returncode}")
                print(process.stderr)

        except Exception as e:
            print(f"Unexpected error while running check_performance.py: {e}")



    # Function to check performance for multiple trades
    def run_threads(self, parameters_for_processes):
        with ThreadPoolExecutor(self.max_threads) as executor:
            # Submit tasks to the thread pool
            futures = [
                executor.submit(self.run_script, *parameters)
                for parameters in parameters_for_processes
            ]

            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()  # Raises exceptions if any occurred in the thread
                except Exception as e:
                    print(f"Error in thread execution: {e}")
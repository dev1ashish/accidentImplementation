import os
from dotenv import load_dotenv
import subprocess
import time

# Load environment variables from .env file
load_dotenv()

def run_command(command):
    return subprocess.Popen(command, shell=True)

# Run backend services
backend_services = [
    "python RunMaster.py",
    "python RunDetect.py",
    "python RunTracker.py",
    "python RunCrash.py"
]

# Run client services
client_services = [
    "python RunCamera.py"
]

# Start backend services
backend_processes = [run_command(service) for service in backend_services]

# Wait a bit for backend services to initialize
time.sleep(5)

# Start client services
client_processes = [run_command(service) for service in client_services]

# Wait for all processes to complete
for process in backend_processes + client_processes:
    process.wait()

from Debug import MainFlow

def run_argus():
    # Your existing code here
    
    # After all processes are complete
    main_flow = MainFlow()
    main_flow.run("path/to/your/video.mp4")
    main_flow.calculate_metrics()

if __name__ == "__main__":
    run_argus()
    
    
    
    # 1518, 1521, 1552, 1566
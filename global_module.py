import config
from utils import connections
import time
import subprocess

db = connections.OSRS_DB(config.OSRS_GE_DB)

# Launch the streamlit app as a subprocess
process = subprocess.Popen(["streamlit", "run", "main.py"])  # Replace 'main.py' with your app filename

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # If you press Ctrl+C in the parent process or trigger a condition, kill the subprocess
    db.closeConnection()
    process.terminate()
    process.wait()
    print("Streamlit app has been closed.")
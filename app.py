import config
from utils import connections
import time
import subprocess

db = connections.OSRS_DB(config.OSRS_GE_DB)

# Launch the streamlit app as a subprocess
process = subprocess.Popen(["streamlit", "run", "server.py"])

# Main engine loop. Wait for manual interupt
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Close connection to db before exiting app
    db.closeConnection()

    # Terminate app process
    process.terminate()

    # Log message to confirm app is closed
    process.wait()
    print("Streamlit app has been closed.")
import time
import subprocess

# Launch the streamlit app as a subprocess
process = subprocess.Popen(["streamlit", "run", "server.py"])

# Main engine loop. Wait for manual interupt
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Terminate app process
    process.terminate()

    # Log message to confirm app is closed
    process.wait()
    print("Streamlit app has been closed.")
# run_streamlit.py
import subprocess
import sys
import os

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run streamlit
subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"])
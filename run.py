#!/usr/bin/env python
"""
SynthiQuery - AI PDF Chat
Run script to start the Streamlit application
"""

import os
import sys
import subprocess
import platform
import webbrowser
from pathlib import Path

def check_env_file():
    """Check if .env file exists and create it if not"""
    env_path = Path('.env')
    if not env_path.exists():
        print("Creating .env file...")
        with open(env_path, 'w') as f:
            f.write("OPENAI_API_KEY=your-api-key-here\n")
            f.write("RAW_DATA_DIR=./data/raw\n")
        print("Please edit the .env file and add your OpenAI API key.")
        
        # Open the .env file with the default editor
        if platform.system() == 'Windows':
            os.system('notepad .env')
        elif platform.system() == 'Darwin':  # macOS
            os.system('open -t .env')
        else:  # Linux
            if os.environ.get('EDITOR'):
                os.system(f"{os.environ.get('EDITOR')} .env")
            else:
                os.system('xdg-open .env')
        
        return False
    return True

def check_data_directories():
    """Ensure data directories exist"""
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    print("Data directories created.")

def start_streamlit():
    """Start the Streamlit application"""
    print("Starting Streamlit application...")
    url = "http://localhost:8501"
    webbrowser.open(url)
    subprocess.run(["streamlit", "run", "app.py"])

def main():
    """Main entry point for the application"""
    print("=" * 50)
    print("SynthiQuery - AI PDF Chat")
    print("=" * 50)
    
    # Check environment setup
    env_ready = check_env_file()
    if not env_ready:
        print("\nPlease restart the script after updating the .env file.")
        sys.exit(0)
    
    # Create data directories
    check_data_directories()
    
    # Start the application
    start_streamlit()

if __name__ == "__main__":
    main() 
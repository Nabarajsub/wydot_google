# entrypoint.py
import os, subprocess, pathlib
from dotenv import load_dotenv

p = os.getenv("DOTENV_PATH", "/etc/secrets/.env")
if pathlib.Path(p).exists():
    load_dotenv(p, override=False)

app = os.getenv("APP_FILE", "flash_cloud_2.5rpo.py")
port = os.getenv("PORT", "8080")

subprocess.run([
    "python", "-m", "streamlit", "run", app,
    "--server.port", port,
    "--server.address", "0.0.0.0"
])

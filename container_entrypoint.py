# container_entrypoint.py
import os, pathlib, shutil, subprocess
from dotenv import load_dotenv

# 1) Load .env mounted from Secret Manager
dotenv_path = os.getenv("DOTENV_PATH", "/etc/secrets/.env")
if pathlib.Path(dotenv_path).exists():
    load_dotenv(dotenv_path, override=False)

# 2) Copy secrets.toml to ~/.streamlit/secrets.toml so st.secrets works
src = os.getenv("SECRETS_TOML_PATH", "/etc/secrets/streamlit/secrets.toml")
if pathlib.Path(src).exists():
    dst_dir = pathlib.Path.home() / ".streamlit"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "secrets.toml"
    if dst.exists():
        dst.unlink()
    shutil.copy(src, dst)

# 3) Run Streamlit app
app = os.getenv("APP_FILE", "flash_cloud_2.5rpo_login.py")
port = os.getenv("PORT", "8080")
subprocess.run([
    "python", "-m", "streamlit", "run", app,
    "--server.port", port, "--server.address", "0.0.0.0"
])

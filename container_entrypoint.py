#!/usr/bin/env python3
import os, pathlib, shutil, subprocess
from dotenv import load_dotenv

# Try stdlib tomllib first; fall back to toml if installed
try:
    import tomllib

    def load_toml(path: pathlib.Path):
        with open(path, "rb") as f:
            return tomllib.load(f)

except ImportError:
    import toml  # requires `toml` package

    def load_toml(path: pathlib.Path):
        return toml.load(path)


# === 0) Debug info: log environment and working dir ===
print("👀 Starting container entrypoint")
print("HOME:", pathlib.Path.home())
print("WORKDIR:", os.getcwd())
print("DOTENV_PATH:", os.getenv("DOTENV_PATH"))
print("SECRETS_TOML_PATH:", os.getenv("SECRETS_TOML_PATH"))

# DEBUG: Print python environment
import sys
print("🐍 sys.path:", sys.path)
try:
    print("📦 Installed packages (pip list):")
    subprocess.run(["pip", "list"], check=False)
except Exception as e:
    print("Could not run pip list:", e)


# === 1) Load .env mounted from Secret Manager ===
dotenv_path = pathlib.Path(os.getenv("DOTENV_PATH", "/etc/secrets/.env"))
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=False)
    print(f"✅ Loaded .env from {dotenv_path}")
else:
    print(f"⚠️ No .env found at {dotenv_path}")

# === 2) Copy secrets.toml to ~/.streamlit/secrets.toml ===
src = pathlib.Path(os.getenv("SECRETS_TOML_PATH", "/etc/secrets/streamlit/secrets.toml"))
dst_dir = pathlib.Path.home() / ".streamlit"
dst_dir.mkdir(parents=True, exist_ok=True)
dst = dst_dir / "secrets.toml"

if src.exists():
    if dst.exists():
        dst.unlink()
    shutil.copy(src, dst)
    print(f"✅ Copied {src} → {dst}")

    # (optional) verify TOML validity immediately
    try:
        data = load_toml(dst)
        print("✅ Parsed secrets.toml sections:", list(data.keys()))
    except Exception as e:
        print("❌ TOML parse error in secrets.toml:", e)
else:
    print(f"❌ Secret file missing at {src}")
    print("⚠️ st.secrets will fail until Secret Manager mount works")

# === 3) Run App (Chainlit or Streamlit) ===
app = os.getenv("APP_FILE", "chatapp.py")
port = os.getenv("PORT", "8080")

# Simple heuristic: Check if file imports chainlit
is_chainlit = False
try:
    print(f"🧐 Checking {app} for 'chainlit' import...")
    if not os.path.exists(app):
        print(f"❌ File {app} does not exist at {os.getcwd()}")
    else:
        with open(app, "r", encoding="utf-8") as f:
            content = f.read()
            if "chainlit" in content:
                is_chainlit = True
                print("✅ Found 'chainlit' in file content.")
            else:
                print("❌ 'chainlit' NOT found in file content.")
except Exception as e:
    print(f"❌ Error reading {app}: {e}")

# Fallback/Override: If it IS chatapp.py, it IS chainlit
if app == "chatapp.py":
    print("⚠️  Force-enabling Chainlit for known app 'chatapp.py'")
    is_chainlit = True

# PRE-FLIGHT CHECK: Verify chainlit is importable
print("🔍 Pre-flight check: Verifying chainlit module...")
try:
    import chainlit
    print(f"✅ chainlit version {chainlit.__version__} found at {chainlit.__file__}")
except ImportError as e:
    print(f"❌ CRITICAL: Cannot import chainlit! Error: {e}")
    print("📦 Listing installed packages:")
    subprocess.run(["pip", "list"], check=False)
    sys.exit(1)

if is_chainlit:
    print(f"🚀 Launching Chainlit: {app} on port {port}")
    subprocess.run([
        "chainlit", "run", app,
        "--port", port, "--host", "0.0.0.0", "--headless"
    ])
else:
    print(f"🚀 Launching Streamlit: {app} on port {port}")
    subprocess.run([
        "python", "-m", "streamlit", "run", app,
        "--server.port", port, "--server.address", "0.0.0.0"
    ])
#this is changed checkpoint

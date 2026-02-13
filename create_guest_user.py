import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.chat_history_store import get_chat_history_store

db = get_chat_history_store()
email = "guest@app.local"
password = "guest_password_secure_123" # Hardcoded for the auto-login button

# Check if exists
user = db.get_user_by_email(email) if hasattr(db, 'get_user_by_email') else None

if not user:
    # Fallback to direct check if method missing
    import sqlite3
    conn = sqlite3.connect("chat_history.sqlite3")
    cur = conn.execute("SELECT id FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    
    if row:
        print("Guest user already exists.")
    else:
        print("Creating guest user...")
        uid, err = db.create_user(email, password, display_name="Guest User")
        if uid:
            db.set_verified(uid)
            print(f"Guest user created (ID: {uid}) and verified.")
        else:
            print(f"Error creating guest user: {err}")
else:
    print("Guest user already exists.")

import sqlite3
import os

# Default path as per code: os.path.join(os.path.dirname(__file__), "..", "chat_history.sqlite3")
# But checking relative to CWD /Users/uw-user/Desktop/wydot_cloud
db_path = "chat_history.sqlite3"

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    # Try the one in utils/.. if different?
    # actually chatapp.py sets CHAINLIT_DB_FILE to /tmp/chainlit.db in Cloud Run setup blocks but that's for chainlit internal db?
    # No, CHAT_DB handles the User table.
    
    # Let's check where chat_history_store.py thinks it is.
    # It says: os.getenv("CHAT_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "chat_history.sqlite3"))
    pass

print(f"Checking database at: {os.path.abspath(db_path)}")

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    email = "subedinabaraj46@gmail.com"
    cur.execute("SELECT id, email, verified, created_at FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    
    if row:
        print(f"User Found: ID={row[0]}, Email={row[1]}, Verified={row[2]}")
        
        # If not verified, verify them manually for convenience
        if row[2] == 0:
            print("User is NOT verified. Verifying manually...")
            cur.execute("UPDATE users SET verified=1 WHERE id=?", (row[0],))
            conn.commit()
            print("User manually verified.")
    else:
        print("User NOT found in database.")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")

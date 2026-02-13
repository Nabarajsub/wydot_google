import sqlite3
import json
import os

DB_PATH = "chat_history.sqlite3"

def inspect_latest():
    if not os.path.exists(DB_PATH):
        print(f"❌ DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print(f"🔎 Scanning 5 most recent assistant messages with sources...")
    
    cur.execute("SELECT id, session_id, content, sources, ts FROM messages WHERE role='assistant' AND sources IS NOT NULL ORDER BY id DESC LIMIT 5")
    rows = cur.fetchall()
    
    if not rows:
        print("❌ No messages with sources found.")
        return
        
    for row in rows:
        print(f"\n==================================================")
        print(f"Msg ID: {row['id']} | Session: {row['session_id']}")
        print(f"Timestamp: {row['ts']}")
        print(f"Content Preview: {row['content'][:50]}...")
        
        sources_raw = row['sources']
        if not sources_raw:
            print("  ⚠️ Sources field is NULL or Empty string (but query filtered for NOT NULL?)")
            continue
            
        try:
            sources = json.loads(sources_raw)
            print(f"  ✅ Parsed {len(sources)} sources.")
            
            for i, src in enumerate(sources):
                index = src.get('index', '?')
                title = src.get('title', 'No Title')
                preview = src.get('preview')
                
                print(f"    -- Source {index}: {title}")
                if preview:
                    print(f"       ✅ Preview present (len={len(preview)})")
                    # print(f"       Preview start: {preview[:30]}...")
                else:
                    print(f"       ❌ PREVIEW MISSING or EMPTY! Keys: {list(src.keys())}")
                    
        except json.JSONDecodeError:
            print(f"  ❌ Failed to decode JSON sources: {sources_raw[:100]}...")

if __name__ == "__main__":
    inspect_latest()

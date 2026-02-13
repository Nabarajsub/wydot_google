import sys
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.chat_history_store import get_chat_history_store

app = FastAPI()

# Allow CORS so frontend (port 8000) can talk to this (port 8001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_DB = get_chat_history_store()

class RegisterRequest(BaseModel):
    email: str
    password: str

class VerifyRequest(BaseModel):
    email: str
    code: str

@app.post("/auth/register")
async def register(req: RegisterRequest):
    uid, error = CHAT_DB.create_user(req.email, req.password, display_name=req.email.split("@")[0])
    if not uid:
        return JSONResponse(status_code=400, content={"error": error or "Registration failed"})
    print(f"User created: {req.email}")
    
    # For Dev Experience: Fetch the code and return it
    # This voids security in prod, but is helpful for this local setup.
    # checking verification_codes.txt is annoying.
    code = None
    try:
        # We need to peek into the DB since create_user handles the code generation internally/privately
        # But wait, create_user in chat_history_store calls set_verification_code.
        # Let's just query it back.
        import sqlite3
        # Direct DB access because CHAT_DB abstraction doesn't expose "get_code"
        # Assuming we are in wydot_cloud and db is chat_history.sqlite3
        conn = sqlite3.connect("chat_history.sqlite3")
        cur = conn.execute("SELECT verification_code FROM users WHERE id=?", (uid,))
        row = cur.fetchone()
        if row:
            code = row[0]
        conn.close()
    except Exception as e:
        print(f"Error fetching code for dev: {e}")

    return JSONResponse(status_code=200, content={"message": "User created.", "dev_code": code})

@app.post("/auth/verify")
async def verify(req: VerifyRequest):
    uid = CHAT_DB.check_verification_code(req.email, req.code)
    if not uid:
        return JSONResponse(status_code=400, content={"error": "Invalid code"})
    CHAT_DB.set_verified(uid)
    print(f"User verified: {req.email}")
    return JSONResponse(status_code=200, content={"message": "Verified"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

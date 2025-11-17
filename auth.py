import sqlite3
import hashlib

def create_users_table():
    conn = sqlite3.connect("data/conversations.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect("data/conversations.db")
    c = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect("data/conversations.db")
    c = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed))
    result = c.fetchone()
    conn.close()
    return result is not None

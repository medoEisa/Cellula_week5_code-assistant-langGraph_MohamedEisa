import sqlite3
from datetime import datetime

def create_conversations_table():
    conn = sqlite3.connect("data/conversations.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        username TEXT,
        message TEXT,
        response TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

def save_message(username, message, response):
    conn = sqlite3.connect("data/conversations.db")
    c = conn.cursor()
    c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
              (username, message, response, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_conversation(username):
    conn = sqlite3.connect("data/conversations.db")
    c = conn.cursor()
    c.execute("SELECT message, response FROM conversations WHERE username=? ORDER BY timestamp", (username,))
    rows = c.fetchall()
    conn.close()
    return rows

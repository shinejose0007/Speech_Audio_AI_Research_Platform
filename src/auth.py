from pathlib import Path
import sqlite3, hashlib, os, datetime

def connect(db_path: Path):
    db_path=Path(db_path); db_path.parent.mkdir(parents=True, exist_ok=True)
    con=sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS login_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        event TEXT,
        created_at TEXT NOT NULL)""")
    con.commit(); return con

def _hash(pw, salt): return hashlib.sha256((salt+pw).encode()).hexdigest()

def register_user(db_path, username, email, password):
    username=(username or '').strip().lower(); email=(email or '').strip().lower()
    if len(username)<3: return False, 'Username must have at least 3 characters.'
    if len(password)<6: return False, 'Password must have at least 6 characters.'
    con=connect(db_path); count=con.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    role='admin' if count==0 else 'user'
    salt=os.urandom(16).hex(); h=_hash(password,salt)
    try:
        con.execute('INSERT INTO users(username,email,password_hash,salt,role,created_at) VALUES(?,?,?,?,?,?)',
                    (username,email,h,salt,role,datetime.datetime.now().isoformat(timespec='seconds')))
        con.commit(); return True, f'Account created. Role: {role}. Please log in.'
    except sqlite3.IntegrityError: return False, 'Username already exists.'

def verify_user(db_path, username, password):
    con=connect(db_path); username=(username or '').strip().lower()
    cur=con.execute('SELECT username,email,password_hash,salt,role FROM users WHERE username=?', (username,))
    r=cur.fetchone(); now=datetime.datetime.now().isoformat(timespec='seconds')
    if r and _hash(password, r[3])==r[2]:
        con.execute('INSERT INTO login_events(username,event,created_at) VALUES(?,?,?)', (username,'login_success',now)); con.commit()
        return True, {'username':r[0], 'email':r[1], 'role':r[4]}
    con.execute('INSERT INTO login_events(username,event,created_at) VALUES(?,?,?)', (username,'login_failed',now)); con.commit()
    return False, None

def list_users(db_path):
    con=connect(db_path); rows=con.execute('SELECT id,username,email,role,created_at FROM users ORDER BY id').fetchall()
    return [{'id':r[0], 'username':r[1], 'email':r[2], 'role':r[3], 'created_at':r[4]} for r in rows]

def list_login_events(db_path, limit=100):
    con=connect(db_path); rows=con.execute('SELECT username,event,created_at FROM login_events ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    return [{'username':r[0], 'event':r[1], 'created_at':r[2]} for r in rows]

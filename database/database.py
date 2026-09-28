"""
database/database.py

Reusable SQLite database layer for the Study Notes & Flashcard Generator.

Uses only Python's built-in sqlite3 library. No external database
systems are used or required.

Database location: database/study.db
The database folder and file are created automatically if they do not
already exist.
"""

import os
import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Directory this file lives in (database/)
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_FILENAME = "study.db"
DB_PATH = os.path.join(_DB_DIR, _DB_FILENAME)


def _ensure_db_folder_exists():
    """Make sure the database/ folder exists before connecting."""
    os.makedirs(_DB_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection():
    """
    Return a SQLite connection to database/study.db.

    - Creates the database folder/file automatically if missing.
    - Enables foreign key enforcement.
    - Uses sqlite3.Row so results behave like dictionaries.
    """
    _ensure_db_folder_exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _ensure_username_column(conn):
    """
    Older study.db files have no students.username column. Add it once, so a
    login (username) can be linked to ONE permanent student row. Existing rows
    are kept untouched (their username stays empty).
    """
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(students)")]
    if "username" not in columns:
        conn.execute("ALTER TABLE students ADD COLUMN username TEXT")
        conn.commit()


def _ensure_users_table(conn):
    """
    Create the login accounts table if it does not exist yet. Safe to call any
    number of times; it never touches existing tables or rows.

    - email is UNIQUE (and case-insensitive), so the same address can never be
      registered twice, even if two registrations arrive at the same moment.
    - Only a salted password HASH is stored, never the password itself.
    - hash_algorithm + iterations are stored so the hashing settings can be
      strengthened later without locking existing users out.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            hash_algorithm TEXT NOT NULL DEFAULT 'pbkdf2_sha256',
            iterations INTEGER NOT NULL DEFAULT 600000,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


def initialize_database():
    """
    Create all required tables if they do not already exist.

    Safe to call multiple times.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                subject TEXT,
                topic TEXT,
                started_at TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                topic TEXT,
                question_type TEXT,
                is_correct INTEGER,
                difficulty TEXT,
                answered_at TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_material (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                subject TEXT,
                topic TEXT,
                content TEXT,
                created_at TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                topic TEXT,
                question TEXT,
                answer TEXT,
                created_at TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        """)

        _ensure_username_column(conn)
        _ensure_users_table(conn)

        conn.commit()
    except sqlite3.Error as e:
        # Do not expose raw database errors to end users.
        print(f"[database] Failed to initialize database: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Users (login accounts)
# ---------------------------------------------------------------------------

class EmailAlreadyExistsError(Exception):
    """Raised by create_user() when the email is already registered."""


def get_user_by_email(email):
    """
    Return the user with this email as a dict, or None if there is none.
    The email is matched case-insensitively (Anisha@gmail.com == anisha@gmail.com).

    Unlike the other helpers in this file, database errors are NOT swallowed here:
    they are raised, so the login page can tell "wrong password" apart from
    "the database is having a problem".
    """
    key = (email or "").strip().lower()
    if not key:
        return None

    conn = get_connection()
    try:
        _ensure_users_table(conn)
        row = conn.execute(
            """
            SELECT id, full_name, email, password_hash, salt,
                   hash_algorithm, iterations, created_at
            FROM users
            WHERE email = ?
            """,
            (key,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(full_name, email, password_hash, salt, iterations, hash_algorithm="pbkdf2_sha256"):
    """
    Save a new login account and return its new user ID.

    The email is normalised to lower case before it is stored. Raises
    EmailAlreadyExistsError if the email is already registered (enforced by the
    UNIQUE constraint, not just by a lookup). Other database errors are raised
    as sqlite3.Error.
    """
    key = (email or "").strip().lower()

    conn = get_connection()
    try:
        _ensure_users_table(conn)
        cursor = conn.execute(
            """
            INSERT INTO users
                (full_name, email, password_hash, salt, hash_algorithm, iterations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                full_name,
                key,
                password_hash,
                salt,
                hash_algorithm,
                int(iterations),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        if "UNIQUE" in str(e).upper():
            raise EmailAlreadyExistsError(key) from e
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def create_student(name):
    """Create a student and return the new student ID, or None on failure."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat())
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[database] Failed to create student: {e}")
        return None
    finally:
        conn.close()


def get_student(student_id):
    """Return the student's information as a dict, or None if not found."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM students WHERE id = ?",
            (student_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"[database] Failed to get student: {e}")
        return None
    finally:
        conn.close()


def get_or_create_student(username, name):
    """
    Return the student id that belongs to this login username.

    - Same username again  -> the SAME id (so progress is kept between visits).
    - First time           -> an older profile with no username but the same
                              name is adopted; otherwise a new student is created.
    Returns None on failure.
    """
    key = (username or "").strip().lower()
    if not key:
        return create_student(name)

    conn = get_connection()
    try:
        _ensure_username_column(conn)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM students WHERE username = ?", (key,))
        row = cursor.fetchone()
        if row:
            return row["id"]

        cursor.execute(
            "SELECT id FROM students WHERE username IS NULL AND lower(name) = lower(?) ORDER BY id LIMIT 1",
            (name,),
        )
        row = cursor.fetchone()
        if row:
            cursor.execute("UPDATE students SET username = ? WHERE id = ?", (key, row["id"]))
            conn.commit()
            return row["id"]

        cursor.execute(
            "INSERT INTO students (name, created_at, username) VALUES (?, ?, ?)",
            (name, datetime.now().isoformat(), key),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[database] Failed to get or create student: {e}")
        return None
    finally:
        conn.close()


def update_student_name(student_id, name):
    """Change a student's name. Returns True on success."""
    conn = get_connection()
    try:
        conn.execute("UPDATE students SET name = ? WHERE id = ?", (name, student_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"[database] Failed to update student name: {e}")
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Study sessions
# ---------------------------------------------------------------------------

def save_study_session(student_id, subject, topic):
    """Save a study session. Returns the new session ID, or None on failure."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO study_sessions (student_id, subject, topic, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, subject, topic, datetime.now().isoformat())
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[database] Failed to save study session: {e}")
        return None
    finally:
        conn.close()


def get_study_sessions(student_id):
    """Return a list of the student's study sessions (as dicts)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM study_sessions
            WHERE student_id = ?
            ORDER BY started_at DESC
            """,
            (student_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[database] Failed to get study sessions: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Study material
# ---------------------------------------------------------------------------

def save_study_material(student_id, subject, topic, content):
    """Save generated study material. Returns the new record ID, or None."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO study_material (student_id, subject, topic, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, subject, topic, content, datetime.now().isoformat())
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[database] Failed to save study material: {e}")
        return None
    finally:
        conn.close()


def get_study_material(student_id):
    """Return a list of saved study material (as dicts) for the student."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM study_material
            WHERE student_id = ?
            ORDER BY created_at DESC
            """,
            (student_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[database] Failed to get study material: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

def save_flashcard(student_id, topic, question, answer):
    """Save a flashcard. Returns the new flashcard ID, or None on failure."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO flashcards (student_id, topic, question, answer, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, topic, question, answer, datetime.now().isoformat())
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[database] Failed to save flashcard: {e}")
        return None
    finally:
        conn.close()


def get_flashcards(student_id):
    """Return a list of saved flashcards (as dicts) for the student."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM flashcards
            WHERE student_id = ?
            ORDER BY created_at DESC
            """,
            (student_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[database] Failed to get flashcards: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Question results
# ---------------------------------------------------------------------------

def save_question_result(student_id, topic, question_type, is_correct, difficulty):
    """
    Save a question result.

    is_correct is stored as 1 or 0 regardless of the truthy value passed in.
    Returns the new record ID, or None on failure.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO question_results
                (student_id, topic, question_type, is_correct, difficulty, answered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                topic,
                question_type,
                1 if is_correct else 0,
                difficulty,
                datetime.now().isoformat()
            )
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[database] Failed to save question result: {e}")
        return None
    finally:
        conn.close()


def get_question_results(student_id):
    """Return a list of the student's question results (as dicts)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM question_results
            WHERE student_id = ?
            ORDER BY answered_at DESC
            """,
            (student_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"[database] Failed to get question results: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_topic_statistics(student_id):
    """
    Return topic-level statistics for the student as a list of dicts:

    [
        {
            "topic": str,
            "total_questions": int,
            "correct_answers": int,
            "incorrect_answers": int,
            "accuracy": float  # percentage, 0-100, rounded to 2 decimals
        },
        ...
    ]
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                topic,
                COUNT(*) AS total_questions,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct_answers,
                SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) AS incorrect_answers
            FROM question_results
            WHERE student_id = ?
            GROUP BY topic
            ORDER BY topic ASC
            """,
            (student_id,)
        )
        rows = cursor.fetchall()

        stats = []
        for row in rows:
            total = row["total_questions"] or 0
            correct = row["correct_answers"] or 0
            incorrect = row["incorrect_answers"] or 0
            accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0

            stats.append({
                "topic": row["topic"],
                "total_questions": total,
                "correct_answers": correct,
                "incorrect_answers": incorrect,
                "accuracy": accuracy
            })

        return stats
    except sqlite3.Error as e:
        print(f"[database] Failed to get topic statistics: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Manual run: initialize the database
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")
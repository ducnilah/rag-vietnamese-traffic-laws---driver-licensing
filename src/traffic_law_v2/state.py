from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from typing import Dict, List, Optional, Protocol
from uuid import uuid4

from traffic_law_v2.config import Settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    id: str
    role: str
    content: str
    created_at: str
    citations: Optional[dict] = None


@dataclass
class Thread:
    id: str
    user_id: str
    title: str
    created_at: str
    messages: List[Message] = field(default_factory=list)


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    created_at: str


class StateStore(Protocol):
    def register_user(self, email: str, password: str) -> User: ...
    def login(self, email: str, password: str) -> tuple[User, str]: ...
    def user_from_token(self, token: str) -> Optional[User]: ...
    def create_thread(self, user_id: str, title: str = "New chat") -> Thread: ...
    def ensure_thread(self, thread_id: str, user_id: str, title: str = "Recovered chat") -> Thread: ...
    def get_thread(self, thread_id: str) -> Optional[Thread]: ...
    def list_threads(self, user_id: str) -> List[Thread]: ...
    def add_message(self, thread_id: str, role: str, content: str, citations: Optional[dict] = None) -> Message: ...
    def recent_history(self, thread_id: str, limit: int = 8) -> str: ...
    def remember(self, user_id: str, key: str, value: str) -> None: ...
    def user_memory(self, user_id: str) -> Dict[str, str]: ...
    def add_feedback(self, user_id: str, message_id: str, rating: int, note: str = "") -> dict: ...


def create_state(settings: Settings) -> StateStore:
    if settings.database_url.lower() in {"", "memory", "inmemory", "in-memory"}:
        return InMemoryState()
    return PostgresState(settings.database_url)


class InMemoryState:
    def __init__(self) -> None:
        self.users: Dict[str, User] = {}
        self.users_by_email: Dict[str, str] = {}
        self.tokens: Dict[str, str] = {}
        self.threads: Dict[str, Thread] = {}
        self.memory: Dict[str, Dict[str, str]] = {}
        self.feedback: List[dict] = []

    def register_user(self, email: str, password: str) -> User:
        normalized = _normalize_email(email)
        _validate_password(password)
        if normalized in self.users_by_email:
            raise ValueError("Email đã tồn tại")
        user = User(id=uuid4().hex, email=normalized, password_hash=_hash_password(password), created_at=now_iso())
        self.users[user.id] = user
        self.users_by_email[normalized] = user.id
        return user

    def login(self, email: str, password: str) -> tuple[User, str]:
        user_id = self.users_by_email.get(email.strip().lower())
        if not user_id:
            raise ValueError("Email hoặc mật khẩu không đúng")
        user = self.users[user_id]
        if not _verify_password(password, user.password_hash):
            raise ValueError("Email hoặc mật khẩu không đúng")
        token = uuid4().hex
        self.tokens[token] = user.id
        return user, token

    def user_from_token(self, token: str) -> Optional[User]:
        user_id = self.tokens.get(token)
        return self.users.get(user_id) if user_id else None

    def create_thread(self, user_id: str, title: str = "New chat") -> Thread:
        thread = Thread(id=uuid4().hex, user_id=user_id, title=title, created_at=now_iso())
        self.threads[thread.id] = thread
        return thread

    def ensure_thread(self, thread_id: str, user_id: str, title: str = "Recovered chat") -> Thread:
        thread = self.threads.get(thread_id)
        if thread:
            return thread
        thread = Thread(id=thread_id, user_id=user_id, title=title, created_at=now_iso())
        self.threads[thread.id] = thread
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        return self.threads.get(thread_id)

    def list_threads(self, user_id: str) -> List[Thread]:
        return [thread for thread in self.threads.values() if thread.user_id == user_id]

    def add_message(self, thread_id: str, role: str, content: str, citations: Optional[dict] = None) -> Message:
        thread = self.threads[thread_id]
        msg = Message(id=uuid4().hex, role=role, content=content, citations=citations, created_at=now_iso())
        thread.messages.append(msg)
        return msg

    def recent_history(self, thread_id: str, limit: int = 8) -> str:
        thread = self.threads.get(thread_id)
        if not thread:
            return ""
        messages = thread.messages[-limit:]
        return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

    def remember(self, user_id: str, key: str, value: str) -> None:
        self.memory.setdefault(user_id, {})[key] = value

    def user_memory(self, user_id: str) -> Dict[str, str]:
        return dict(self.memory.get(user_id, {}))

    def add_feedback(self, user_id: str, message_id: str, rating: int, note: str = "") -> dict:
        row = {"user_id": user_id, "message_id": message_id, "rating": rating, "note": note, "created_at": now_iso()}
        self.feedback.append(row)
        return row


class PostgresState:
    def __init__(self, database_url: str) -> None:
        from sqlalchemy import create_engine

        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        from sqlalchemy import text

        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations JSONB,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_user_created ON chat_threads(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at)",
        ]
        with self.engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

    def register_user(self, email: str, password: str) -> User:
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        normalized = _normalize_email(email)
        _validate_password(password)
        user = User(id=uuid4().hex, email=normalized, password_hash=_hash_password(password), created_at=now_iso())
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO users (id, email, password_hash, created_at) "
                        "VALUES (:id, :email, :password_hash, :created_at)"
                    ),
                    user.__dict__,
                )
        except IntegrityError as exc:
            raise ValueError("Email đã tồn tại") from exc
        return user

    def login(self, email: str, password: str) -> tuple[User, str]:
        from sqlalchemy import text

        normalized = email.strip().lower()
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT * FROM users WHERE email = :email"), {"email": normalized}).mappings().first()
            if not row or not _verify_password(password, row["password_hash"]):
                raise ValueError("Email hoặc mật khẩu không đúng")
            user = _user_from_row(row)
            token = uuid4().hex
            conn.execute(
                text("INSERT INTO auth_tokens (token, user_id, created_at) VALUES (:token, :user_id, :created_at)"),
                {"token": token, "user_id": user.id, "created_at": now_iso()},
            )
        return user, token

    def user_from_token(self, token: str) -> Optional[User]:
        from sqlalchemy import text

        if not token:
            return None
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT users.* FROM users "
                    "JOIN auth_tokens ON auth_tokens.user_id = users.id "
                    "WHERE auth_tokens.token = :token"
                ),
                {"token": token},
            ).mappings().first()
        return _user_from_row(row) if row else None

    def create_thread(self, user_id: str, title: str = "New chat") -> Thread:
        from sqlalchemy import text

        thread = Thread(id=uuid4().hex, user_id=user_id, title=title, created_at=now_iso())
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO chat_threads (id, user_id, title, created_at) VALUES (:id, :user_id, :title, :created_at)"),
                {"id": thread.id, "user_id": thread.user_id, "title": thread.title, "created_at": thread.created_at},
            )
        return thread

    def ensure_thread(self, thread_id: str, user_id: str, title: str = "Recovered chat") -> Thread:
        existing = self.get_thread(thread_id)
        if existing:
            return existing
        from sqlalchemy import text

        thread = Thread(id=thread_id, user_id=user_id, title=title, created_at=now_iso())
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO chat_threads (id, user_id, title, created_at) VALUES (:id, :user_id, :title, :created_at)"),
                {"id": thread.id, "user_id": thread.user_id, "title": thread.title, "created_at": thread.created_at},
            )
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT * FROM chat_threads WHERE id = :id"), {"id": thread_id}).mappings().first()
            if not row:
                return None
            messages = self._messages_for_thread(conn, thread_id)
        return _thread_from_row(row, messages)

    def list_threads(self, user_id: str) -> List[Thread]:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT * FROM chat_threads WHERE user_id = :user_id ORDER BY created_at DESC"),
                {"user_id": user_id},
            ).mappings().all()
            counts = dict(
                conn.execute(
                    text(
                        "SELECT thread_id, COUNT(*) AS count FROM chat_messages "
                        "WHERE thread_id IN (SELECT id FROM chat_threads WHERE user_id = :user_id) "
                        "GROUP BY thread_id"
                    ),
                    {"user_id": user_id},
                ).all()
            )
        threads = []
        for row in rows:
            messages = [Message(id="", role="", content="", created_at="") for _ in range(int(counts.get(row["id"], 0)))]
            threads.append(_thread_from_row(row, messages))
        return threads

    def add_message(self, thread_id: str, role: str, content: str, citations: Optional[dict] = None) -> Message:
        from sqlalchemy import text

        msg = Message(id=uuid4().hex, role=role, content=content, citations=citations, created_at=now_iso())
        with self.engine.begin() as conn:
            exists = conn.execute(text("SELECT 1 FROM chat_threads WHERE id = :id"), {"id": thread_id}).first()
            if not exists:
                raise KeyError(thread_id)
            conn.execute(
                text(
                    "INSERT INTO chat_messages (id, thread_id, role, content, citations, created_at) "
                    "VALUES (:id, :thread_id, :role, :content, CAST(:citations AS JSONB), :created_at)"
                ),
                {
                    "id": msg.id,
                    "thread_id": thread_id,
                    "role": msg.role,
                    "content": msg.content,
                    "citations": json.dumps(citations) if citations is not None else None,
                    "created_at": msg.created_at,
                },
            )
        return msg

    def recent_history(self, thread_id: str, limit: int = 8) -> str:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT role, content FROM chat_messages WHERE thread_id = :thread_id "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"thread_id": thread_id, "limit": limit},
            ).mappings().all()
        rows = list(reversed(rows))
        return "\n".join(f"{row['role']}: {row['content']}" for row in rows)

    def remember(self, user_id: str, key: str, value: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO user_memory (user_id, key, value, updated_at) "
                    "VALUES (:user_id, :key, :value, :updated_at) "
                    "ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at"
                ),
                {"user_id": user_id, "key": key, "value": value, "updated_at": now_iso()},
            )

    def user_memory(self, user_id: str) -> Dict[str, str]:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT key, value FROM user_memory WHERE user_id = :user_id"), {"user_id": user_id}).all()
        return {key: value for key, value in rows}

    def add_feedback(self, user_id: str, message_id: str, rating: int, note: str = "") -> dict:
        from sqlalchemy import text

        row = {"id": uuid4().hex, "user_id": user_id, "message_id": message_id, "rating": rating, "note": note, "created_at": now_iso()}
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO feedback (id, user_id, message_id, rating, note, created_at) "
                    "VALUES (:id, :user_id, :message_id, :rating, :note, :created_at)"
                ),
                row,
            )
        return {key: value for key, value in row.items() if key != "id"}

    def _messages_for_thread(self, conn, thread_id: str) -> List[Message]:
        from sqlalchemy import text

        rows = conn.execute(
            text("SELECT * FROM chat_messages WHERE thread_id = :thread_id ORDER BY created_at ASC"),
            {"thread_id": thread_id},
        ).mappings().all()
        return [_message_from_row(row) for row in rows]


def extract_memory_facts(text: str) -> Dict[str, str]:
    lowered = text.lower()
    facts: Dict[str, str] = {}
    if "tôi tên là" in lowered and not any(mark in lowered for mark in ("?", "gì", "không")):
        name = text.lower().split("tôi tên là", 1)[1].split(",", 1)[0].strip(" .!?")
        if name and name not in {"gì", "ai"}:
            facts["preferred_name"] = name.title()
    for token in lowered.replace(",", " ").split():
        if token.isdigit() and 10 <= int(token) <= 90 and "tuổi" in lowered:
            facts["age"] = token
            break
    return facts


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if not normalized or "@" not in normalized:
        raise ValueError("Email không hợp lệ")
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < 6:
        raise ValueError("Mật khẩu cần tối thiểu 6 ký tự")


def _hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, salt, digest = encoded.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return hmac.compare_digest(candidate, digest)


def _user_from_row(row) -> User:
    return User(id=row["id"], email=row["email"], password_hash=row["password_hash"], created_at=row["created_at"])


def _thread_from_row(row, messages: List[Message] | None = None) -> Thread:
    return Thread(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=row["created_at"],
        messages=messages or [],
    )


def _message_from_row(row) -> Message:
    citations = row["citations"]
    if isinstance(citations, str):
        citations = json.loads(citations)
    return Message(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        citations=citations,
        created_at=row["created_at"],
    )

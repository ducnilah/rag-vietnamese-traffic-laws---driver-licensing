from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import os
from typing import Dict, List, Optional
from uuid import uuid4


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


class InMemoryState:
    def __init__(self) -> None:
        self.users: Dict[str, User] = {}
        self.users_by_email: Dict[str, str] = {}
        self.tokens: Dict[str, str] = {}
        self.threads: Dict[str, Thread] = {}
        self.memory: Dict[str, Dict[str, str]] = {}
        self.feedback: List[dict] = []

    def register_user(self, email: str, password: str) -> User:
        normalized = email.strip().lower()
        if not normalized or "@" not in normalized:
            raise ValueError("Email không hợp lệ")
        if len(password) < 6:
            raise ValueError("Mật khẩu cần tối thiểu 6 ký tự")
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

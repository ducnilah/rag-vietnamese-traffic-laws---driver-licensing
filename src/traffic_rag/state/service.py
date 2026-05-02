from __future__ import annotations

import json
import base64
import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from sqlalchemy import and_, create_engine, or_, select
    from sqlalchemy.orm import Session, sessionmaker

    from .models import Base, Message, Thread, ThreadSummary, User, UserMemory

    SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover
    SQLALCHEMY_AVAILABLE = False


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class ThreadDTO:
    id: str
    user_id: str
    title: str
    archived: bool
    created_at: str
    last_active_at: str


@dataclass(frozen=True)
class MessageDTO:
    id: str
    thread_id: str
    role: str
    content: str
    citations: Optional[Dict[str, object]]
    created_at: str


@dataclass(frozen=True)
class UserDTO:
    id: str
    email: Optional[str]
    is_active: bool
    created_at: str


class ConversationService:
    def __init__(self, db_url: str, auth_secret: Optional[str] = None) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "sqlalchemy is not installed. Install with: python3 -m pip install sqlalchemy"
            )
        self.engine = create_engine(db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, class_=Session, expire_on_commit=False)
        self.auth_secret = (auth_secret or os.getenv("TRAFFIC_RAG_AUTH_SECRET") or "dev-secret-change-me").encode(
            "utf-8"
        )

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ensure_user(self, user_id: str, email: Optional[str] = None) -> None:
        with self.session_factory() as session:
            user = session.get(User, user_id)
            if user is None:
                session.add(User(id=user_id, email=email))
            elif email and not user.email:
                user.email = email
                user.updated_at = utcnow()
            session.commit()

    def register_user(self, email: str, password: str) -> UserDTO:
        normalized = email.strip().lower()
        if not normalized or "@" not in normalized:
            raise ValueError("invalid email")
        if len(password) < 6:
            raise ValueError("password too short")
        with self.session_factory() as session:
            exists = session.execute(select(User).where(User.email == normalized)).scalar_one_or_none()
            if exists is not None:
                raise ValueError("email already exists")
            now = utcnow()
            user = User(
                id=_new_id(),
                email=normalized,
                password_hash=self._hash_password(password),
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            session.add(user)
            session.commit()
            return self._user_to_dto(user)

    def authenticate(self, email: str, password: str) -> Optional[UserDTO]:
        normalized = email.strip().lower()
        with self.session_factory() as session:
            user = session.execute(select(User).where(User.email == normalized)).scalar_one_or_none()
            if user is None or not user.is_active:
                return None
            if not user.password_hash:
                return None
            if not self._verify_password(password, user.password_hash):
                return None
            return self._user_to_dto(user)

    def get_user(self, user_id: str) -> Optional[UserDTO]:
        with self.session_factory() as session:
            row = session.get(User, user_id)
            return self._user_to_dto(row) if row else None

    def issue_access_token(self, user_id: str, expires_in_sec: int = 86400) -> str:
        exp = int(datetime.now(timezone.utc).timestamp()) + int(expires_in_sec)
        payload = {"sub": user_id, "exp": exp}
        return self._sign_payload(payload)

    def verify_access_token(self, token: str) -> Optional[str]:
        payload = self._verify_payload(token)
        if payload is None:
            return None
        sub = payload.get("sub")
        exp = payload.get("exp")
        if not isinstance(sub, str) or not isinstance(exp, int):
            return None
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if exp < now_ts:
            return None
        return sub

    def create_thread(self, user_id: str, title: str = "New chat") -> ThreadDTO:
        self.ensure_user(user_id)
        now = utcnow()
        thread = Thread(
            id=_new_id(),
            user_id=user_id,
            title=title.strip() or "New chat",
            archived=False,
            created_at=now,
            last_active_at=now,
        )
        with self.session_factory() as session:
            session.add(thread)
            session.commit()
        return self._thread_to_dto(thread)

    def list_threads(self, user_id: str, limit: int = 20) -> List[ThreadDTO]:
        with self.session_factory() as session:
            rows = session.execute(
                select(Thread)
                .where(Thread.user_id == user_id, Thread.archived.is_(False))
                .order_by(Thread.last_active_at.desc())
                .limit(limit)
            ).scalars()
            return [self._thread_to_dto(row) for row in rows]

    def list_threads_any_status(self, user_id: str, limit: int = 20) -> List[ThreadDTO]:
        with self.session_factory() as session:
            rows = session.execute(
                select(Thread).where(Thread.user_id == user_id).order_by(Thread.last_active_at.desc()).limit(limit)
            ).scalars()
            return [self._thread_to_dto(row) for row in rows]

    def get_thread(self, thread_id: str) -> Optional[ThreadDTO]:
        with self.session_factory() as session:
            row = session.get(Thread, thread_id)
            return self._thread_to_dto(row) if row else None

    def update_thread(
        self,
        thread_id: str,
        *,
        title: Optional[str] = None,
        archived: Optional[bool] = None,
    ) -> Optional[ThreadDTO]:
        with self.session_factory() as session:
            row = session.get(Thread, thread_id)
            if row is None:
                return None
            touched = False
            if title is not None:
                normalized = title.strip() or "New chat"
                if row.title != normalized:
                    row.title = normalized
                    touched = True
            if archived is not None:
                new_archived = bool(archived)
                if row.archived != new_archived:
                    row.archived = new_archived
                    touched = True
            if touched:
                row.last_active_at = utcnow()
                session.commit()
            return self._thread_to_dto(row)

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        citations: Optional[Dict[str, object]] = None,
    ) -> MessageDTO:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of: user, assistant, system")

        with self.session_factory() as session:
            thread = session.get(Thread, thread_id)
            if thread is None:
                raise KeyError(f"thread not found: {thread_id}")
            msg = Message(
                id=_new_id(),
                thread_id=thread_id,
                role=role,
                content=content.strip(),
                citations_json=(json.dumps(citations, ensure_ascii=False) if citations else None),
                created_at=utcnow(),
            )
            thread.last_active_at = utcnow()
            session.add(msg)
            session.commit()
            return self._message_to_dto(msg)

    def list_messages(self, thread_id: str, limit: int = 50, before_message_id: Optional[str] = None) -> List[MessageDTO]:
        with self.session_factory() as session:
            stmt = select(Message).where(Message.thread_id == thread_id)
            if before_message_id:
                anchor = session.get(Message, before_message_id)
                if anchor is None or anchor.thread_id != thread_id:
                    return []
                stmt = stmt.where(
                    or_(
                        Message.created_at < anchor.created_at,
                        and_(Message.created_at == anchor.created_at, Message.id < anchor.id),
                    )
                )
            rows = session.execute(
                stmt.order_by(Message.created_at.desc(), Message.id.desc()).limit(limit)
            ).scalars()
            rows_list = list(rows)
            rows_list.reverse()
            return [self._message_to_dto(row) for row in rows_list]

    def refresh_summary(self, thread_id: str, max_messages: int = 10) -> str:
        messages = self.list_messages(thread_id, limit=max_messages)
        parts: List[str] = []
        for msg in messages[-max_messages:]:
            prefix = "U" if msg.role == "user" else "A"
            parts.append(f"{prefix}: {msg.content[:180]}")
        summary_text = " | ".join(parts)[:2000]

        with self.session_factory() as session:
            row = session.get(ThreadSummary, thread_id)
            if row is None:
                row = ThreadSummary(thread_id=thread_id, summary=summary_text, updated_at=utcnow())
                session.add(row)
            else:
                row.summary = summary_text
                row.updated_at = utcnow()
            session.commit()
        return summary_text

    def get_summary(self, thread_id: str) -> Optional[str]:
        with self.session_factory() as session:
            row = session.get(ThreadSummary, thread_id)
            return row.summary if row else None

    def upsert_memory(self, user_id: str, key: str, value: str, confidence: float = 0.7) -> None:
        self.ensure_user(user_id)
        confidence = max(0.0, min(1.0, float(confidence)))
        with self.session_factory() as session:
            row = session.execute(
                select(UserMemory).where(UserMemory.user_id == user_id, UserMemory.memory_key == key)
            ).scalar_one_or_none()
            if row is None:
                row = UserMemory(
                    id=_new_id(),
                    user_id=user_id,
                    memory_key=key,
                    memory_value=value,
                    confidence=confidence,
                    updated_at=utcnow(),
                )
                session.add(row)
            else:
                row.memory_value = value
                row.confidence = confidence
                row.updated_at = utcnow()
            session.commit()

    def list_memory(self, user_id: str) -> List[Dict[str, object]]:
        with self.session_factory() as session:
            rows = session.execute(
                select(UserMemory)
                .where(UserMemory.user_id == user_id)
                .order_by(UserMemory.updated_at.desc())
            ).scalars()
            return [
                {
                    "key": row.memory_key,
                    "value": row.memory_value,
                    "confidence": row.confidence,
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    @staticmethod
    def _thread_to_dto(row: Thread) -> ThreadDTO:
        return ThreadDTO(
            id=row.id,
            user_id=row.user_id,
            title=row.title,
            archived=bool(row.archived),
            created_at=row.created_at.isoformat(),
            last_active_at=row.last_active_at.isoformat(),
        )

    @staticmethod
    def _message_to_dto(row: Message) -> MessageDTO:
        citations = json.loads(row.citations_json) if row.citations_json else None
        return MessageDTO(
            id=row.id,
            thread_id=row.thread_id,
            role=row.role,
            content=row.content,
            citations=citations,
            created_at=row.created_at.isoformat(),
        )

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return "pbkdf2_sha256$120000$" + base64.urlsafe_b64encode(salt).decode("ascii") + "$" + base64.urlsafe_b64encode(
            digest
        ).decode("ascii")

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            algo, rounds_s, salt_b64, digest_b64 = encoded.split("$", 3)
            if algo != "pbkdf2_sha256":
                return False
            rounds = int(rounds_s)
            salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
            expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    def _sign_payload(self, payload: Dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        body = base64.urlsafe_b64encode(raw).decode("ascii")
        sig = hmac.new(self.auth_secret, body.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{body}.{sig}"

    def _verify_payload(self, token: str) -> Optional[Dict[str, object]]:
        try:
            body, sig = token.rsplit(".", 1)
        except ValueError:
            return None
        expected = hmac.new(self.auth_secret, body.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            data = base64.urlsafe_b64decode(body.encode("ascii"))
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _user_to_dto(row: User) -> UserDTO:
        return UserDTO(
            id=row.id,
            email=row.email,
            is_active=bool(row.is_active),
            created_at=row.created_at.isoformat(),
        )

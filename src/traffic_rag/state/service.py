from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    from sqlalchemy import create_engine, select
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


class ConversationService:
    def __init__(self, db_url: str) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "sqlalchemy is not installed. Install with: python3 -m pip install sqlalchemy"
            )
        self.engine = create_engine(db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, class_=Session, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ensure_user(self, user_id: str, email: Optional[str] = None) -> None:
        with self.session_factory() as session:
            user = session.get(User, user_id)
            if user is None:
                session.add(User(id=user_id, email=email))
            elif email and not user.email:
                user.email = email
            session.commit()

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

    def get_thread(self, thread_id: str) -> Optional[ThreadDTO]:
        with self.session_factory() as session:
            row = session.get(Thread, thread_id)
            return self._thread_to_dto(row) if row else None

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

    def list_messages(self, thread_id: str, limit: int = 50) -> List[MessageDTO]:
        with self.session_factory() as session:
            rows = session.execute(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
            ).scalars()
            return [self._message_to_dto(row) for row in rows]

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

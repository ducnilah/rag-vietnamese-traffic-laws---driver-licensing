import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from traffic_rag.state.service import MessageDTO


@dataclass(frozen=True)
class MemoryFact:
    key: str
    value: str
    confidence: float = 0.95


_NAME_RE = re.compile(r"(?:tôi tên là|mình tên là|tên tôi là)\s+([a-zA-ZÀ-ỹĐđ\s]{2,40})", re.IGNORECASE)
_AGE_RE = re.compile(r"(?:tôi|mình)?\s*(\d{1,2})\s*tuổi", re.IGNORECASE)


def extract_user_facts(text: str) -> List[MemoryFact]:
    facts: List[MemoryFact] = []
    match_name = _NAME_RE.search(text)
    if match_name:
        name = " ".join(match_name.group(1).strip().split())
        name = re.sub(r"[?.!,;:]+$", "", name).strip()
        if name:
            facts.append(MemoryFact(key="preferred_name", value=name))
    match_age = _AGE_RE.search(text)
    if match_age:
        facts.append(MemoryFact(key="age", value=match_age.group(1)))
    return facts


def build_conversation_context(
    memory_items: Iterable[Dict[str, object]],
    recent_messages: Iterable[MessageDTO],
    max_memory_items: int = 8,
    max_messages: int = 8,
    max_message_chars: int = 220,
) -> str:
    mem_lines: List[str] = []
    for idx, item in enumerate(memory_items):
        if idx >= max_memory_items:
            break
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if key and value:
            mem_lines.append(f"- {key}: {value}")

    history_lines: List[str] = []
    recent = list(recent_messages)[-max_messages:]
    for row in recent:
        role = "User" if row.role == "user" else "Assistant"
        history_lines.append(f"{role}: {row.content[:max_message_chars]}")

    sections: List[str] = []
    if mem_lines:
        sections.append("Known user profile:\n" + "\n".join(mem_lines))
    if history_lines:
        sections.append("Recent conversation:\n" + "\n".join(history_lines))
    return "\n\n".join(sections).strip()


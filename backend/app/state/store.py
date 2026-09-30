import json
from pathlib import Path
from typing import Optional

from langchain_core.messages import messages_from_dict, messages_to_dict


STATE_DIR = Path(__file__).parent.parent.parent / "data" / "sessions"


class SessionStore:
    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        return STATE_DIR / f"{safe}.json"

    def load(self, session_id: str) -> Optional[dict]:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {
                "current_recipe": raw.get("current_recipe"),
                "current_step": raw.get("current_step", 0),
                "active_timers": raw.get("active_timers", []),
                "constraints": raw.get("constraints", []),
                "messages": messages_from_dict(raw.get("messages", [])),
            }
        except Exception as e:
            print(f"Failed to load session {session_id}: {e}")
            return None

    def save(
        self,
        session_id: str,
        current_recipe: Optional[dict],
        current_step: int,
        active_timers: list,
        messages: list,
        constraints: list | None = None,
    ) -> None:
        try:
            payload = {
                "current_recipe": current_recipe,
                "current_step": current_step,
                "active_timers": active_timers,
                "constraints": constraints or [],
                "messages": messages_to_dict(messages),
            }
            with open(self._path(session_id), "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            print(f"Failed to save session {session_id}: {e}")

    def clear(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()


_store = SessionStore()


def load_session(session_id: str) -> Optional[dict]:
    return _store.load(session_id)


def save_session(session_id: str, **kwargs) -> None:
    _store.save(session_id, **kwargs)


def clear_session(session_id: str) -> None:
    _store.clear(session_id)
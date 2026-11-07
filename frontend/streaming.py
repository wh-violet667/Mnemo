"""Background streaming for the Streamlit UI.

The HTTP stream runs in a daemon thread so the page stays interactive and a
stop button can be clicked while tokens are arriving.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

import requests

API_BASE_URL = os.getenv("MNEMO_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_ID = int(os.getenv("MNEMO_USER_ID", "1"))


class StreamSession:
    def __init__(
        self,
        query: str,
        knowledge_base_id: str,
        conversation_id: str,
        previous_messages: List[Dict[str, Any]],
    ) -> None:
        self.query = query
        self.knowledge_base_id = knowledge_base_id
        self.conversation_id = conversation_id
        self.previous_messages = previous_messages
        self.answer = ""
        self.sources: List[Dict[str, Any]] = []
        self.status = "正在连接…"
        self.error = ""
        self.done = False
        self.stopped = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.response: Optional[requests.Response] = None

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "answer": self.answer,
                "sources": list(self.sources),
                "status": self.status,
                "error": self.error,
                "done": self.done,
                "stopped": self.stopped,
            }

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            response = self.response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def _set(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _run(self) -> None:
        params = {
            "query": self.query,
            "knowledge_base_id": self.knowledge_base_id,
            "user_id": USER_ID,
            "conversation_id": self.conversation_id,
            "top_k": 5,
        }
        response: Optional[requests.Response] = None
        try:
            response = requests.get(
                f"{API_BASE_URL}/chat_with_agent/stream",
                params=params,
                stream=True,
                timeout=180,
            )
            with self.lock:
                self.response = response
            if response.status_code != 200:
                raise RuntimeError(f"后端返回 {response.status_code}")

            self._set(status="正在生成…")
            for line in response.iter_lines(decode_unicode=True):
                if self.stop_event.is_set():
                    break
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                event_type = event.get("type")
                content = event.get("content", "")
                if event_type == "status":
                    with self.lock:
                        if not self.answer:
                            self.status = str(content)
                elif event_type == "answer":
                    with self.lock:
                        self.answer += str(content)
                        self.status = ""
                elif event_type == "sources":
                    self._set(sources=content if isinstance(content, list) else [])
                elif event_type == "error":
                    self._set(error=str(content))
        except Exception as exc:
            if not self.stop_event.is_set():
                self._set(error=str(exc))
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            with self.lock:
                self.response = None
                self.stopped = self.stop_event.is_set()
                self.done = True
                self.status = ""

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

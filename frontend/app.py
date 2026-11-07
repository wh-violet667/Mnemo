"""Mnemo Streamlit UI — Studio Sidebar layout.

设计目标：
- 深色侧栏：会话 / 知识库 / 记忆
- 浅色对话区：引用来源、记忆命中、工具状态
- 右侧记忆抽屉：按类型查看、删除长期记忆
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import bleach
import requests
import streamlit as st

from streaming import StreamSession

API_BASE_URL = os.getenv("MNEMO_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_ID = int(os.getenv("MNEMO_USER_ID", "1"))

MEMORY_TYPES = {
    "semantic": ("语义", "sem"),
    "episodic": ("情节", "epi"),
    "procedural": ("程序", "proc"),
    "summary": ("摘要", "sum"),
}
FILTER_OPTIONS = ["全部", "语义", "情节", "程序", "摘要"]
FILTER_TO_TYPE = {label: key for key, (label, _) in MEMORY_TYPES.items()}
FILTER_TO_TYPE["全部"] = ""

CHAT_HEIGHT = 560
MEMORY_HEIGHT = 800

MARKDOWN_TAGS = {
    "p", "br", "hr", "strong", "em", "del", "blockquote",
    "ul", "ol", "li", "code", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td",
    "a", "sup", "sub",
}
MARKDOWN_ATTRS = {
    "a": ["href", "title"],
    "code": ["class"],
    "pre": ["class"],
    "th": ["align"],
    "td": ["align"],
}

st.set_page_config(
    page_title="Mnemo",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

CSS = """
<style>
  :root {
    --ink: #241f1b;
    --muted: #6f665b;
    --line: #e5ddd1;
    --canvas: #f7f3ec;
    --panel: #fffdf9;
    --accent: #bf5737;
    --accent-dark: #9a4229;
    --sidebar: #1c1a18;
    --sidebar-line: #302c27;
    --sidebar-text: #efe9df;
    --sidebar-muted: #b3a99b;
  }
  .stApp { background: var(--canvas); }
  .block-container { max-width: 1500px; padding: .8rem 1.4rem 1.1rem; }
  header[data-testid="stHeader"] { background: transparent; pointer-events: none; }
  header[data-testid="stHeader"] button { pointer-events: auto; }
  div[data-testid="stToolbar"] { pointer-events: none !important; }
  div[data-testid="stAppDeployButton"] { display: none; }
  div[data-testid="InputInstructions"] { display: none !important; }
  #MainMenu, footer { visibility: hidden; }
  button[data-testid="stExpandSidebarButton"] {
    width: 32px; height: 32px; border-radius: 8px;
    background: #fffdf9; border: 1px solid #ddd4c7;
    color: #6c6459; box-shadow: 0 2px 10px rgba(50, 35, 20, .12);
    z-index: 1000001;
  }
  button[data-testid="stExpandSidebarButton"]:hover {
    background: #fdf0ea; color: var(--accent-dark);
  }

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] { background: var(--sidebar); }
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1rem .8rem; }
  section[data-testid="stSidebar"] * { color: var(--sidebar-text); }
  section[data-testid="stSidebar"] .brand-row {
    display: flex; align-items: center; gap: 9px; padding: 2px 4px 14px;
  }
  section[data-testid="stSidebar"] .brand-logo {
    width: 28px; height: 28px; border-radius: 8px; background: var(--accent);
    color: #fff; display: grid; place-items: center; font-weight: 800; font-size: 15px;
  }
  section[data-testid="stSidebar"] .brand-name { font-weight: 700; font-size: 17px; color: #fff; letter-spacing: -.02em; }
  section[data-testid="stSidebar"] .brand-ver {
    margin-left: auto; font: .72rem ui-monospace, Consolas, monospace; color: var(--sidebar-muted);
    border: 1px solid #3a3530; border-radius: 4px; padding: 2px 5px;
  }
  section[data-testid="stSidebar"] .side-label {
    font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--sidebar-muted); padding: 10px 6px 4px;
  }
  section[data-testid="stSidebar"] .stButton button {
    background: transparent; border: 1px solid transparent; color: #d9d2c8;
    border-radius: 7px; justify-content: flex-start; text-align: left;
    padding: .42rem .55rem; font-size: .82rem; min-height: 2rem;
  }
  section[data-testid="stSidebar"] [class*="st-key-conv_"] button,
  section[data-testid="stSidebar"] [class*="st-key-conv_"] button * {
    justify-content: flex-start !important; text-align: left !important;
  }
  section[data-testid="stSidebar"] .stButton button:hover {
    background: #26231f; color: #fff; border-color: transparent;
  }
  section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: var(--accent); color: #fff; font-weight: 600;
  }
  section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: #a94a30; }
  section[data-testid="stSidebar"] .stButton button:focus,
  section[data-testid="stSidebar"] .stButton button:focus-visible,
  section[data-testid="stSidebar"] .stButton button:active {
    background: #26231f !important; color: #fff !important;
    border-color: #3a3530 !important; box-shadow: none !important;
  }
  section[data-testid="stSidebar"] .stButton button[kind="primary"]:focus,
  section[data-testid="stSidebar"] .stButton button[kind="primary"]:focus-visible,
  section[data-testid="stSidebar"] .stButton button[kind="primary"]:active {
    background: var(--accent) !important; color: #fff !important;
    border-color: var(--accent) !important;
  }
  section[data-testid="stSidebar"] input {
    background: #26231f !important; color: var(--sidebar-text) !important;
    border-color: #3a3530 !important; border-radius: 7px !important;
  }
  section[data-testid="stSidebar"] input::placeholder { color: #b3a99b !important; }
  section[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
  section[data-testid="stSidebar"] [data-baseweb="input"],
  section[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: #26231f !important; border-color: #3a3530 !important;
    border-radius: 7px !important; box-shadow: none !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #26231f !important; border-color: #3a3530 !important;
    color: var(--sidebar-text) !important; border-radius: 7px !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] svg { color: var(--sidebar-muted) !important; }
  section[data-testid="stSidebar"] details {
    background: #24211e; border: 1px solid #322e29; border-radius: 8px; margin-top: 6px;
  }
  section[data-testid="stSidebar"] summary { color: #d9d2c8 !important; font-size: .78rem; }
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #26231f; border-radius: 8px; padding: 6px;
  }
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #b8b0a5 !important; }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
    color: #b8b0a5 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {
    background: #26231f !important;
  }
  section[data-testid="stSidebar"] hr { border-color: var(--sidebar-line); }
  section[data-testid="stSidebar"] .side-foot {
    border-top: 1px solid var(--sidebar-line); margin-top: 12px; padding: 10px 4px 2px;
    font-size: .72rem; color: var(--sidebar-muted);
  }

  /* ---------- main ---------- */
  .main-header {
    display: flex; align-items: flex-start; gap: 14px; padding: 6px 2px 14px;
    border-bottom: 1px solid var(--line); margin-bottom: 4px;
  }
  .main-header h1 { margin: 0; font-size: 1.12rem; letter-spacing: -.02em; color: var(--ink); }
  .main-header .sub { margin-top: 3px; font-size: .72rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  .main-header .dot { width: 5px; height: 5px; border-radius: 50%; background: #5f8b62; display: inline-block; }
  .main-header .pills { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
  .pill {
    display: inline-flex; align-items: center; gap: 4px; height: 26px; padding: 0 9px;
    border: 1px solid #ddd4c7; background: #fffdf9; border-radius: 999px;
    font-size: .72rem; color: #6c6459; white-space: nowrap;
  }
  .pill.accent { color: var(--accent-dark); border-color: #e8c9bc; background: #fdf0ea; }
  .pill .mini { font-size: .72rem; opacity: .62; }

  /* messages */
  .msg-row { display: flex; gap: 10px; margin: 0 0 18px; }
  .msg-row.user { justify-content: flex-end; }
  .msg-avatar {
    width: 28px; height: 28px; border-radius: 50%; flex: 0 0 auto;
    display: grid; place-items: center; font-size: .72rem; font-weight: 700;
  }
  .msg-avatar.me { background: #ded3c4; color: #6d6255; }
  .msg-avatar.ai { background: var(--accent); color: #fff; font-size: 13px; }
  .msg-body { min-width: 0; max-width: 92%; }
  .msg-row.user .msg-body { max-width: 74%; }
  .bubble {
    background: #e9dfd1; border-radius: 14px 14px 4px 14px;
    padding: 9px 13px; font-size: .88rem; line-height: 1.6; color: #2a2520;
  }
  .msg-name { font-size: .76rem; font-weight: 700; color: var(--ink); margin: 2px 0 5px; display: flex; align-items: center; gap: 8px; }
  .msg-status { font-size: .72rem; color: var(--muted); font-weight: 400; }
  .answer { font-size: .88rem; line-height: 1.72; color: #2a2520; }
  .answer p { margin: 0 0 8px; }
  .answer p:last-child { margin-bottom: 0; }
  .answer ul, .answer ol { margin: 4px 0 8px 1.2rem; }
  .cite {
    display: inline-flex; align-items: center; justify-content: center; min-width: 17px;
    height: 17px; padding: 0 4px; margin: 0 1px; border-radius: 4px;
    background: #e4f2ee; color: #0f766e; font: .72rem ui-monospace, Consolas, monospace;
    vertical-align: 1px; text-decoration: none;
  }
  .cite:hover { background: #cfe9e2; }
  .source-card {
    margin-top: 10px; background: #fffdf9; border: 1px solid #e3dbce;
    border-radius: 10px; max-width: 560px; overflow: hidden;
  }
  .source-summary {
    display: flex; gap: 9px; align-items: flex-start; padding: 9px 11px;
    cursor: pointer; list-style: none; outline: none;
  }
  .source-summary::-webkit-details-marker { display: none; }
  .source-summary:hover { background: #faf4eb; }
  .source-head { min-width: 0; flex: 1; }
  .file-icon {
    width: 30px; height: 34px; border-radius: 6px; background: #f4e6de;
    color: #9e4c32; display: grid; place-items: center; flex: 0 0 auto; font-size: .72rem;
  }
  .source-name { font-size: .78rem; font-weight: 700; color: var(--ink); }
  .source-meta { font-size: .72rem; color: var(--muted); margin-top: 2px; }
  .source-toggle {
    margin-left: auto; color: var(--accent-dark); font-size: .72rem; white-space: nowrap;
  }
  .source-toggle::before { content: "展开"; }
  .source-card[open] .source-toggle::before { content: "收起"; }
  .source-note { margin-top: 8px; font-size: .72rem; color: #6f665b; }
  .source-snippet {
    font-size: .72rem; color: #6d6458; line-height: 1.5;
    padding: 0 11px 10px 50px; max-height: 220px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
  }
  div[class*="st-key-msg_actions_"] { margin: -10px 0 10px; }
  div[class*="st-key-msg_actions_"] button {
    background: transparent !important; border: 0 !important;
    color: var(--muted) !important; font-size: .72rem !important;
    min-height: 24px !important; padding: 2px 4px !important;
    box-shadow: none !important; white-space: nowrap !important;
  }
  div[class*="st-key-msg_actions_"] button:hover {
    color: var(--accent-dark) !important; background: #f6ede5 !important;
  }
  .memory-hit {
    margin-top: 8px; display: inline-flex; align-items: center; gap: 5px;
    font-size: .72rem; color: #5f574d; background: #efe7db; border-radius: 999px; padding: 3px 8px;
  }
  .status-line { font-size: .72rem; color: var(--muted); margin: 2px 0 8px; }
  .inline-error { color: #9a4229; font-size: .78rem; margin-top: 6px; }

  /* composer */
  div[data-testid="stForm"] {
    background: var(--panel); border: 1px solid #ddd4c7; border-radius: 14px;
    padding: 10px 12px 8px; box-shadow: 0 6px 22px rgba(60,45,30,.06);
  }
  div[data-testid="stForm"] textarea {
    border: 0 !important; background: transparent !important; box-shadow: none !important;
    font-size: .9rem !important; color: var(--ink) !important; padding: 2px 0 !important;
  }
  div[data-testid="stForm"] textarea::placeholder { color: #6f665b !important; }
  div[data-testid="stForm"] [data-testid="stTextAreaRootElement"],
  div[data-testid="stForm"] [data-baseweb="textarea"],
  div[data-testid="stForm"] [data-baseweb="base-input"] {
    background: transparent !important; border-color: transparent !important;
    box-shadow: none !important;
  }
  div[data-testid="stForm"] button {
    background: var(--accent) !important; color: #fff !important; border: 0 !important;
    border-radius: 8px !important; font-weight: 700 !important;
  }
  div[data-testid="stForm"] button:hover { background: #a94a30 !important; }
  div[data-testid="stForm"] div[data-testid="stElementContainer"]:has(
    div[data-testid="stFormSubmitButton"]
  ) { width: auto !important; margin-left: auto !important; }
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    width: 38px !important; height: 38px !important; min-height: 38px !important;
    display: block !important; padding: 0 !important; border-radius: 10px !important;
  }
  .composer-pill { font-size: .72rem; color: #6f665b; padding-top: 8px; }
  .hintline { text-align: center; font-size: .72rem; color: #6f665b; margin-top: 5px; }

  /* chat scroll area */
  div.st-key-chat_scroll {
    height: calc(100vh - 340px) !important;
    min-height: 220px;
    flex: 0 0 auto !important;
  }
  div[data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll) {
    height: calc(100vh - 340px) !important;
    min-height: 220px !important;
    flex: 0 0 calc(100vh - 340px) !important;
  }


  /* memory */
  div.st-key-memory_panel {
    background: #fffcf7; border: 1px solid var(--line) !important;
    border-radius: 14px !important; padding: 12px 12px 8px !important;
    height: 800px; max-height: calc(100vh - 40px);
  }
  div[data-testid="stButtonGroup"] { gap: 4px !important; margin: 0 0 8px; }
  div[data-testid="stButtonGroup"] button {
    min-height: 28px !important; padding: 3px 7px !important;
    font-size: .72rem !important; border-radius: 6px !important;
  }
  div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
    background: var(--accent) !important; border-color: var(--accent) !important;
    color: #fff !important;
  }
  div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_control"] {
    background: transparent !important; border-color: #e3dbce !important;
    color: #6c6459 !important;
  }
  div.st-key-memory_panel button[kind="tertiary"] {
    background: transparent !important; border: 0 !important; color: #a79d90 !important;
    min-height: 24px !important; padding: 2px 4px !important; box-shadow: none !important;
  }
  div.st-key-memory_panel button[kind="tertiary"]:hover {
    background: #f6ede5 !important; color: var(--accent-dark) !important;
  }
  .mem-head { display: flex; align-items: center; gap: 8px; }
  .mem-head h3 { margin: 0; font-size: .95rem; color: var(--ink); }
  .mem-sub { font-size: .72rem; color: var(--muted); line-height: 1.45; margin: 3px 0 8px; }
  .mem-card {
    border: 1px solid #e7dfd3; background: #fff; border-radius: 9px;
    padding: 9px 10px; margin: 0 0 7px;
  }
  .mem-top { display: flex; align-items: center; gap: 5px; margin-bottom: 4px; }
  .mem-type {
    font-size: .72rem; letter-spacing: .04em; text-transform: uppercase;
    border-radius: 3px; padding: 1px 5px; font-weight: 700;
  }
  .t-sem { color: #0f6b60; background: #e4f2ee; }
  .t-epi { color: #8a5a22; background: #f7ecd9; }
  .t-proc { color: #5a4a8a; background: #eeeafa; }
  .t-sum { color: #6d6255; background: #f0ebe3; }
  .mem-score { margin-left: auto; font: .72rem ui-monospace, Consolas, monospace; color: #6f665b; }
  .mem-content { font-size: .74rem; line-height: 1.5; color: #3a332c; }
  .mem-meta { display: flex; gap: 5px; margin-top: 5px; color: #6f665b; font-size: .72rem; }
  .mem-confirm { margin: 2px 0 6px; color: #9a4229; font-size: .72rem; }
  .mem-more { margin-top: 3px; }
  .mem-more summary {
    list-style: none; cursor: pointer; color: var(--accent-dark);
    font-size: .72rem; width: fit-content;
  }
  .mem-more summary::-webkit-details-marker { display: none; }
  .mem-more summary::before { content: "＋ "; }
  .mem-more[open] summary::before { content: "－ "; }
  .mem-full {
    margin-top: 5px; padding-top: 5px; border-top: 1px dashed #eee5d8;
    max-height: 180px; overflow-y: auto; padding-right: 2px;
  }
  div.st-key-memory_panel [data-testid="stTextInputRootElement"] {
    background: #fff !important; border: 1px solid #e7dfd3 !important;
    border-radius: 8px !important; min-height: 34px !important;
  }
  div.st-key-memory_panel [data-baseweb="base-input"] { background: transparent !important; }
  div.st-key-memory_panel input { color: var(--ink) !important; font-size: .72rem !important; }
  div.st-key-memory_panel [data-testid="stTextInput"] { margin-bottom: 2px; }
  .mem-empty { color: #6f665b; font-size: .72rem; text-align: center; padding: 26px 8px; line-height: 1.6; }
  .mem-foot { border-top: 1px solid var(--line); padding-top: 7px; margin-top: 6px; color: var(--muted); font-size: .72rem; }
  .mem-tabs { margin: 0 0 6px; }

  /* welcome */
  .welcome { padding: 52px 10px 22px; text-align: center; }
  .welcome h2 { margin: 0 0 6px; font-size: 1.45rem; letter-spacing: -.03em; color: var(--ink); }
  .welcome p { margin: 0 auto; max-width: 430px; color: var(--muted); font-size: .8rem; line-height: 1.6; }
  .welcome .marks { margin: 18px auto 0; color: #7a7064; font-size: .72rem; letter-spacing: .08em; }

  @media (max-width: 1100px) {
    .main-header .pills .pill:not(.accent) { display: none; }
    .msg-body { max-width: 100%; }
    div[data-testid="stHorizontalBlock"]:has(.st-key-memory_panel) > div[data-testid="stColumn"]:first-child {
      display: none !important;
    }
    div[data-testid="stColumn"]:has(.st-key-memory_panel) {
      flex: 1 1 100% !important; width: 100% !important; min-width: 0 !important;
    }
    div.st-key-memory_panel {
      height: calc(100vh - 48px) !important; max-height: 800px !important;
      min-height: 0 !important; flex: 0 0 auto !important;
      box-shadow: 0 18px 50px rgba(40, 28, 18, .12);
    }
    div[data-testid="stLayoutWrapper"]:has(> .st-key-memory_panel) {
      height: calc(100vh - 48px) !important;
      flex: 0 0 calc(100vh - 48px) !important;
    }
  }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- state & API
DEFAULTS = {
    "bootstrapped": False,
    "boot_error": None,
    "conversations": [],
    "conversation_id": None,
    "messages": [],
    "kbs": [],
    "selected_kb": "",
    "kb_files": {},
    "kb_widget": "",
    "pending_kb": None,
    "memory_open": False,
    "memories": [],
    "memory_count": 0,
    "memory_error": "",
    "active_stream": None,
    "retrieval_mode": "fast",
    "history_loaded_for": None,
    "confirm_delete_kb": False,
    "confirm_delete_chat": False,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api(method: str, path: str, **kwargs) -> Any:
    timeout = kwargs.pop("timeout", 20)
    try:
        response = requests.request(
            method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = (exc.response.text or "")[:200]
        raise RuntimeError(detail or str(exc)) from exc


def fetch_conversations() -> List[Dict[str, Any]]:
    return api(
        "GET", "/conversations", params={"user_id": USER_ID}, timeout=10
    ).get("conversations", [])


def find_fallback_sources(
    previous_messages: List[Dict[str, Any]], content: str
) -> List[Dict[str, Any]]:
    """追问可能沿用上文的 [n] 引用，但没有重新检索，回填最近一轮来源。"""
    cited = {int(index) for index in re.findall(r"\[(\d+)\]", content)}
    if not cited:
        return []
    for message in reversed(previous_messages):
        if message.get("role") != "assistant":
            continue
        matched = [
            source
            for source in message.get("sources") or []
            if source.get("index") in cited
        ]
        if matched:
            return matched
    return []


def fetch_messages(conversation_id: str) -> List[Dict[str, Any]]:
    data = api(
        "GET",
        f"/conversations/{conversation_id}/messages",
        params={"user_id": USER_ID, "limit": 300},
        timeout=10,
    )
    messages = []
    for item in data.get("messages", []):
        created = item.get("created_at")
        try:
            timestamp = datetime.fromisoformat(
                str(created).replace("Z", "+00:00")
            ).strftime("%H:%M")
        except Exception:
            timestamp = ""
        messages.append(
            {
                "role": "user" if item.get("role") == "human" else "assistant",
                "content": item.get("content", ""),
                "sources": item.get("sources") or [],
                "timestamp": timestamp,
            }
        )

    assistant_messages: List[Dict[str, Any]] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        if not message["sources"]:
            fallback = find_fallback_sources(assistant_messages, message["content"])
            if fallback:
                message["sources"] = fallback
                message["sources_from_previous_turn"] = True
        if message["sources"]:
            assistant_messages.append(message)

    return messages


def fetch_kbs() -> List[str]:
    return api(
        "GET", "/knowledge-bases", params={"user_id": USER_ID}, timeout=10
    ).get("knowledge_bases", [])


def fetch_files(kb_id: str) -> List[Dict[str, Any]]:
    return api(
        "GET",
        f"/knowledge-bases/{kb_id}/files",
        params={"user_id": USER_ID},
        timeout=10,
    ).get("files", [])


def cache_kb_files(kb_id: str) -> None:
    """缓存知识库文件列表，供顶栏展示文件数。"""
    if not kb_id:
        return
    try:
        st.session_state.kb_files[kb_id] = fetch_files(kb_id)
    except Exception:
        pass


def pick_kb(kb_id: Optional[str]) -> str:
    """只允许真实存在的知识库；没有知识库时返回空字符串。"""
    kbs = st.session_state.get("kbs") or []
    if kb_id and kb_id in kbs:
        return kb_id
    return kbs[0] if kbs else ""


def fetch_memories() -> List[Dict[str, Any]]:
    return api(
        "GET", "/memories", params={"user_id": USER_ID, "limit": 200}, timeout=15
    ).get("memories", [])


def bootstrap() -> None:
    if st.session_state.bootstrapped:
        return
    try:
        kbs = fetch_kbs()
        st.session_state.kbs = kbs
        initial_kb = kbs[0] if kbs else ""

        conversations = fetch_conversations()
        if not conversations:
            conversation = api(
                "POST",
                "/conversations",
                params={
                    "user_id": USER_ID,
                    "title": "新对话",
                    "knowledge_base_id": initial_kb,
                },
                timeout=10,
            )
            conversations = [conversation]

        st.session_state.conversations = conversations
        current = next(
            (
                item
                for item in conversations
                if item["conversation_id"] == st.session_state.conversation_id
            ),
            conversations[0],
        )
        st.session_state.conversation_id = current["conversation_id"]
        st.session_state.selected_kb = pick_kb(current.get("knowledge_base_id"))
        st.session_state.kb_widget = st.session_state.selected_kb
        cache_kb_files(st.session_state.selected_kb)
        st.session_state.messages = fetch_messages(current["conversation_id"])
        st.session_state.history_loaded_for = current["conversation_id"]

        try:
            memories = fetch_memories()
            st.session_state.memories = memories
            st.session_state.memory_count = len(memories)
            st.session_state.memory_error = ""
        except Exception as exc:
            st.session_state.memories = []
            st.session_state.memory_count = 0
            st.session_state.memory_error = str(exc)

        st.session_state.bootstrapped = True
        st.session_state.boot_error = None
    except Exception as exc:
        st.session_state.boot_error = str(exc)


# ---------------------------------------------------------------- actions
def refresh_conversations() -> None:
    try:
        st.session_state.conversations = fetch_conversations()
    except Exception:
        pass


def persist_knowledge_base(kb_id: str) -> None:
    """切换到真实知识库时立即写回会话，刷新后不再回退到旧值。"""
    conversation_id = st.session_state.conversation_id
    if not conversation_id:
        return
    try:
        api(
            "PATCH",
            f"/conversations/{conversation_id}/knowledge-base",
            params={"user_id": USER_ID, "knowledge_base_id": kb_id},
            timeout=10,
        )
        for conversation in st.session_state.conversations:
            if conversation["conversation_id"] == conversation_id:
                conversation["knowledge_base_id"] = kb_id
                break
    except Exception as exc:
        st.warning(f"知识库选择未保存：{exc}")


def new_conversation() -> None:
    try:
        kb = pick_kb(st.session_state.selected_kb)
        conversation = api(
            "POST",
            "/conversations",
            params={
                "user_id": USER_ID,
                "title": "新对话",
                "knowledge_base_id": kb,
            },
            timeout=10,
        )
        st.session_state.conversations = fetch_conversations() or [conversation]
        st.session_state.conversation_id = conversation["conversation_id"]
        st.session_state.messages = []
        st.session_state.history_loaded_for = conversation["conversation_id"]
        st.session_state.selected_kb = kb
        st.session_state.kb_widget = kb
        st.session_state.confirm_delete_chat = False
        cache_kb_files(kb)
        st.rerun()
    except Exception as exc:
        st.error(f"创建会话失败：{exc}")


def select_conversation(conversation: Dict[str, Any]) -> None:
    conversation_id = conversation["conversation_id"]
    try:
        st.session_state.messages = fetch_messages(conversation_id)
    except Exception as exc:
        st.error(f"读取会话失败：{exc}")
        return

    st.session_state.conversation_id = conversation_id
    kb = pick_kb(conversation.get("knowledge_base_id"))
    st.session_state.selected_kb = kb
    st.session_state.kb_widget = kb
    st.session_state.history_loaded_for = conversation_id
    st.session_state.confirm_delete_chat = False
    cache_kb_files(kb)
    st.rerun()


def delete_current_conversation() -> None:
    conversation_id = st.session_state.conversation_id
    if not conversation_id:
        return
    try:
        api(
            "DELETE",
            f"/conversations/{conversation_id}",
            params={"user_id": USER_ID},
            timeout=10,
        )
    except Exception as exc:
        st.error(f"删除会话失败：{exc}")
        return

    try:
        conversations = fetch_conversations()
    except Exception:
        conversations = []
    st.session_state.conversations = conversations
    if conversations:
        current = conversations[0]
        st.session_state.conversation_id = current["conversation_id"]
        st.session_state.messages = fetch_messages(current["conversation_id"])
        kb = pick_kb(current.get("knowledge_base_id"))
        st.session_state.selected_kb = kb
        st.session_state.kb_widget = kb
    else:
        new_conversation()
    st.rerun()


def _last_user_content(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _start_stream(text: str, append_user: bool) -> None:
    if st.session_state.get("active_stream"):
        return
    if not st.session_state.conversation_id:
        new_conversation()

    messages = st.session_state.messages
    previous_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if append_user:
        messages.append(
            {
                "role": "user",
                "content": text,
                "sources": [],
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )

    session = StreamSession(
        query=text,
        knowledge_base_id=st.session_state.selected_kb,
        conversation_id=st.session_state.conversation_id,
        previous_messages=previous_messages,
        retrieval_mode=st.session_state.get("retrieval_mode", "fast"),
    )
    st.session_state.active_stream = session
    session.start()
    st.rerun()


def send_query(text: str) -> None:
    text = (text or "").strip()
    if not text or st.session_state.get("active_stream"):
        return
    _start_stream(text, append_user=True)


def regenerate_last_answer() -> None:
    messages = st.session_state.messages
    if not messages or messages[-1].get("role") != "assistant":
        return
    if st.session_state.get("active_stream"):
        return
    user_text = _last_user_content(messages[:-1])
    if not user_text:
        return

    try:
        api(
            "DELETE",
            f"/conversations/{st.session_state.conversation_id}/last-turn",
            params={"user_id": USER_ID},
            timeout=10,
        )
    except Exception as exc:
        st.error(f"重新生成失败：{exc}")
        return

    messages.pop()
    _start_stream(user_text, append_user=False)


def retry_last_answer() -> None:
    messages = st.session_state.messages
    if not messages or messages[-1].get("role") != "assistant":
        return
    if st.session_state.get("active_stream"):
        return
    user_text = _last_user_content(messages[:-1])
    if not user_text:
        return
    messages.pop()
    _start_stream(user_text, append_user=False)


# ---------------------------------------------------------------- rendering
def render_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    timestamp: str = "",
    streaming: bool = False,
    sources_from_previous_turn: bool = False,
    anchor_key: str = "msg",
    trace: Optional[Dict[str, Any]] = None,
) -> str:
    if role == "user":
        return (
            "<div class='msg-row user'>"
            f"<div class='msg-body'><div class='bubble'>{html.escape(content)}</div></div>"
            "<div class='msg-avatar me'>V</div>"
            "</div>"
        )

    from markdown import markdown

    try:
        rendered = markdown(content or "", extensions=["fenced_code", "tables"])
        rendered = bleach.clean(
            rendered,
            tags=MARKDOWN_TAGS,
            attributes=MARKDOWN_ATTRS,
            protocols={"http", "https", "mailto"},
            strip=True,
        )
    except Exception:
        rendered = html.escape(content or "")

    anchor = html.escape(anchor_key or "msg")
    rendered = re.sub(
        r"\[(\d+)\]",
        lambda match: (
            f"<a class='cite' href='#source-{anchor}-{match.group(1)}' "
            f"title='查看来源 {match.group(1)}'>{match.group(1)}</a>"
        ),
        rendered,
    )

    trace_html = ""
    if isinstance(trace, dict) and trace.get("mode"):
        mode_label = {"fast": "快速", "deep": "深入", "auto": "自动"}.get(
            str(trace.get("mode")), str(trace.get("mode"))
        )
        elapsed_s = (trace.get("elapsed_ms") or 0) / 1000
        parts = [
            f"本次：{mode_label}",
            f"{trace.get('rounds', '?')} 轮",
            f"{trace.get('parent_count', '?')} 条证据",
            f"{elapsed_s:.1f}s",
        ]
        if trace.get("escalated"):
            parts.append("无结果已升级深入")
        if trace.get("degraded"):
            parts.append("Agentic 关闭已降级")
        trace_html = f"<div class='source-note'>{' · '.join(parts)}</div>"

    source_html = ""
    if sources_from_previous_turn and sources:
        source_html += (
            "<div class='source-note'>本轮未重新检索，沿用上一轮来源</div>"
        )
    for source in (sources or [])[:5]:
        index = html.escape(str(source.get("index", "")))
        raw_name = str(source.get("file_name", "未知文件"))
        file_name = html.escape(raw_name)
        extension = raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else ""
        icon = {"pdf": "PDF", "docx": "DOC", "doc": "DOC", "txt": "TXT"}.get(
            extension, "FILE"
        )
        snippet = html.escape(str(source.get("snippet", "")))
        score = source.get("score")
        meta = "引用片段"
        if isinstance(score, (int, float)):
            meta = f"相关度 {score:.3f}"
        source_id = f"source-{anchor}-{index}"
        source_html += (
            f"<details class='source-card' id='{source_id}' open>"
            "<summary class='source-summary'>"
            f"<div class='file-icon'>{icon}</div>"
            "<div class='source-head'>"
            f"<div class='source-name'>[{index}] {file_name}</div>"
            f"<div class='source-meta'>{meta}</div>"
            "</div>"
            "<span class='source-toggle'></span>"
            "</summary>"
            f"<div class='source-snippet'>{snippet}</div>"
            "</details>"
        )

    status = (
        "<span class='msg-status'>正在生成…</span>"
        if streaming
        else f"<span class='msg-status'>{html.escape(timestamp or '已保存')}</span>"
    )
    return (
        "<div class='msg-row assistant'>"
        "<div class='msg-avatar ai'>M</div>"
        "<div class='msg-body'>"
        f"<div class='msg-name'>Mnemo {status}</div>"
        f"<div class='answer'>{rendered}</div>"
        f"{trace_html}"
        f"{source_html}"
        "</div></div>"
    )


def render_welcome() -> None:
    st.markdown(
        "<div class='welcome'>"
        "<h2>今天想记住点什么？</h2>"
        "<p>问知识库、回忆之前的偏好，或让我总结一份文档。"
        "回答会标注来源，记忆会在后台自动整理。</p>"
        "<div class='marks'>MEMORY · RETRIEVAL · CITATIONS</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    suggestions = [
        "退款期限是多久？",
        "总结一下差旅报销流程",
        "我之前说过什么偏好？",
    ]
    columns = st.columns(3)
    for index, (column, text) in enumerate(zip(columns, suggestions)):
        with column:
            if st.button(text, key=f"suggest_{index}", use_container_width=True):
                send_query(text)


def render_header() -> None:
    current = next(
        (
            item
            for item in st.session_state.conversations
            if item["conversation_id"] == st.session_state.conversation_id
        ),
        {},
    )
    title = html.escape(current.get("title") or "新对话")
    kb = html.escape(st.session_state.selected_kb or "未选择知识库")
    file_count = len(st.session_state.kb_files.get(st.session_state.selected_kb, []) or [])
    file_text = f" · {file_count} 文件" if file_count else ""
    memory_status = (
        "记忆 · 离线"
        if st.session_state.memory_error
        else f"记忆 · {st.session_state.memory_count} 条"
    )
    st.markdown(
        "<div class='main-header'>"
        f"<div><h1>{title}</h1>"
        f"<div class='sub'><span class='dot'></span>{kb}{file_text} · 已保存</div></div>"
        "<div class='pills'>"
        f"<span class='pill accent'>{memory_status}</span>"
        "<span class='pill'>qwen-plus</span>"
        "<span class='pill'>qwen3-rerank</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_composer() -> None:
    streaming = bool(st.session_state.get("active_stream"))
    with st.form("composer", clear_on_submit=True):
        prompt = st.text_area(
            "消息",
            height=62,
            placeholder="继续追问，或让我总结一份文档…",
            label_visibility="collapsed",
            disabled=streaming,
        )
        col1, col2, col3 = st.columns([1.5, 1.5, 0.45])
        with col1:
            st.markdown(
                f"<div class='composer-pill'>知识库 · "
                f"{html.escape(st.session_state.selected_kb or '未选择')}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("<div class='composer-pill'>模型 · qwen-plus</div>", unsafe_allow_html=True)
        with col3:
            submitted = st.form_submit_button(
                "↑", use_container_width=False, disabled=streaming
            )
    st.markdown(
        "<div class='hintline'>Ctrl + Enter 发送 · Enter 换行 · 回答会标注来源</div>",
        unsafe_allow_html=True,
    )
    st.segmented_control(
        "检索模式",
        options=["fast", "auto", "deep"],
        format_func=lambda m: {"auto": "自动", "fast": "快速", "deep": "深入"}[m],
        key="retrieval_mode",
        label_visibility="collapsed",
    )
    if submitted and prompt.strip():
        send_query(prompt)


def finalize_stream(session: StreamSession) -> None:
    snap = session.snapshot()
    answer = snap["answer"].strip()
    sources = snap["sources"]
    error = snap["error"]
    stopped = snap["stopped"]

    sources_from_previous_turn = False
    if not sources:
        fallback = find_fallback_sources(session.previous_messages, answer)
        if fallback:
            sources = fallback
            sources_from_previous_turn = True

    if error and not answer:
        answer = f"❌ {error}"
    elif error:
        answer = f"{answer}\n\n❌ {error}"
    if stopped:
        answer = f"{answer}\n\n*（已停止生成）*".strip() if answer else "（已停止生成）"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer or "❌ 没有收到回答",
            "sources": sources,
            "sources_from_previous_turn": sources_from_previous_turn,
            "trace": snap.get("trace"),
            "error": bool(error),
            "stopped": stopped,
            "timestamp": datetime.now().strftime("%H:%M"),
        }
    )
    st.session_state.active_stream = None
    refresh_conversations()
    st.rerun()


@st.fragment(run_every=0.5)
def render_live_stream() -> None:
    session = st.session_state.get("active_stream")
    if not session:
        return
    snap = session.snapshot()

    if snap["answer"]:
        st.markdown(
            render_message(
                "assistant",
                snap["answer"],
                snap["sources"],
                streaming=True,
                anchor_key="live",
                trace=snap.get("trace"),
            ),
            unsafe_allow_html=True,
        )
    elif snap["status"]:
        st.markdown(
            f"<div class='status-line'>{html.escape(snap['status'])}</div>",
            unsafe_allow_html=True,
        )

    if snap["error"]:
        st.markdown(
            f"<div class='inline-error'>❌ {html.escape(snap['error'])}</div>",
            unsafe_allow_html=True,
        )

    if snap["done"]:
        finalize_stream(session)
        return

    stop_col, info_col = st.columns([1, 8])
    with stop_col:
        if st.button(
            "停止",
            key="stop_stream",
            type="secondary",
            use_container_width=False,
        ):
            session.stop()
            st.rerun(scope="fragment")
    with info_col:
        st.caption("正在生成，可随时停止")


def render_message_actions(
    message: Dict[str, Any], index: int, is_last: bool
) -> None:
    if message.get("role") != "assistant":
        return
    with st.container(key=f"msg_actions_{index}"):
        copy_col, action_col, _ = st.columns([1, 2, 7])
        with copy_col:
            with st.popover("复制", use_container_width=False):
                st.code(message.get("content", ""), language=None)
        if is_last:
            with action_col:
                if message.get("error") or message.get("stopped"):
                    if st.button(
                        "重试",
                        key=f"retry_{index}",
                        use_container_width=False,
                    ):
                        retry_last_answer()
                else:
                    if st.button(
                        "重新生成",
                        key=f"regen_{index}",
                        use_container_width=False,
                    ):
                        regenerate_last_answer()


def render_memory_panel() -> None:
    with st.container(height=MEMORY_HEIGHT, border=True, key="memory_panel"):
        title_col, refresh_col, close_col = st.columns(
            [5, 1, 1], vertical_alignment="center"
        )
        with title_col:
            count_label = (
                "离线" if st.session_state.memory_error else f"{len(st.session_state.memories)} 条"
            )
            st.markdown(
                "<div class='mem-head'><h3>记忆</h3>"
                f"<span class='pill' style='height:22px;font-size:.72rem'>{count_label}</span></div>",
                unsafe_allow_html=True,
            )
        with refresh_col:
            if st.button("↻", key="refresh_memory", type="tertiary", help="刷新记忆"):
                try:
                    memories = fetch_memories()
                    st.session_state.memories = memories
                    st.session_state.memory_count = len(memories)
                    st.session_state.memory_error = ""
                    st.rerun()
                except Exception as exc:
                    st.session_state.memory_error = str(exc)
                    st.rerun()
        with close_col:
            if st.button("✕", key="close_memory", type="tertiary", help="关闭记忆抽屉"):
                st.session_state.memory_open = False
                st.rerun()

        st.markdown(
            "<div class='mem-sub'>每 180 秒检查一次，超过 400 tokens 自动整理；"
            "删除后不再参与后续检索。</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.memory_error:
            st.markdown(
                "<div class='inline-error'>记忆服务连接中断，请点右上角 ↻ 重试。</div>",
                unsafe_allow_html=True,
            )
            with st.expander("错误详情", expanded=False):
                st.code(st.session_state.memory_error)
            st.markdown(
                "<div class='mem-foot'>记忆是用户级数据，不随会话删除。</div>",
                unsafe_allow_html=True,
            )
            return

        selected_label = st.segmented_control(
            "记忆类型",
            FILTER_OPTIONS,
            default="全部",
            label_visibility="collapsed",
            key="memory_filter",
            width="content",
        )

        search_query = st.text_input(
            "搜索记忆",
            placeholder="搜索记忆内容…",
            label_visibility="collapsed",
            key="memory_search",
        )

        selected_type = FILTER_TO_TYPE.get(selected_label or "全部", "")
        memories = [
            item
            for item in st.session_state.memories
            if (not selected_type or item.get("memory_type") == selected_type)
            and (
                not search_query
                or search_query.lower() in (item.get("content") or "").lower()
            )
        ]

        if not memories:
            empty_text = (
                "没有匹配的记忆，试试其他关键词。"
                if search_query
                else "还没有这类记忆。<br>当对话超过 token 阈值后，后台任务会自动提取。"
            )
            st.markdown(
                f"<div class='mem-empty'>{empty_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            for memory in memories:
                memory_type = memory.get("memory_type", "summary")
                label, css_class = MEMORY_TYPES.get(memory_type, ("摘要", "sum"))
                importance = memory.get("importance")
                score = f"重要性 {float(importance):.1f}" if isinstance(importance, (int, float)) else "—"
                created = memory.get("created_at")
                try:
                    created_text = datetime.fromtimestamp(int(created)).strftime("%m-%d %H:%M")
                except Exception:
                    created_text = "—"
                accessed = memory.get("last_access_at")
                try:
                    accessed_text = datetime.fromtimestamp(int(accessed)).strftime("%m-%d %H:%M")
                    meta_text = f"创建 {created_text} · 访问 {accessed_text}"
                except Exception:
                    meta_text = f"创建 {created_text}"
                content = html.escape(str(memory.get("content") or ""))
                preview = (
                    content if len(content) <= 120 else content[:120].rstrip() + "…"
                )
                more_html = ""
                if len(content) > 120:
                    more_html = (
                        "<details class='mem-more'><summary>展开全文</summary>"
                        f"<div class='mem-content mem-full'>{content}</div></details>"
                    )
                card_col, action_col = st.columns([9, 1], vertical_alignment="top")
                with card_col:
                    st.markdown(
                        "<div class='mem-card'>"
                        "<div class='mem-top'>"
                        f"<span class='mem-type t-{css_class}'>{label}</span>"
                        f"<span class='mem-score'>{score}</span>"
                        "</div>"
                        f"<div class='mem-content'>{preview}</div>"
                        f"{more_html}"
                        f"<div class='mem-meta'>{meta_text}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                memory_id = memory.get("id")
                confirm_key = f"confirm_mem_{memory_id}"
                with action_col:
                    if st.button(
                        "✕",
                        key=f"del_mem_{memory_id}",
                        help="删除这条记忆",
                        type="tertiary",
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()

                if st.session_state.get(confirm_key):
                    st.markdown(
                        "<div class='mem-confirm'>确认删除这条记忆？</div>",
                        unsafe_allow_html=True,
                    )
                    yes_col, no_col, _ = st.columns([1, 1, 5])
                    with yes_col:
                        if st.button(
                            "确认",
                            key=f"confirm_mem_yes_{memory_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                api(
                                    "DELETE",
                                    f"/memories/{memory_id}",
                                    params={"user_id": USER_ID},
                                    timeout=10,
                                )
                                st.session_state.memories = [
                                    item
                                    for item in st.session_state.memories
                                    if item.get("id") != memory_id
                                ]
                                st.session_state.memory_count = len(
                                    st.session_state.memories
                                )
                                st.session_state[confirm_key] = False
                                st.rerun()
                            except Exception as exc:
                                st.session_state[confirm_key] = False
                                st.error(f"删除失败：{exc}")
                    with no_col:
                        if st.button(
                            "取消",
                            key=f"confirm_mem_no_{memory_id}",
                            use_container_width=True,
                        ):
                            st.session_state[confirm_key] = False
                            st.rerun()

        st.markdown(
            "<div class='mem-foot'>记忆是用户级数据，不随会话删除。</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- sidebar
def render_sidebar() -> None:
    streaming = bool(st.session_state.get("active_stream"))
    with st.sidebar:
        st.markdown(
            "<div class='brand-row'>"
            "<div class='brand-logo'>M</div>"
            "<div class='brand-name'>Mnemo</div>"
            "<div class='brand-ver'>v0.1</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "＋ 新对话",
            type="primary",
            use_container_width=True,
            key="new_chat",
            disabled=streaming,
        ):
            new_conversation()

        search = st.text_input(
            "搜索会话",
            placeholder="搜索会话…",
            label_visibility="collapsed",
            key="conversation_search",
        )

        st.markdown("<div class='side-label'>对话</div>", unsafe_allow_html=True)
        conversations = st.session_state.conversations
        if search:
            conversations = [
                item
                for item in conversations
                if search.lower() in (item.get("title") or "").lower()
            ]

        shown = 0
        for conversation in conversations:
            if shown >= 14:
                break
            shown += 1
            active = conversation["conversation_id"] == st.session_state.conversation_id
            full_title = conversation.get("title") or "新对话"
            title = full_title if len(full_title) <= 13 else full_title[:13] + "…"
            label = ("● " if active else "   ") + title
            if st.button(
                label,
                key=f"conv_{conversation['conversation_id']}",
                use_container_width=True,
                type="primary" if active else "secondary",
                help=full_title if len(full_title) > 13 else None,
                disabled=streaming,
            ):
                select_conversation(conversation)

        if shown == 0:
            st.caption("没有匹配的会话")

        if st.button(
            "🗑 删除当前对话",
            key="delete_chat",
            use_container_width=True,
            disabled=streaming,
        ):
            st.session_state.confirm_delete_chat = True
            st.rerun()

        if st.session_state.get("confirm_delete_chat"):
            st.warning("确认删除当前对话？此操作不可撤销。")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "确认删除",
                    key="delete_chat_confirm",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.confirm_delete_chat = False
                    delete_current_conversation()
            with cancel_col:
                if st.button(
                    "取消",
                    key="delete_chat_cancel",
                    use_container_width=True,
                ):
                    st.session_state.confirm_delete_chat = False
                    st.rerun()

        st.markdown("<div class='side-label'>知识库</div>", unsafe_allow_html=True)
        kbs = st.session_state.kbs
        pending_kb = st.session_state.pop("pending_kb", None)
        if pending_kb is not None:
            st.session_state.kb_widget = (
                pending_kb if pending_kb in kbs else (kbs[0] if kbs else "")
            )
        if kbs:
            if st.session_state.get("kb_widget") not in kbs:
                st.session_state.kb_widget = pick_kb(st.session_state.selected_kb)
            previous_kb = st.session_state.selected_kb
            selected_kb = st.selectbox(
                "知识库",
                kbs,
                key="kb_widget",
                label_visibility="collapsed",
                disabled=streaming,
            )
            st.session_state.selected_kb = selected_kb
            if selected_kb != previous_kb:
                persist_knowledge_base(selected_kb)
            if selected_kb and selected_kb not in st.session_state.kb_files:
                cache_kb_files(selected_kb)
        else:
            selected_kb = ""
            st.session_state.selected_kb = ""
            st.session_state.kb_widget = ""
            st.caption("还没有知识库，请先在下方创建")

        with st.expander("管理知识库", expanded=False):
            new_kb = st.text_input(
                "新知识库名称",
                placeholder="新知识库名称",
                label_visibility="collapsed",
                key="new_kb_name",
            )
            if st.button(
                "创建知识库",
                key="create_kb",
                use_container_width=True,
                disabled=streaming,
            ):
                if new_kb.strip():
                    try:
                        api(
                            "POST",
                            "/knowledge-bases",
                            params={
                                "knowledge_base_id": new_kb.strip(),
                                "user_id": USER_ID,
                            },
                            timeout=10,
                        )
                        st.session_state.kbs = fetch_kbs()
                        st.session_state.selected_kb = new_kb.strip()
                        st.session_state.pending_kb = new_kb.strip()
                        persist_knowledge_base(new_kb.strip())
                        cache_kb_files(new_kb.strip())
                        st.rerun()
                    except Exception as exc:
                        st.error(f"创建失败：{exc}")
                else:
                    st.warning("请输入知识库名称")

            if selected_kb:
                files = st.session_state.kb_files.get(selected_kb)
                if files is None:
                    try:
                        files = fetch_files(selected_kb)
                        st.session_state.kb_files[selected_kb] = files
                    except Exception as exc:
                        st.error(f"读取文件失败：{exc}")
                        files = []
                if files:
                    for index, file in enumerate(files[:12]):
                        file_col, action_col = st.columns([5, 1])
                        status = str(file.get("status") or "")
                        status_label = {
                            "processing": "处理中",
                            "failed": "失败",
                            "ready": "就绪",
                        }.get(status, status)
                        status_html = ""
                        if status and status != "ready":
                            color = "#b45309" if status == "processing" else "#b91c1c"
                            status_html = (
                                f" <span style='color:{color};font-size:.64rem'>"
                                f"[{html.escape(status_label)}]</span>"
                            )
                        with file_col:
                            st.markdown(
                                f"<div style='font-size:.72rem;color:#c5bbad;padding-top:5px'>📄 {html.escape(str(file.get('file_name', ''))[:22])}{status_html}</div>",
                                unsafe_allow_html=True,
                            )
                            if status == "failed":
                                st.caption(
                                    f"处理失败：{html.escape(str(file.get('error') or '未知错误'))[:80]}"
                                )
                        with action_col:
                            if st.button(
                                "🗑",
                                key=f"del_file_{selected_kb}_{index}",
                                help="删除文件",
                            ):
                                try:
                                    api(
                                        "DELETE",
                                        f"/knowledge-bases/{selected_kb}/documents/{file['file_hash']}",
                                        params={"user_id": USER_ID},
                                        timeout=15,
                                    )
                                    st.session_state.kb_files.pop(selected_kb, None)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"删除失败：{exc}")
                else:
                    st.caption("暂无文件")

                uploaded = st.file_uploader(
                    "上传文档",
                    type=["pdf", "docx", "doc"],
                    key=f"upload_{selected_kb}",
                    label_visibility="collapsed",
                )
                if uploaded is not None and st.button(
                    "上传",
                    key=f"upload_btn_{selected_kb}",
                    use_container_width=True,
                    disabled=streaming,
                ):
                    with st.spinner("上传并处理中…"):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/document_upload",
                                data={
                                    "knowledge_base_id": selected_kb,
                                    "user_id": str(USER_ID),
                                },
                                files={
                                    "file": (
                                        uploaded.name,
                                        uploaded.getvalue(),
                                        uploaded.type,
                                    )
                                },
                                timeout=120,
                            )
                            response.raise_for_status()
                            st.session_state.kb_files.pop(selected_kb, None)
                            st.success("已上传，后台处理中")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"上传失败：{exc}")

                confirm = st.checkbox(
                    "确认删除当前知识库", key="confirm_delete_kb"
                )
                if st.button(
                    "删除知识库",
                    key="delete_kb",
                    disabled=not confirm or streaming,
                    use_container_width=True,
                ):
                    try:
                        api(
                            "DELETE",
                            f"/knowledge-bases/{selected_kb}",
                            params={"user_id": USER_ID},
                            timeout=30,
                        )
                        st.session_state.kbs = fetch_kbs()
                        st.session_state.kb_files.pop(selected_kb, None)
                        remaining = st.session_state.kbs
                        fallback_kb = remaining[0] if remaining else ""
                        st.session_state.selected_kb = fallback_kb
                        st.session_state.pending_kb = fallback_kb
                        st.session_state.confirm_delete_kb = False
                        persist_knowledge_base(fallback_kb)
                        if fallback_kb:
                            cache_kb_files(fallback_kb)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

        st.markdown("<div class='side-label'>工作区</div>", unsafe_allow_html=True)
        if st.session_state.memory_error:
            memory_label = "🧠 记忆 · 离线"
        else:
            memory_label = "🧠 记忆"
            if st.session_state.memory_count:
                memory_label += f" · {st.session_state.memory_count}"
        if st.button(memory_label, key="toggle_memory", use_container_width=True):
            st.session_state.memory_open = not st.session_state.memory_open
            st.rerun()

        st.markdown(
            "<div class='side-foot'>© 2026 Mnemo · MIT License</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- page
init_state()
bootstrap()

if st.session_state.boot_error:
    st.error(f"无法连接后端：{st.session_state.boot_error}")
    st.caption(f"后端地址：{API_BASE_URL}")
    if st.button("重试"):
        st.session_state.boot_error = None
        st.session_state.bootstrapped = False
        st.rerun()
    st.stop()

render_sidebar()

if st.session_state.memory_open:
    chat_col, memory_col = st.columns([2.8, 1.2], gap="large")
else:
    chat_col = st.container()
    memory_col = None

with chat_col:
    render_header()
    with st.container(
        height=CHAT_HEIGHT, border=False, key="chat_scroll", autoscroll=True
    ):
        if not st.session_state.messages and not st.session_state.active_stream:
            render_welcome()
        else:
            for index, message in enumerate(st.session_state.messages):
                st.markdown(
                    render_message(
                        message["role"],
                        message["content"],
                        message.get("sources"),
                        message.get("timestamp", ""),
                        sources_from_previous_turn=message.get(
                            "sources_from_previous_turn", False
                        ),
                        anchor_key=f"m{index}",
                        trace=message.get("trace"),
                    ),
                    unsafe_allow_html=True,
                )
                render_message_actions(
                    message,
                    index,
                    index == len(st.session_state.messages) - 1,
                )
            if st.session_state.get("active_stream"):
                render_live_stream()
    render_composer()

if memory_col is not None:
    with memory_col:
        render_memory_panel()

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
MEMORY_TYPE_HELP = {
    "semantic": "稳定事实与偏好",
    "episodic": "具体事件与历史互动",
    "procedural": "可复用的方法与步骤",
    "summary": "一段对话的摘要",
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
  section[data-testid="stSidebar"] [data-testid="stPopover"] button {
    min-height: 30px; padding: 0 6px; border: 1px solid transparent;
    background: transparent !important; color: var(--sidebar-muted) !important;
    box-shadow: none !important; font-size: 1rem; line-height: 1;
  }
  section[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
    background: #26231f !important; color: #fff !important;
  }
  div[data-baseweb="popover"] button[kind="primary"],
  div[data-testid="stPopoverBody"] button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
  }
  div[data-baseweb="popover"] button[kind="primary"]:hover,
  div[data-testid="stPopoverBody"] button[kind="primary"]:hover {
    background: #a94a30 !important;
    border-color: #a94a30 !important;
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
  section[data-testid="stSidebar"] details summary,
  section[data-testid="stSidebar"] details[open] summary {
    background: #24211e !important; color: #d9d2c8 !important;
    border-radius: 7px 7px 0 0;
  }
  section[data-testid="stSidebar"] details:not([open]) summary { border-radius: 7px; }
  section[data-testid="stSidebar"] details summary:hover { background: #2c2824 !important; }
  section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #3a3224 !important; border: 1px solid #5b4a2d !important;
  }
  section[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #f0d7a6 !important; }
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
  /* Conversation labels must truncate, never wrap into a second line. */
  section[data-testid="stSidebar"] [class*="st-key-conv_"] button,
  section[data-testid="stSidebar"] [class*="st-key-conv_"] button * {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    min-width: 0 !important;
  }
  section[data-testid="stSidebar"] [class*="st-key-conv_"] button p {
    display: block !important; width: 100% !important; margin: 0 !important;
  }
  /* A collapsed sidebar must not reserve its 300px width in the main layout. */
  body:has(button[data-testid="stExpandSidebarButton"]) section[data-testid="stSidebar"] {
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    flex-basis: 0 !important;
    overflow: hidden !important;
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
    font-size: .72rem; color: #6c6459; white-space: nowrap; cursor: default;
  }
  .pill.accent { color: var(--accent-dark); border-color: #e8c9bc; background: #fdf0ea; }
  .pill .mini { font-size: .72rem; opacity: .62; }
  .header-detail { position: relative; display: inline-flex; }
  .header-detail-summary {
    display: inline-flex; align-items: center; cursor: pointer;
    list-style: none; outline: none;
  }
  .header-detail-summary::-webkit-details-marker { display: none; }
  .header-detail-summary:hover { background: #faf4eb; }
  .header-detail[open] .header-detail-summary {
    color: var(--accent-dark); border-color: #e8c9bc; background: #fdf0ea;
  }
  .header-detail-panel {
    position: absolute; right: 0; top: calc(100% + 8px);
    width: 292px; max-width: min(292px, calc(100vw - 32px));
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 11px; box-shadow: 0 14px 32px rgba(50, 35, 20, .18);
    padding: 11px 12px 9px; z-index: 1000002;
  }
  .hd-title {
    font-size: .78rem; font-weight: 700; color: var(--ink);
    padding-bottom: 7px; margin-bottom: 7px; border-bottom: 1px solid #eee4d7;
  }
  .hd-row {
    display: grid; grid-template-columns: 68px 1fr; gap: 8px;
    align-items: baseline; padding: 2px 0; font-size: .7rem;
  }
  .hd-row span { color: var(--muted); }
  .hd-row b {
    color: var(--ink); font-weight: 600; text-align: right;
    overflow-wrap: anywhere; line-height: 1.45;
  }
  .hd-foot {
    margin-top: 7px; padding-top: 6px; border-top: 1px dashed #eee4d7;
    color: var(--muted); font-size: .64rem;
  }

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
    white-space: pre-wrap; word-break: break-word;
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
    margin-top: 6px; background: #fffdf9; border: 1px solid #e3dbce;
    border-radius: 9px; max-width: 560px; overflow: hidden;
    scroll-margin-top: 14px;
  }
  .source-preview {
    margin-top: 2px; font-size: .7rem; line-height: 1.35; color: #6d6458;
    display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .source-card[open] .source-preview { display: none; }
  .source-summary {
    display: flex; gap: 7px; align-items: flex-start; padding: 7px 9px;
    cursor: pointer; list-style: none; outline: none;
  }
  .source-summary::-webkit-details-marker { display: none; }
  .source-summary:hover { background: #faf4eb; }
  .source-head { min-width: 0; flex: 1; }
  .file-icon {
    width: 26px; height: 30px; border-radius: 5px; background: #f4e6de;
    color: #9e4c32; display: grid; place-items: center; flex: 0 0 auto; font-size: .6rem;
  }
  .source-name { font-size: .74rem; font-weight: 700; color: var(--ink); }
  .source-locator {
    margin-left: 5px; font-size: .58rem; font-weight: 400; color: var(--muted);
    background: #f4eee5; border-radius: 4px; padding: 1px 4px; white-space: nowrap;
  }
  .source-meta { font-size: .68rem; color: var(--muted); margin-top: 1px; }
  .source-toggle {
    margin-left: auto; color: var(--accent-dark); font-size: .68rem; white-space: nowrap;
  }
  .source-toggle::before { content: "展开"; }
  .source-card[open] .source-toggle::before { content: "收起"; }
  .source-note { margin-top: 8px; font-size: .72rem; color: #6f665b; }
  .source-snippet {
    font-size: .78rem; color: #6d6458; line-height: 1.55;
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
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    background: var(--accent) !important; color: #fff !important; border: 0 !important;
    border-radius: 8px !important; font-weight: 700 !important;
  }
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
    background: #a94a30 !important;
  }
  div[data-testid="stForm"] div[data-testid="stElementContainer"]:has(
    div[data-testid="stFormSubmitButton"]
  ) { width: auto !important; margin-left: auto !important; }
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    width: 38px !important; height: 38px !important; min-height: 38px !important;
    display: block !important; padding: 0 !important; border-radius: 10px !important;
  }
  div.st-key-composer_shell {
    background: var(--panel); border: 1px solid #ddd4c7; border-radius: 14px;
    padding: 8px 12px 6px; box-shadow: 0 6px 22px rgba(60,45,30,.06);
  }
  div.st-key-composer_shell div[data-testid="stForm"] {
    background: transparent !important; border: 0 !important;
    box-shadow: none !important; padding: 0 !important;
  }
  .composer-pill { font-size: .72rem; color: #6f665b; padding-top: 2px; }
  .hintline { text-align: center; font-size: .68rem; color: #6f665b; margin-top: 3px; }

  /* chat scroll area */
  div.st-key-chat_scroll {
    height: calc(100vh - 380px) !important;
    height: calc(100dvh - 380px) !important;
    min-height: 220px;
    flex: 0 0 auto !important;
  }
  div[data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll) {
    height: calc(100vh - 380px) !important;
    height: calc(100dvh - 380px) !important;
    min-height: 220px !important;
    flex: 0 0 calc(100vh - 380px) !important;
  }

  /* Streamlit makes columns full-width below 640px. Keep the small control
     rows horizontal so menus and message actions stay attached to their item. */
  div[class*="st-key-conv_row_"] [data-testid="stHorizontalBlock"],
  div[class*="st-key-mem_row_"] [data-testid="stHorizontalBlock"],
  div[class*="st-key-file_row_"] [data-testid="stHorizontalBlock"],
  div[class*="st-key-msg_actions_"] [data-testid="stHorizontalBlock"],
  div[class*="st-key-delete_chat_confirm_"] [data-testid="stHorizontalBlock"] {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: .35rem !important;
  }
  div[class*="st-key-conv_row_"] [data-testid="stColumn"],
  div[class*="st-key-mem_row_"] [data-testid="stColumn"],
  div[class*="st-key-file_row_"] [data-testid="stColumn"],
  div[class*="st-key-msg_actions_"] [data-testid="stColumn"],
  div[class*="st-key-delete_chat_confirm_"] [data-testid="stColumn"] {
    min-width: 0 !important;
    width: auto !important;
  }
  div[class*="st-key-conv_row_"] [data-testid="stColumn"]:first-child,
  div[class*="st-key-mem_row_"] [data-testid="stColumn"]:first-child,
  div[class*="st-key-file_row_"] [data-testid="stColumn"]:first-child {
    flex: 1 1 auto !important;
  }
  div[class*="st-key-conv_row_"] [data-testid="stColumn"]:last-child,
  div[class*="st-key-mem_row_"] [data-testid="stColumn"]:last-child,
  div[class*="st-key-file_row_"] [data-testid="stColumn"]:last-child {
    flex: 0 0 48px !important;
    width: 48px !important;
    max-width: 48px !important;
  }
  div[class*="st-key-msg_actions_"] [data-testid="stColumn"]:nth-child(1) {
    flex: 0 0 64px !important; width: 64px !important; max-width: 64px !important;
  }
  div[class*="st-key-msg_actions_"] [data-testid="stColumn"]:nth-child(2) {
    flex: 0 0 auto !important; width: auto !important;
  }
  div[class*="st-key-msg_actions_"] [data-testid="stColumn"]:nth-child(3) {
    flex: 1 1 auto !important;
  }
  div[class*="st-key-delete_chat_confirm_"] [data-testid="stColumn"] {
    flex: 1 1 0 !important;
  }

  @media (max-width: 640px) {
    div.st-key-chat_scroll {
      height: calc(100dvh - 415px) !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll) {
      height: calc(100dvh - 415px) !important;
      flex: 0 0 calc(100dvh - 415px) !important;
    }
    .hintline { margin-top: 1px; font-size: .64rem; }
    section[data-testid="stSidebar"] {
      width: 100vw !important;
      min-width: 100vw !important;
      max-width: 100vw !important;
      box-shadow: 18px 0 42px rgba(20, 14, 8, .32);
    }
    div[class*="st-key-file_row_"] [data-testid="stMarkdownContainer"] {
      min-width: 0 !important; overflow: hidden !important;
    }
    div[class*="st-key-file_row_"] [data-testid="stMarkdownContainer"] > div,
    div[class*="st-key-file_row_"] [data-testid="stMarkdownContainer"] p {
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
    }
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
  .mem-sub { font-size: .68rem; color: var(--muted); line-height: 1.45; margin: 3px 0 8px; }
  div.st-key-memory_header {
    position: sticky; top: 0; z-index: 3; background: #fffcf7;
    padding-bottom: 2px;
  }
  div.st-key-memory_header [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
  }
  div.st-key-memory_header [data-testid="stColumn"] {
    min-width: 0 !important;
  }
  .mem-card {
    border: 1px solid #e7dfd3; background: #fff; border-radius: 9px;
    padding: 9px 10px; margin: 0 0 7px;
  }
  .mem-top { display: flex; align-items: center; gap: 5px; margin-bottom: 4px; }
  .mem-type {
    font-size: .64rem; letter-spacing: .04em; text-transform: uppercase;
    border-radius: 3px; padding: 1px 5px; font-weight: 700;
  }
  .t-sem { color: #0f6b60; background: #e4f2ee; }
  .t-epi { color: #8a5a22; background: #f7ecd9; }
  .t-proc { color: #5a4a8a; background: #eeeafa; }
  .t-sum { color: #6d6255; background: #f0ebe3; }
  .mem-score { margin-left: auto; font-size: .64rem; color: #6f665b; }
  .mem-content { font-size: .78rem; line-height: 1.5; color: #3a332c; }
  .mem-meta { display: flex; gap: 5px; margin-top: 4px; color: #6f665b; font-size: .68rem; }
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

  @media (min-width: 900px) {
    section[data-testid="stSidebar"] {
      width: 300px !important; min-width: 300px !important; max-width: 300px !important;
      flex: 0 0 300px !important;
    }
  }
  @media (max-width: 1100px) {
    .main-header .pills .pill:not(.accent) { display: none; }
    .msg-body { max-width: 100%; }

    /* Narrow screens: keep the chat in place and turn the memory column into
       a fixed right drawer with a scrim, instead of replacing the chat. */
    div[data-testid="stHorizontalBlock"]:has(.st-key-memory_panel) > div[data-testid="stColumn"]:first-child {
      display: block !important;
      flex: 1 1 100% !important;
      width: 100% !important;
      min-width: 0 !important;
    }
    div[data-testid="stColumn"]:has(.st-key-memory_panel) {
      position: fixed !important;
      top: 0; right: 0; bottom: 0;
      width: min(92vw, 420px) !important;
      max-width: 420px !important;
      min-width: 0 !important;
      height: 100dvh !important;
      z-index: 1000000;
      padding: 0 !important;
      margin: 0 !important;
      background: transparent !important;
    }
    div[data-testid="stColumn"]:has(.st-key-memory_panel)::before {
      content: "";
      position: fixed;
      inset: 0;
      background: rgba(28, 26, 24, .34);
      z-index: -1;
      pointer-events: auto;
    }
    div[data-testid="stColumn"]:has(.st-key-memory_panel) > div {
      height: 100% !important;
    }
    div.st-key-memory_panel {
      height: 100dvh !important;
      max-height: 100dvh !important;
      min-height: 0 !important;
      border-radius: 0 !important;
      border-right: 0 !important;
      border-top: 0 !important;
      border-bottom: 0 !important;
      box-shadow: -18px 0 50px rgba(40, 28, 18, .22);
    }
    div[data-testid="stLayoutWrapper"]:has(> .st-key-memory_panel) {
      height: 100dvh !important;
      flex: 0 0 100dvh !important;
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
    "runtime": {},
    "active_stream": None,
    "retrieval_mode": "fast",
    "history_loaded_for": None,
    "confirm_delete_chat": False,
    "show_all_conversations": False,
    "files_expanded": {},
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


def _relative_time(value: Any) -> str:
    """Human-friendly timestamp for memory cards; 0 means never accessed."""
    try:
        stamp = int(value or 0)
    except (TypeError, ValueError):
        return "未访问"
    if stamp <= 0:
        return "未访问"
    delta = max(0, int(datetime.now().timestamp()) - stamp)
    if delta < 60:
        return "刚刚"
    if delta < 3600:
        return f"{delta // 60} 分钟前"
    if delta < 86400:
        return f"{delta // 3600} 小时前"
    if delta < 7 * 86400:
        return f"{delta // 86400} 天前"
    return datetime.fromtimestamp(stamp).strftime("%m-%d")


def _copy_text(text: str) -> str:
    """Strip the Markdown decoration that st.code would otherwise show raw."""
    cleaned = str(text or "")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.M)
    return cleaned.strip()


def _importance_label(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    if value >= 0.85:
        return "高"
    if value >= 0.7:
        return "中"
    return "低"


def _relevance_label(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "引用片段"
    if value >= 0.85:
        return "高相关"
    if value >= 0.75:
        return "中相关"
    return "参考"


def _index_status(files: List[Dict[str, Any]]) -> tuple[str, str]:
    total = len(files or [])
    if total == 0:
        return "未索引", "#8a8074"
    statuses = [str(item.get("status") or "").lower() for item in files]
    failed = statuses.count("failed")
    processing = statuses.count("processing")
    if failed:
        return f"索引异常 · {failed} 个文件失败", "#b91c1c"
    if processing:
        return f"索引中 · {processing}/{total}", "#b45309"
    return f"索引就绪 · {total} 个文件", "#5f8b62"


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
                "trace": item.get("trace") or {},
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


def fetch_runtime() -> Dict[str, Any]:
    try:
        return api("GET", "/runtime", timeout=5)
    except Exception:
        return {}


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


def fetch_memories_payload() -> tuple[List[Dict[str, Any]], int]:
    data = api(
        "GET", "/memories", params={"user_id": USER_ID, "limit": 200}, timeout=15
    )
    memories = data.get("memories", [])
    total = data.get("total")
    if not isinstance(total, int):
        total = len(memories)
    return memories, total


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
            memories, memory_total = fetch_memories_payload()
            st.session_state.memories = memories
            st.session_state.memory_count = memory_total
            st.session_state.memory_error = ""
        except Exception as exc:
            st.session_state.memories = []
            st.session_state.memory_count = 0
            st.session_state.memory_error = str(exc)

        st.session_state.runtime = fetch_runtime()

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
def render_sources_html(
    sources: Optional[List[Dict[str, Any]]],
    anchor_key: str = "msg",
    sources_from_previous_turn: bool = False,
) -> str:
    """Render the compact evidence list separately from the answer body.

    Keeping sources out of ``render_message`` lets the message actions sit
    directly under the answer instead of below a wall of evidence cards.
    """

    source_html = ""
    if sources_from_previous_turn and sources:
        source_html += (
            "<div class='source-note'>本轮未重新检索，沿用上一轮来源</div>"
        )
    for source in sources or []:
        index = html.escape(str(source.get("index", "")))
        raw_name = str(source.get("file_name", "未知文件"))
        file_name = html.escape(raw_name)
        extension = raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else ""
        icon = {"pdf": "PDF", "docx": "DOC", "doc": "DOC", "txt": "TXT"}.get(
            extension, "FILE"
        )
        quote = str(source.get("quote") or source.get("snippet") or "").strip()
        snippet = str(source.get("snippet") or "").strip()
        # Old messages may only have a raw snippet with DOCX line breaks.
        body_text = re.sub(r"\s+", " ", (quote or snippet)).strip()
        preview = body_text[:90].rstrip() + ("…" if len(body_text) > 90 else "")
        locator = str(source.get("locator") or "").strip()
        locator_html = (
            f"<span class='source-locator'>{html.escape(locator)}</span>"
            if locator
            else ""
        )
        score = source.get("score")
        meta = _relevance_label(score)
        score_title = (
            f" title='相关度 {score:.3f}'" if isinstance(score, (int, float)) else ""
        )
        source_id = f"source-{html.escape(anchor_key or 'msg')}-{index}"
        source_html += (
            f"<details class='source-card' id='{source_id}'>"
            "<summary class='source-summary'>"
            f"<div class='file-icon'>{icon}</div>"
            "<div class='source-head'>"
            f"<div class='source-name'>[{index}] {file_name}{locator_html}</div>"
            f"<div class='source-meta'{score_title}>{meta}</div>"
            f"<div class='source-preview'>{html.escape(preview)}</div>"
            "</div>"
            "<span class='source-toggle'></span>"
            "</summary>"
            f"<div class='source-snippet'>{html.escape(body_text)}</div>"
            "</details>"
        )
    return source_html


def render_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    timestamp: str = "",
    streaming: bool = False,
    sources_from_previous_turn: bool = False,
    anchor_key: str = "msg",
    trace: Optional[Dict[str, Any]] = None,
    show_sources: bool = True,
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
    if isinstance(trace, dict):
        parts = []
        mode = trace.get("mode")
        if mode:
            mode_label = {"fast": "快速", "deep": "深入", "auto": "自动"}.get(
                str(mode), str(mode)
            )
            parts.append(f"本次：{mode_label}")
        if trace.get("rounds") is not None:
            parts.append(f"{trace.get('rounds')} 轮")
        if trace.get("parent_count") is not None:
            parts.append(f"{trace.get('parent_count')} 条召回")
        if trace.get("elapsed_ms") is not None:
            try:
                parts.append(f"{float(trace.get('elapsed_ms')) / 1000:.1f}s")
            except (TypeError, ValueError):
                pass
        if trace.get("escalated"):
            parts.append("无结果已升级深入")
        if trace.get("degraded"):
            parts.append("Agentic 关闭已降级")
        memory_count = trace.get("memory_count")
        if isinstance(memory_count, int) and memory_count > 0:
            parts.append(f"使用 {memory_count} 条记忆")
        if parts:
            trace_html = f"<div class='source-note'>{' · '.join(parts)}</div>"

    source_html = (
        render_sources_html(sources, anchor_key, sources_from_previous_turn)
        if show_sources
        else ""
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
        "《紫罗兰永恒花园》的制作公司是谁？",
        "《紫罗兰永恒花园》原作小说一共有几卷？",
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
    files = st.session_state.kb_files.get(st.session_state.selected_kb, []) or []
    status_text, dot_color = _index_status(files)
    status_line = f"{kb} · {status_text}" if st.session_state.selected_kb else kb
    memory_status = (
        "记忆 · 离线"
        if st.session_state.memory_error
        else f"记忆 · {st.session_state.memory_count} 条"
    )
    runtime = st.session_state.get("runtime") or {}
    runtime_models = runtime.get("models") or {}
    chunk_config = runtime.get("chunk_config") or {}
    runtime_git = runtime.get("git") or {}
    mode_label = {"fast": "快速", "auto": "自动", "deep": "深入"}.get(
        str(st.session_state.get("retrieval_mode") or "fast"), "快速"
    )
    detail_rows = [
        ("知识库", kb),
        ("文件", f"{len(files)} 个"),
        ("记忆", "离线" if st.session_state.memory_error else f"{st.session_state.memory_count} 条"),
        ("检索模式", mode_label),
        ("生成模型", html.escape(str(runtime_models.get("agent") or "—"))),
        ("重排模型", html.escape(str(runtime_models.get("rerank") or "—"))),
        ("向量模型", html.escape(str(runtime_models.get("embedding") or "—"))),
        (
            "分块",
            f"{chunk_config.get('parent_size', '—')} / {chunk_config.get('child_size', '—')}",
        ),
    ]
    detail_html = "".join(
        f"<div class='hd-row'><span>{label}</span><b>{value}</b></div>"
        for label, value in detail_rows
    )
    git_sha = str(runtime_git.get("sha") or "").strip()
    git_note = ""
    if git_sha:
        git_note = (
            f"<div class='hd-foot'>代码 {html.escape(git_sha)}"
            f"{' · 有未提交改动' if runtime_git.get('dirty') else ''}</div>"
        )
    st.markdown(
        "<div class='main-header'>"
        f"<div><h1>{title}</h1>"
        f"<div class='sub'><span class='dot' style='background:{dot_color}'></span>"
        f"{status_line}</div></div>"
        "<div class='pills'>"
        f"<span class='pill accent' title='记忆面板入口在左侧工作区'>{memory_status}</span>"
        "<details class='header-detail'>"
        "<summary class='pill header-detail-summary'>详情</summary>"
        "<div class='header-detail-panel'>"
        "<div class='hd-title'>运行时详情</div>"
        f"{detail_html}"
        f"{git_note}"
        "</div></details>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_composer() -> None:
    streaming = bool(st.session_state.get("active_stream"))
    with st.container(key="composer_shell"):
        toolbar_left, toolbar_right = st.columns([1.2, 3.8], vertical_alignment="center")
        with toolbar_left:
            st.markdown(
                f"<div class='composer-pill'>知识库 · "
                f"{html.escape(st.session_state.selected_kb or '未选择')}</div>",
                unsafe_allow_html=True,
            )
        with toolbar_right:
            st.segmented_control(
                "检索模式",
                options=["fast", "auto", "deep"],
                format_func=lambda m: {"auto": "自动", "fast": "快速", "deep": "深入"}[m],
                key="retrieval_mode",
                label_visibility="collapsed",
                disabled=streaming,
            )

        with st.form("composer", clear_on_submit=True):
            input_col, send_col = st.columns([6, 1], vertical_alignment="bottom")
            with input_col:
                prompt = st.text_area(
                    "消息",
                    height=54,
                    placeholder="继续追问，或让我总结一份文档…",
                    label_visibility="collapsed",
                    disabled=streaming,
                )
            with send_col:
                submitted = st.form_submit_button(
                    "↑", help="发送", use_container_width=False, disabled=streaming
                )
        st.markdown(
            "<div class='hintline'>Ctrl + Enter 发送 · Enter 换行 · 回答会标注来源</div>",
            unsafe_allow_html=True,
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
                streaming=True,
                anchor_key="live",
                trace=snap.get("trace"),
                show_sources=False,
            ),
            unsafe_allow_html=True,
        )
        if snap["sources"]:
            st.markdown(
                render_sources_html(snap["sources"], "live"),
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
            with st.popover("复制", use_container_width=False, key=f"copy_{index}"):
                st.code(_copy_text(message.get("content", "")), language=None)
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
        with st.container(key="memory_header"):
            title_col, refresh_col, close_col = st.columns(
                [5, 1, 1], vertical_alignment="center"
            )
            with title_col:
                count_label = (
                    "离线"
                    if st.session_state.memory_error
                    else f"{st.session_state.memory_count} 条"
                )
                st.markdown(
                    "<div class='mem-head'><h3>记忆</h3>"
                    f"<span class='pill' style='height:22px;font-size:.72rem'>{count_label}</span></div>",
                    unsafe_allow_html=True,
                )
            with refresh_col:
                if st.button("↻", key="refresh_memory", type="tertiary", help="刷新记忆"):
                    try:
                        memories, memory_total = fetch_memories_payload()
                        st.session_state.memories = memories
                        st.session_state.memory_count = memory_total
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
            "<div class='mem-sub'>对话足够长后自动整理；删除后不再参与后续检索。</div>",
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
                type_help = MEMORY_TYPE_HELP.get(memory_type, "")
                importance = memory.get("importance")
                score = _importance_label(importance)
                score_title = (
                    f" title='重要性 {float(importance):.2f}'"
                    if isinstance(importance, (int, float))
                    else ""
                )
                created = memory.get("created_at")
                try:
                    created_text = datetime.fromtimestamp(int(created)).strftime("%m-%d")
                except Exception:
                    created_text = "—"
                accessed_text = _relative_time(memory.get("last_access_at"))
                meta_text = (
                    f"未访问 · 创建 {created_text}"
                    if accessed_text == "未访问"
                    else f"{accessed_text} 使用 · 创建 {created_text}"
                )
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
                memory_id = memory.get("id")
                with st.container(key=f"mem_row_{memory_id}"):
                    card_col, action_col = st.columns(
                        [9, 1], vertical_alignment="top"
                    )
                    with card_col:
                        st.markdown(
                            "<div class='mem-card'>"
                            "<div class='mem-top'>"
                            f"<span class='mem-type t-{css_class}' title='{type_help}'>{label}</span>"
                            f"<span class='mem-score'{score_title}>{score}</span>"
                            "</div>"
                            f"<div class='mem-content'>{preview}</div>"
                            f"{more_html}"
                            f"<div class='mem-meta'>{meta_text}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    with action_col:
                        with st.popover(
                            "⋯", help="更多操作", key=f"mem_menu_{memory_id}"
                        ):
                            st.caption("删除后不再参与检索。")
                            if st.button(
                                "删除这条记忆",
                                key=f"del_mem_{memory_id}",
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
                                    st.session_state.memory_count = max(
                                        0,
                                        int(st.session_state.memory_count or 0) - 1,
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"删除失败：{exc}")

        st.markdown(
            "<div class='mem-foot'>记忆是用户级数据，不随会话删除。</div>",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=4)
def render_kb_processing_watch(kb_id: str) -> None:
    """Poll a processing KB and rerun the app when it becomes ready."""
    files = st.session_state.kb_files.get(kb_id) or []
    if not any(
        str(item.get("status") or "").lower() == "processing" for item in files
    ):
        return
    st.caption("后台索引中，页面会自动刷新…")
    try:
        fresh = fetch_files(kb_id)
    except Exception:
        return
    st.session_state.kb_files[kb_id] = fresh
    if not any(
        str(item.get("status") or "").lower() == "processing" for item in fresh
    ):
        st.rerun(scope="app")


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

        show_all = bool(st.session_state.get("show_all_conversations"))
        visible_conversations = (
            conversations if (search or show_all) else conversations[:14]
        )
        for conversation in visible_conversations:
            conversation_id = conversation["conversation_id"]
            active = conversation_id == st.session_state.conversation_id
            full_title = conversation.get("title") or "新对话"
            label = ("● " if active else "") + full_title
            with st.container(key=f"conv_row_{conversation_id}"):
                row_main, row_menu = st.columns(
                    [6, 1], vertical_alignment="center"
                )
                with row_main:
                    if st.button(
                        label,
                        key=f"conv_{conversation_id}",
                        use_container_width=True,
                        type="primary" if active else "secondary",
                        help=full_title,
                        disabled=streaming,
                    ):
                        select_conversation(conversation)
                with row_menu:
                    with st.popover(
                        "⋯",
                        key=f"conv_menu_{conversation_id}",
                        help="重命名会话",
                    ):
                        new_title = st.text_input(
                            "会话标题",
                            value=full_title,
                            key=f"rename_input_{conversation_id}",
                            label_visibility="collapsed",
                        )
                        if st.button(
                            "保存标题",
                            key=f"rename_save_{conversation_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                api(
                                    "PATCH",
                                    f"/conversations/{conversation_id}",
                                    params={"user_id": USER_ID, "title": new_title},
                                    timeout=10,
                                )
                                refresh_conversations()
                                st.rerun()
                            except Exception as exc:
                                st.error(f"重命名失败：{exc}")

        if not visible_conversations:
            st.caption("没有匹配的会话")
        elif len(conversations) > 14 and not search:
            if st.button(
                "收起会话" if show_all else f"查看全部 {len(conversations)} 条",
                key="toggle_conversations",
                use_container_width=True,
                disabled=streaming,
            ):
                st.session_state.show_all_conversations = not show_all
                st.rerun()

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
            with st.container(key="delete_chat_confirm_row"):
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

        cached_files = st.session_state.kb_files.get(selected_kb, []) or []
        cached_statuses = [str(item.get("status") or "").lower() for item in cached_files]
        processing_count = cached_statuses.count("processing")
        failed_count = cached_statuses.count("failed")
        if failed_count:
            kb_expander_label = f"管理知识库 · {failed_count} 个失败"
        elif processing_count:
            kb_expander_label = f"管理知识库 · 索引中 {processing_count}/{len(cached_files)}"
        else:
            kb_expander_label = "管理知识库"

        with st.expander(kb_expander_label, expanded=False):
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
                    expanded_map = st.session_state.get("files_expanded", {}) or {}
                    show_all_files = bool(expanded_map.get(selected_kb))
                    if len(files) > 12:
                        if st.button(
                            "收起文件" if show_all_files else f"查看全部 {len(files)} 个文件",
                            key=f"toggle_files_{selected_kb}",
                            use_container_width=True,
                        ):
                            expanded_map = dict(expanded_map)
                            expanded_map[selected_kb] = not show_all_files
                            st.session_state.files_expanded = expanded_map
                            st.rerun()
                        st.caption(
                            f"显示 {len(files) if show_all_files else 12}/{len(files)} 个文件"
                        )
                    visible_files = files if show_all_files else files[:12]
                    for index, file in enumerate(visible_files):
                        file_hash = str(file.get("file_hash") or "")
                        with st.container(key=f"file_row_{selected_kb}_{file_hash}"):
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
                                with st.popover(
                                    "🗑",
                                    help="删除文件",
                                    key=f"file_menu_{selected_kb}_{file_hash}",
                                ):
                                    st.caption(
                                        f"确认删除 {str(file.get('file_name', ''))[:32]}？"
                                        "删除后不可恢复。"
                                    )
                                    if st.button(
                                        "删除文件",
                                        key=f"del_file_{selected_kb}_{file_hash}",
                                        type="primary",
                                        use_container_width=True,
                                    ):
                                        try:
                                            api(
                                                "DELETE",
                                                f"/knowledge-bases/{selected_kb}/documents/{file['file_hash']}",
                                                params={"user_id": USER_ID},
                                                timeout=15,
                                            )
                                            st.session_state.kb_files.pop(selected_kb, None)
                                            cache_kb_files(selected_kb)
                                            st.rerun()
                                        except Exception as exc:
                                            st.error(f"删除失败：{exc}")
                                            st.session_state.kb_files.pop(
                                                selected_kb, None
                                            )
                                            cache_kb_files(selected_kb)
                    processing = any(
                        str(item.get("status") or "").lower() == "processing"
                        for item in files
                    )
                    if processing:
                        render_kb_processing_watch(selected_kb)
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
                            cache_kb_files(selected_kb)
                            st.success("已上传，后台处理中")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"上传失败：{exc}")

                confirm = st.checkbox(
                    "确认删除当前知识库",
                    key=f"confirm_delete_kb_{selected_kb}",
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
                        persist_knowledge_base(fallback_kb)
                        if fallback_kb:
                            cache_kb_files(fallback_kb)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")
                        # PostgreSQL 与 Milvus 是两步删除；如果第一步已成功，
                        # 重新拉取列表可以避免界面继续显示已删除的知识库。
                        try:
                            st.session_state.kbs = fetch_kbs()
                            if selected_kb not in st.session_state.kbs:
                                st.session_state.kb_files.pop(selected_kb, None)
                                fallback = (
                                    st.session_state.kbs[0]
                                    if st.session_state.kbs
                                    else ""
                                )
                                st.session_state.selected_kb = fallback
                                st.session_state.pending_kb = fallback
                        except Exception:
                            pass

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
                        timestamp=message.get("timestamp", ""),
                        anchor_key=f"m{index}",
                        trace=message.get("trace"),
                        show_sources=False,
                    ),
                    unsafe_allow_html=True,
                )
                render_message_actions(
                    message,
                    index,
                    index == len(st.session_state.messages) - 1,
                )
                if message.get("sources"):
                    st.markdown(
                        render_sources_html(
                            message.get("sources"),
                            f"m{index}",
                            message.get("sources_from_previous_turn", False),
                        ),
                        unsafe_allow_html=True,
                    )
            if st.session_state.get("active_stream"):
                render_live_stream()
    render_composer()

if memory_col is not None:
    with memory_col:
        render_memory_panel()

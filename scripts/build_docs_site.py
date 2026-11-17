"""Build the GitHub Pages version of docs/project-introduction.md.

The Markdown file stays the source of truth.  GitHub Actions runs this script
and publishes ``docs/index.html`` through GitHub Pages.
"""

from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "project-introduction.md"
OUTPUT = ROOT / "docs" / "index.html"

REPO_URL = "https://github.com/wh-violet667/Mnemo"
PAGES_URL = "https://wh-violet667.github.io/Mnemo/"
MARKDOWN_URL = f"{REPO_URL}/blob/main/docs/project-introduction.md"


def build_nav(toc_tokens) -> str:
    links = []
    for token in toc_tokens:
        if token.get("level") != 2:
            continue
        name = re.sub(r"<[^>]+>", "", token.get("name") or "").strip()
        anchor = html.escape(str(token.get("id") or ""), quote=True)
        if name and anchor:
            links.append(f'<a href="#{anchor}">{html.escape(name)}</a>')
    return "\n".join(links)


def build_page(body_markdown: str) -> str:
    parser = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"permalink": False}},
    )
    content = parser.convert(body_markdown)
    # Make wide result tables scroll on small screens without breaking layout.
    content = content.replace("<table>", '<div class="table-wrap"><table>')
    content = content.replace("</table>", "</table></div>")
    nav = build_nav(parser.toc_tokens)

    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mnemo · 项目介绍与答辩问答</title>
  <meta name="description" content="Mnemo：基于知识库与长期记忆的个性化 RAG 助手。项目架构、核心组件、评测结果与老师追问回答。">
  <meta property="og:title" content="Mnemo · 项目介绍与答辩问答">
  <meta property="og:description" content="父子块检索、LangGraph Agent、PostgreSQL 长期记忆、证据级评测与分块实验。">
  <meta property="og:type" content="article">
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-2: #f1f5f9;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --accent: #6d28d9;
      --accent-2: #0ea5e9;
      --accent-soft: #ede9fe;
      --code-bg: #0f172a;
      --code-text: #e2e8f0;
      --hero-a: #0f172a;
      --hero-b: #312e81;
      --hero-c: #6d28d9;
      --shadow: 0 12px 40px rgba(15, 23, 42, .08);
      --radius: 16px;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #070b17;
        --surface: #0d1425;
        --surface-2: #131c30;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --border: #253047;
        --accent: #a78bfa;
        --accent-2: #38bdf8;
        --accent-soft: #21183f;
        --code-bg: #020617;
        --code-text: #dbeafe;
        --shadow: 0 16px 50px rgba(0, 0, 0, .35);
      }
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      line-height: 1.78;
      -webkit-font-smoothing: antialiased;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .hero {
      position: relative;
      overflow: hidden;
      padding: 64px 24px 46px;
      color: #fff;
      background:
        radial-gradient(circle at 82% 18%, rgba(56, 189, 248, .28), transparent 30%),
        radial-gradient(circle at 10% 80%, rgba(167, 139, 250, .28), transparent 34%),
        linear-gradient(135deg, var(--hero-a), var(--hero-b) 55%, var(--hero-c));
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: 0;
      background-image: linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: linear-gradient(to bottom, black, transparent 80%);
      pointer-events: none;
    }
    .hero-inner { position: relative; z-index: 1; max-width: 1180px; margin: 0 auto; }
    .kicker {
      display: inline-flex; gap: 8px; align-items: center;
      padding: 6px 12px; border: 1px solid rgba(255,255,255,.24);
      border-radius: 999px; font-size: .82rem; letter-spacing: .08em;
      text-transform: uppercase; color: rgba(255,255,255,.82);
      background: rgba(255,255,255,.06);
    }
    h1 {
      max-width: 900px; margin: 18px 0 14px;
      font-size: clamp(2.2rem, 6vw, 4rem); line-height: 1.1;
      letter-spacing: -.04em;
    }
    .lead { max-width: 850px; margin: 0; font-size: clamp(1rem, 2vw, 1.18rem); color: rgba(255,255,255,.82); }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0 0; }
    .badges span {
      padding: 5px 10px; border-radius: 999px; font-size: .78rem;
      border: 1px solid rgba(255,255,255,.2);
      background: rgba(255,255,255,.08); color: rgba(255,255,255,.9);
    }
    .stats {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px; max-width: 760px; margin: 30px 0 0;
    }
    .stat {
      padding: 14px 16px; border-radius: 14px;
      border: 1px solid rgba(255,255,255,.16);
      background: rgba(15, 23, 42, .28); backdrop-filter: blur(8px);
    }
    .stat strong { display: block; font-size: 1.55rem; line-height: 1.2; }
    .stat span { display: block; margin-top: 2px; font-size: .78rem; color: rgba(255,255,255,.65); }
    .hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 26px; }
    .button {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 10px 16px; border-radius: 10px; font-weight: 600;
      border: 1px solid rgba(255,255,255,.18); color: #fff;
      background: rgba(255,255,255,.1);
    }
    .button.primary { color: #1e1b4b; background: #fff; border-color: #fff; }
    .button:hover { text-decoration: none; transform: translateY(-1px); }
    .layout {
      display: grid; grid-template-columns: 250px minmax(0, 1fr);
      gap: 40px; max-width: 1180px; margin: 0 auto; padding: 36px 24px 72px;
    }
    .sidebar { position: sticky; top: 22px; align-self: start; max-height: calc(100vh - 44px); overflow: auto; }
    .side-title { margin: 0 0 10px; font-size: .75rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }
    .sidebar nav { display: grid; gap: 2px; }
    .sidebar nav a {
      display: block; padding: 7px 10px; border-radius: 8px;
      color: var(--muted); font-size: .88rem; line-height: 1.45;
    }
    .sidebar nav a:hover { color: var(--text); background: var(--surface-2); text-decoration: none; }
    .sidebar nav a.active { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
    .side-links { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border); display: grid; gap: 8px; }
    .side-links a { font-size: .84rem; color: var(--muted); }
    .content { min-width: 0; }
    .content h2 {
      margin: 48px 0 18px; padding-top: 10px; border-top: 1px solid var(--border);
      font-size: clamp(1.5rem, 3vw, 2.05rem); line-height: 1.3; letter-spacing: -.02em;
    }
    .content h2:first-child { margin-top: 0; border-top: 0; padding-top: 0; }
    .content h3 { margin: 30px 0 12px; font-size: 1.2rem; line-height: 1.4; }
    .content p { margin: 0 0 14px; }
    .content ul, .content ol { padding-left: 1.35rem; margin: 0 0 16px; }
    .content li { margin: 4px 0; }
    .content blockquote {
      margin: 18px 0; padding: 12px 16px; border-left: 4px solid var(--accent);
      background: var(--accent-soft); border-radius: 0 10px 10px 0; color: var(--text);
    }
    .content blockquote p { margin: 0; }
    .content hr { border: 0; border-top: 1px solid var(--border); margin: 34px 0; }
    .table-wrap { overflow-x: auto; margin: 18px 0; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
    .content table { width: 100%; border-collapse: collapse; font-size: .9rem; min-width: 560px; }
    .content th, .content td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
    .content th { background: var(--surface-2); font-weight: 600; white-space: nowrap; }
    .content tr:last-child td { border-bottom: 0; }
    .content code { padding: 2px 6px; border-radius: 5px; background: var(--surface-2); font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace; font-size: .88em; }
    .content pre { overflow-x: auto; padding: 16px; border-radius: 12px; background: var(--code-bg); color: var(--code-text); line-height: 1.6; }
    .content pre code { padding: 0; background: transparent; color: inherit; font-size: .86em; }
    .content .question {
      margin: 26px 0 14px; padding: 12px 16px;
      border-left: 4px solid var(--accent-2); border-radius: 0 10px 10px 0;
      background: var(--surface); box-shadow: var(--shadow);
    }
    .content .question strong { color: var(--accent); }
    .content .question-en { margin-top: -6px; color: var(--muted); font-size: .94rem; }
    .content .question-en strong { color: var(--muted); }
    .content .answer-en { color: var(--muted); }
    .content strong { color: var(--text); }
    .footer { margin-top: 54px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: .84rem; }
    #toTop {
      position: fixed; right: 22px; bottom: 22px; display: none;
      width: 38px; height: 38px; border-radius: 999px; border: 1px solid var(--border);
      background: var(--surface); color: var(--text); box-shadow: var(--shadow);
      cursor: pointer; font-size: 18px; z-index: 20;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; gap: 18px; padding-top: 20px; }
      .sidebar { position: static; max-height: none; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
      .sidebar nav { display: flex; flex-wrap: wrap; gap: 4px; }
      .sidebar nav a { padding: 6px 9px; background: var(--surface); border: 1px solid var(--border); }
      .side-links { display: flex; flex-wrap: wrap; border-top: 0; padding-top: 0; margin-top: 10px; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero { padding-top: 44px; }
    }
    @media (max-width: 520px) {
      .hero { padding: 36px 18px 30px; }
      .layout { padding-left: 18px; padding-right: 18px; }
      .stat strong { font-size: 1.25rem; }
    }
    @media print {
      .hero { background: #fff; color: #000; padding: 10px 0; }
      .hero::after, .sidebar, .hero-actions, .badges, .stats, #toTop { display: none; }
      .layout { display: block; max-width: none; padding: 0; }
      .content h2 { break-after: avoid; }
      .content table, .content pre { break-inside: avoid; }
    }
  </style>
</head>
<body>
<header class="hero">
  <div class="hero-inner">
    <div class="kicker">Course Project · RAG + Long-term Memory</div>
    <h1>Mnemo<br>项目介绍与答辩问答</h1>
    <p class="lead">一个带长期记忆、可追溯引用的个性化 RAG 助手。文档进入父子块索引，LangGraph Agent 负责检索与生成，PostgreSQL 负责事实与版本，Milvus 负责可重建的检索投影。</p>
    <div class="badges">
      <span>FastAPI</span><span>LangGraph</span><span>Milvus</span>
      <span>PostgreSQL</span><span>Redis</span><span>Streamlit</span>
    </div>
    <div class="stats">
      <div class="stat"><strong>97</strong><span>单元测试</span></div>
      <div class="stat"><strong>50</strong><span>评测问题</span></div>
      <div class="stat"><strong>0.957</strong><span>最佳 Recall@5</span></div>
      <div class="stat"><strong>48</strong><span>答辩问答</span></div>
    </div>
    <div class="hero-actions">
      <a class="button primary" href="__REPO__">GitHub 仓库</a>
      <a class="button" href="__MARKDOWN__">Markdown 源文件</a>
      <a class="button" href="__REPO__#readme">README</a>
    </div>
  </div>
</header>

<div class="layout">
  <aside class="sidebar">
    <p class="side-title">目录</p>
    <nav>
      __NAV__
    </nav>
    <div class="side-links">
      <a href="__REPO__">← 返回 GitHub 仓库</a>
      <a href="__MARKDOWN__">查看 Markdown 源文件</a>
    </div>
  </aside>

  <main class="content">
    __CONTENT__
    <footer class="footer">
      Mnemo · 课程项目介绍 · 生成日期 __DATE__ · <a href="__REPO__">GitHub</a>
    </footer>
  </main>
</div>

<button id="toTop" type="button" title="回到顶部">↑</button>
<script>
  // 给 “Q1：...” 段落加左侧强调条，便于答辩时快速定位。
  document.querySelectorAll('main p > strong').forEach(function (strong) {
    const text = strong.textContent.trim();
    if (/^Q\\d+[：:]/.test(text)) {
      strong.parentElement.classList.add('question');
    } else if (/^Q\\d+ \\(EN\\):/.test(text)) {
      strong.parentElement.classList.add('question-en');
    } else if (text === 'EN:') {
      strong.parentElement.classList.add('answer-en');
    }
  });
  // 目录链接在当前阅读位置高亮。
  const links = Array.from(document.querySelectorAll('.sidebar nav a'));
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + entry.target.id));
    });
  }, { rootMargin: '-15% 0px -70% 0px' });
  sections.forEach(section => observer.observe(section));
  const top = document.getElementById('toTop');
  window.addEventListener('scroll', function () {
    top.style.display = window.scrollY > 600 ? 'block' : 'none';
  });
  top.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
</script>
</body>
</html>
"""
    return (
        template.replace("__REPO__", REPO_URL)
        .replace("__MARKDOWN__", MARKDOWN_URL)
        .replace("__NAV__", nav)
        .replace("__CONTENT__", content)
        .replace("__DATE__", date.today().isoformat())
    )


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.find("## 0.")
    if start < 0:
        raise SystemExit(f"找不到正文起点：{SOURCE}")
    page = build_page(text[start:])
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(page)} chars)")


if __name__ == "__main__":
    main()

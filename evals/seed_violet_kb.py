"""把 violet-evergarden-kb 语料上传到正在运行的 Mnemo 后端。

前置：后端已启动。只上传 docx/ 下 12 个文档；data/qa-benchmark.zh.jsonl
是评测标签，按语料 README 要求绝不入库（避免评测泄漏）。

用法：
    python evals/seed_violet_kb.py --base-url http://127.0.0.1:8010
    python evals/seed_violet_kb.py --corpus-dir data/violet-evergarden-kb --check-only

语料默认放在 `data/violet-evergarden-kb/`（不提交到 Git），也可通过
`VIOLET_KB_DIR` 或 `--corpus-dir` 指定。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_DIR = ROOT / "data" / "violet-evergarden-kb"
TERMINAL_STATES = {"completed", "success", "failed", "error", "ready"}


def resolve_kb_dir(value: str | None) -> Path:
    """解析语料根目录；相对路径按项目根目录处理。"""
    if not value:
        return DEFAULT_KB_DIR
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def main() -> None:
    parser = argparse.ArgumentParser(description="上传 violet-evergarden 语料到知识库")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--kb-id", default="violet-kb")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="语料根目录（含 docx/），默认 data/violet-evergarden-kb 或 VIOLET_KB_DIR",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="只检查语料，不上传"
    )
    args = parser.parse_args()

    kb_dir = resolve_kb_dir(args.corpus_dir or os.getenv("VIOLET_KB_DIR"))
    docx_dir = kb_dir / "docx"
    if not docx_dir.is_dir():
        docx_dir = kb_dir
    if not docx_dir.is_dir():
        raise SystemExit(
            f"找不到语料目录 {docx_dir}。请把 violet-evergarden-kb 放到 "
            "data/violet-evergarden-kb/，或用 --corpus-dir / VIOLET_KB_DIR 指定。"
        )

    # README.docx 是语料说明，不是评测文档；避免它进入知识库成为噪声。
    docx_files = sorted(
        path
        for path in docx_dir.glob("*.docx")
        if path.name.lower() != "readme.docx"
    )
    if not docx_files:
        raise SystemExit(f"{docx_dir} 为空，找不到 docx 语料")
    if len(docx_files) != 12:
        names = ", ".join(path.name for path in docx_files)
        raise SystemExit(f"预期 12 份评测文档，实际 {len(docx_files)} 份：{names}")

    if args.check_only:
        print(f"corpus ok: {len(docx_files)} docx in {docx_dir}")
        return

    base = args.base_url.rstrip("/")
    response = requests.post(
        f"{base}/knowledge-bases",
        params={"knowledge_base_id": args.kb_id, "user_id": args.user_id},
        timeout=15,
    )
    print("create kb:", response.status_code, response.text[:120])

    for docx in docx_files:
        with docx.open("rb") as handle:
            response = requests.post(
                f"{base}/document_upload",
                data={"knowledge_base_id": args.kb_id, "user_id": str(args.user_id)},
                files={
                    "file": (
                        docx.name,
                        handle,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                timeout=120,
            )
        print(f"upload {docx.name}: {response.status_code} {response.text[:100]}")

    expected = len(docx_files)
    for _ in range(120):
        time.sleep(2)
        response = requests.get(
            f"{base}/knowledge-bases/{args.kb_id}/files",
            params={"user_id": args.user_id},
            timeout=15,
        )
        if response.status_code != 200:
            continue
        files = response.json().get("files", [])
        done = [f for f in files if (f.get("status") or "").lower() in TERMINAL_STATES]
        failed = [f for f in done if (f.get("status") or "").lower() in ("failed", "error")]
        if failed:
            names = [(f.get("file_name"), f.get("error")) for f in failed]
            raise SystemExit(f"以下文件处理失败: {names}")
        if len(done) >= expected:
            print(f"ready: {expected} files in {args.kb_id}")
            return
    raise SystemExit(f"等待超时：{args.kb_id} 中只有 {len(done)}/{expected} 个文件就绪")


if __name__ == "__main__":
    main()

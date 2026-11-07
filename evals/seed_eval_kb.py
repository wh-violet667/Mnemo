"""把评测语料上传到正在运行的 Mnemo 后端。

前置：
    python evals/build_corpus.py
    python evals/seed_eval_kb.py --base-url http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "evals" / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description="上传评测语料到知识库")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--kb-id", default="eval-kb")
    args = parser.parse_args()

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("evals/corpus 为空，先运行 python evals/build_corpus.py")

    base = args.base_url.rstrip("/")
    response = requests.post(
        f"{base}/knowledge-bases",
        params={"knowledge_base_id": args.kb_id, "user_id": args.user_id},
        timeout=15,
    )
    print("create kb:", response.status_code, response.text[:120])

    for pdf in pdfs:
        with pdf.open("rb") as handle:
            response = requests.post(
                f"{base}/document_upload",
                data={"knowledge_base_id": args.kb_id, "user_id": str(args.user_id)},
                files={"file": (pdf.name, handle, "application/pdf")},
                timeout=120,
            )
        print(f"upload {pdf.name}: {response.status_code} {response.text[:100]}")

    expected = len(pdfs)
    for _ in range(60):
        time.sleep(2)
        response = requests.get(
            f"{base}/knowledge-bases/{args.kb_id}/files",
            params={"user_id": args.user_id},
            timeout=15,
        )
        if response.status_code == 200 and response.json().get("count", 0) >= expected:
            print(f"ready: {expected} files in {args.kb_id}")
            return
    raise SystemExit(f"等待超时：{args.kb_id} 中没有出现 {expected} 个文件")


if __name__ == "__main__":
    main()

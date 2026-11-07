"""生成检索评测用的 5 份 PDF 语料。

运行：
    python evals/build_corpus.py
生成到 evals/corpus/，然后运行 evals/seed_eval_kb.py 上传。
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "evals" / "corpus"

DOCUMENTS = {
    "refund_policy.pdf": {
        "title": "Mnemo Refund Policy (Current)",
        "body": (
            "Customers may request a refund within 30 days of purchase. "
            "Refunds are processed within 5 business days. "
            "To start a refund, email support@mnemo.test with the order number. "
            "Refunds are issued to the original payment method. "
            "Shipping fees are non-refundable unless the item is defective."
        ),
    },
    "employee_handbook.pdf": {
        "title": "Mnemo Employee Handbook (Current)",
        "body": (
            "The probation period is 3 months. "
            "Employees receive 10 days of paid annual leave per year. "
            "Standard working hours are 09:00 to 18:00, Monday to Friday. "
            "Employees may work remotely up to 2 days per week. "
            "Overtime requires manager approval."
        ),
    },
    "travel_policy.pdf": {
        "title": "Mnemo Travel Policy (Current)",
        "body": (
            "Hotel accommodation is capped at 500 CNY per night. "
            "Domestic flights must be economy class. "
            "Submit reimbursement within 7 days after the trip. "
            "Manager approval is required before booking. "
            "Receipts are required for all expenses."
        ),
    },
    "security_policy.pdf": {
        "title": "Mnemo Security Policy (Current)",
        "body": (
            "Passwords must be at least 12 characters and rotated every 90 days. "
            "Multi-factor authentication is mandatory for all employees. "
            "Security incidents must be reported within 1 hour. "
            "Do not share credentials. "
            "Use the company VPN when working remotely."
        ),
    },
    "product_faq.pdf": {
        "title": "Mnemo Product FAQ (Current)",
        "body": (
            "The knowledge base supports PDF and DOCX files. "
            "The maximum upload size is 200 MB. "
            "Uploaded documents are retained for 30 days after deletion. "
            "The default knowledge base searches across all libraries. "
            "Each user can create multiple knowledge bases."
        ),
    },
    "refund_policy_legacy.pdf": {
        "title": "Legacy Refund Policy (Archived)",
        "body": (
            "The legacy policy allowed refunds within 14 days of purchase. "
            "Refunds were processed within 10 business days. "
            "This policy is archived and applies only to orders placed before 2024."
        ),
    },
    "travel_policy_emea.pdf": {
        "title": "EMEA Travel Policy",
        "body": (
            "For the EMEA region, hotel accommodation is capped at 800 EUR per night. "
            "Business class is allowed for flights longer than 6 hours. "
            "Reimbursement must be submitted within 14 days."
        ),
    },
    "security_policy_contractor.pdf": {
        "title": "Contractor Security Policy",
        "body": (
            "Contractors must use passwords of at least 8 characters and rotate them every 180 days. "
            "Multi-factor authentication is recommended but not mandatory. "
            "Incidents must be reported within 4 hours."
        ),
    },
    "employee_handbook_intern.pdf": {
        "title": "Intern Handbook",
        "body": (
            "The internship probation period is 6 months. "
            "Interns receive 5 days of paid annual leave per year. "
            "Interns may work remotely 1 day per week."
        ),
    },
    "product_faq_legacy.pdf": {
        "title": "Legacy Product FAQ (Archived)",
        "body": (
            "The legacy product supports PDF files only. "
            "The maximum upload size is 100 MB. "
            "Documents are retained for 7 days after deletion."
        ),
    },
}


def build() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, data in DOCUMENTS.items():
        document = fitz.open()
        page = document.new_page()
        text = data["title"] + "\n\n" + textwrap.fill(data["body"], width=88)
        page.insert_textbox(
            fitz.Rect(72, 72, 540, 780),
            text,
            fontsize=12,
            lineheight=1.5,
        )
        output = CORPUS_DIR / filename
        document.save(output)
        document.close()
        print(f"created {output}")


if __name__ == "__main__":
    build()

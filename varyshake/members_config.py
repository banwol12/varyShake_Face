"""
NCT 127 Members Configuration (Shared Module)
==============================================
모든 Python 스크립트에서 공유하는 멤버 정보 및 경로 상수.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBERS_DIR = os.path.join(BASE_DIR, "public", "members")
DESCRIPTORS_JSON = os.path.join(BASE_DIR, "public", "descriptors.json")
MEMBERS_JSON = os.path.join(BASE_DIR, "public", "members.json")

# NCT 127 Active Members (7)
MEMBERS = [
    {"id": "johnny",  "eng": "Johnny",  "kor": "쟈니"},
    {"id": "taeyong", "eng": "Taeyong", "kor": "태용"},
    {"id": "yuta",    "eng": "Yuta",    "kor": "유타"},
    {"id": "doyoung", "eng": "Doyoung", "kor": "도영"},
    {"id": "jaehyun", "eng": "Jaehyun", "kor": "재현"},
    {"id": "jungwoo", "eng": "Jungwoo", "kor": "정우"},
    {"id": "haechan", "eng": "Haechan", "kor": "해찬"},
]

# ID-only set for quick membership checks
ALLOWED_IDS = {m["id"] for m in MEMBERS}

# ID -> display name lookup
MEMBER_NAMES = {m["id"]: {"name": m["eng"], "korName": m["kor"]} for m in MEMBERS}

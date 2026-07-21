import os
import json
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBERS_DIR = os.path.join(BASE_DIR, "public", "members")
OUTPUT_JSON = os.path.join(BASE_DIR, "public", "descriptors.json")
MEMBERS_JSON = os.path.join(BASE_DIR, "public", "members.json")

MEMBERS = [
    {"id": "johnny", "eng": "Johnny", "kor": "쟈니"},
    {"id": "taeyong", "eng": "Taeyong", "kor": "태용"},
    {"id": "yuta", "eng": "Yuta", "kor": "유타"},
    {"id": "doyoung", "eng": "Doyoung", "kor": "도영"},
    {"id": "jaehyun", "eng": "Jaehyun", "kor": "재현"},
    {"id": "jungwoo", "eng": "Jungwoo", "kor": "정우"},
    {"id": "haechan", "eng": "Haechan", "kor": "해찬"}
]

def check_dependencies():
    try:
        import face_recognition
        import cv2
        return True
    except ImportError:
        print("[!] 'face_recognition' or 'opencv-python' is not installed.")
        return False

def apply_lighting_normalization(img_cv2):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 조명 정규화 적용"""
    import cv2
    import numpy as np
    
    # LAB 색상 공간으로 변환하여 밝기(L) 채널만 정규화
    lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def build():
    if not check_dependencies():
        return

    import face_recognition
    import cv2

    print("==========================================================")
    print("  NCT 127 128-D DESCRIPTORS OFFLINE PRE-BUILDER v2.0     ")
    print("==========================================================")
    
    labeled_descriptors = []
    total_images_processed = 0
    total_descriptors_kept = 0

    for m in MEMBERS:
        member_id = m["id"]
        eng_name = m["eng"]
        member_dir = os.path.join(MEMBERS_DIR, member_id)

        if not os.path.exists(member_dir):
            print(f"[-] [{eng_name}] Directory not found (skipped)")
            continue

        image_files = [
            f for f in sorted(os.listdir(member_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        print(f"\n[+] [{eng_name}] Processing {len(image_files)} images...")
        member_descriptors = []

        for fname in image_files:
            total_images_processed += 1
            fpath = os.path.join(member_dir, fname)

            try:
                # 1. OpenCV로 로드하여 조명 정규화
                img_cv2 = cv2.imread(fpath)
                if img_cv2 is None:
                    continue

                # 조명 및 대조 정규화
                norm_cv2 = apply_lighting_normalization(img_cv2)
                rgb_img = cv2.cvtColor(norm_cv2, cv2.COLOR_BGR2RGB)

                # 2. 특징점 계산
                encs = face_recognition.face_encodings(rgb_img)
                if encs and len(encs) == 1:
                    # float 리스트로 변환하여 저장
                    desc_list = encs[0].tolist()
                    member_descriptors.append(desc_list)
                    total_descriptors_kept += 1
            except Exception as e:
                pass

        if member_descriptors:
            labeled_descriptors.append({
                "label": member_id,
                "descriptors": member_descriptors
            })
            print(f"  [v] {len(member_descriptors)}/{len(image_files)} 128-D vectors extracted for {eng_name}")

    # descriptors.json 쓰기
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labeled_descriptors, f)

    print(f"\n==================================================")
    print(f"[SUCCESS] descriptors.json created successfully!")
    print(f"  - Total images processed: {total_images_processed}")
    print(f"  - Total 128-D vectors saved: {total_descriptors_kept}")
    print(f"  - Output file: {OUTPUT_JSON}")
    print(f"==================================================")

    # members.json도 함께 최신화
    try:
        import generate_db_json
        generate_db_json.main()
    except Exception as e:
        print(f"members.json 갱신 중 오류: {e}")

if __name__ == "__main__":
    build()

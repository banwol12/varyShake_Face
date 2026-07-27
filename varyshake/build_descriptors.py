import os
import sys
import json
import ssl
import time

try:
    import numpy as np
except ImportError:
    np = None

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
        import cv2
        return True
    except ImportError:
        print("[!] 'opencv-python' is not installed.")
        return False

def apply_concert_white_balance(img_cv2):
    """Gray-World White Balance: 콘서트 무대 조명(빨강/파랑/보라) 색 편향 자동 중화"""
    import cv2
    result = img_cv2.copy().astype('float32')
    avg_b = result[:, :, 0].mean()
    avg_g = result[:, :, 1].mean()
    avg_r = result[:, :, 2].mean()
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0:
        result[:, :, 0] *= (avg_all / avg_b)
    if avg_g > 0:
        result[:, :, 1] *= (avg_all / avg_g)
    if avg_r > 0:
        result[:, :, 2] *= (avg_all / avg_r)

    return np.clip(result, 0, 255).astype('uint8')

def apply_lighting_normalization(img_cv2):
    """콘서트 무대 조명 보정 + CLAHE 밝기/대비 정규화 (Stage-Grade Preprocessing)"""
    import cv2
    # 1단계: Gray-World White Balance로 무대 색 조명 편향 중화
    wb_img = apply_concert_white_balance(img_cv2)

    # 2단계: LAB 색공간에서 CLAHE 밝기 정규화
    lab = cv2.cvtColor(wb_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 3단계: 색상 채널(a, b) 극단값 평활화 (보라/초록 조명 등 극단적 색조 완화)
    a = cv2.normalize(a, None, 100, 155, cv2.NORM_MINMAX)
    b = cv2.normalize(b, None, 100, 155, cv2.NORM_MINMAX)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def align_face(rgb_img, face_landmarks=None):
    """양쪽 눈 좌표 기반 얼굴 수평 회전 정렬 (Face Alignment)"""
    import cv2
    try:
        import face_recognition
    except ImportError:
        face_recognition = None

    if face_recognition is None:
        return rgb_img

    if face_landmarks is None:
        landmarks_list = face_recognition.face_landmarks(rgb_img)
        if not landmarks_list:
            return rgb_img
        face_landmarks = landmarks_list[0]

    if 'left_eye' not in face_landmarks or 'right_eye' not in face_landmarks:
        return rgb_img

    left_eye_pts = np.array(face_landmarks['left_eye'])
    right_eye_pts = np.array(face_landmarks['right_eye'])

    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[2] - left_eye_center[0] if len(right_eye_center) > 1 and len(left_eye_center) > 0 else (right_eye_center[0] - left_eye_center[0])
    
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                   (left_eye_center[1] + right_eye_center[1]) / 2.0)

    h, w = rgb_img.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(rgb_img, M, (w, h), flags=cv2.INTER_CUBIC)
    return aligned

import math

def l2_normalize(vec):
    """벡터 L2 정규화 (단위 벡터 변환)"""
    try:
        import numpy as np
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            return v.tolist()
        return (v / norm).tolist()
    except ImportError:
        val_sq = sum(float(x) ** 2 for x in vec)
        norm = math.sqrt(val_sq)
        if norm == 0:
            return [float(x) for x in vec]
        return [float(x) / norm for x in vec]

def init_insightface_app():
    """InsightFace 512D ArcFace 모델 로드 시도"""
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("[*] Initializing InsightFace 512D ArcFace Engine (buffalo_l)...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[OK] InsightFace 512D ArcFace Engine initialized successfully!")
        return app
    except Exception as e:
        print(f"[!] InsightFace initialization warning: {e}")
        return None

def build():
    if not check_dependencies():
        return

    try:
        import face_recognition
    except ImportError:
        face_recognition = None
    import cv2

    print("==========================================================")
    print("  NCT 127 ADVANCED 512-D / 128-D FACE BUILDER v4.0       ")
    print("==========================================================")

    insight_app = init_insightface_app()

    if insight_app is not None:
        mode_name = "InsightFace ArcFace 512-D (SOTA AI Model)"
    else:
        mode_name = "dlib / OpenCV 128-D Aligned Normalization"
    print(f"[*] Engine Mode: {mode_name}")
    
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
                img_cv2 = cv2.imread(fpath)
                if img_cv2 is None:
                    continue

                extracted_vec = None

                # 1. InsightFace 512-D ArcFace 시도
                if insight_app is not None:
                    faces = insight_app.get(img_cv2)
                    if faces and len(faces) >= 1:
                        # 가장 큰 얼굴 선택
                        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                        if hasattr(faces[0], 'normed_embedding') and faces[0].normed_embedding is not None:
                            extracted_vec = faces[0].normed_embedding.tolist()
                        elif hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                            extracted_vec = l2_normalize(faces[0].embedding)

                # 2. InsightFace 실패 시 dlib 128-D Fallback
                if extracted_vec is None and face_recognition is not None:
                    norm_cv2 = apply_lighting_normalization(img_cv2)
                    rgb_img = cv2.cvtColor(norm_cv2, cv2.COLOR_BGR2RGB)
                    aligned_rgb = align_face(rgb_img)
                    encs = face_recognition.face_encodings(aligned_rgb)
                    if not encs:
                        encs = face_recognition.face_encodings(rgb_img)
                    if encs and len(encs) >= 1:
                        extracted_vec = l2_normalize(encs[0])

                if extracted_vec is not None:
                    member_descriptors.append(extracted_vec)
                    total_descriptors_kept += 1

            except Exception as e:
                pass

        if member_descriptors:
            labeled_descriptors.append({
                "label": member_id,
                "descriptors": member_descriptors
            })
            vector_dim = len(member_descriptors[0])
            print(f"  [v] {len(member_descriptors)}/{len(image_files)} {vector_dim}-D vectors extracted for {eng_name}")

    # descriptors.json 쓰기
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labeled_descriptors, f)

    vector_dim_info = len(labeled_descriptors[0]["descriptors"][0]) if labeled_descriptors else 128
    print(f"\n==================================================")
    print(f"[SUCCESS] descriptors.json created successfully!")
    print(f"  - Engine Mode: {mode_name}")
    print(f"  - Total images processed: {total_images_processed}")
    print(f"  - Total {vector_dim_info}-D vectors saved: {total_descriptors_kept}")
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



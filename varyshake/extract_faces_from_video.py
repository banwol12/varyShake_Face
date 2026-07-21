import os
import sys
import argparse
import glob
import time
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

MEMBERS = [
    {"id": "johnny", "eng": "Johnny", "kor": "쟈니"},
    {"id": "taeyong", "eng": "Taeyong", "kor": "태용"},
    {"id": "yuta", "eng": "Yuta", "kor": "유타"},
    {"id": "doyoung", "eng": "Doyoung", "kor": "도영"},
    {"id": "jaehyun", "eng": "Jaehyun", "kor": "재현"},
    {"id": "jungwoo", "eng": "Jungwoo", "kor": "정우"},
    {"id": "haechan", "eng": "Haechan", "kor": "해찬"}
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBERS_DIR = os.path.join(BASE_DIR, "public", "members")

def check_dependencies():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import face_recognition
    except ImportError:
        missing.append("face-recognition")

    if missing:
        print("❌ 필수 라이브러리가 설치되어 있지 않습니다:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

def apply_lighting_normalization(img_cv2):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 조명 및 대조 정규화"""
    import cv2
    lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def load_member_anchors(member_id):
    import face_recognition
    import cv2
    member_dir = os.path.join(MEMBERS_DIR, member_id)
    if not os.path.exists(member_dir):
        return []

    all_files = []
    for f in os.listdir(member_dir):
        if not f.startswith("video_") and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            all_files.append(os.path.join(member_dir, f))

    encodings = []
    print(f"📌 '{member_id}' 앵커(기준) 이미지 {len(all_files)}장 분석 중...")
    for path in all_files:
        try:
            img_cv2 = cv2.imread(path)
            if img_cv2 is None:
                continue
            norm_cv2 = apply_lighting_normalization(img_cv2)
            rgb_img = cv2.cvtColor(norm_cv2, cv2.COLOR_BGR2RGB)
            face_encs = face_recognition.face_encodings(rgb_img)
            if face_encs:
                encodings.append(face_encs[0])
        except Exception:
            pass

    print(f"  ✓ {len(encodings)}개의 기준 얼굴 특징점 로드 완료.")
    return encodings

def calc_blur_score(image_cv2):
    import cv2
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_video(video_path, member_id, blur_threshold=100.0, tolerance=0.48, sample_interval_sec=0.5, max_faces=15, min_diversity_dist=0.22):
    import cv2
    import face_recognition

    anchor_encs = load_member_anchors(member_id)
    if not anchor_encs:
        print(f"❌ '{member_id}' 멤버의 기존 앵커(기준) 이미지가 필요합니다.")
        return

    output_dir = os.path.join(MEMBERS_DIR, member_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎬 동영상 처리 시작: {video_path}")
    print(f"   - 대상 멤버: {member_id}")
    print(f"   - 최소 선명도(Blur Score): {blur_threshold}")
    print(f"   - 엄격도 Threshold(Tolerance): {tolerance}")
    print(f"   - 중복 포즈 차단(Diversity Distance): {min_diversity_dist}")
    print(f"   - 조명 정규화(CLAHE): 적용됨")

    # Handle YouTube URL automatically
    temp_downloaded_file = None
    if video_path.startswith("http://") or video_path.startswith("https://"):
        import subprocess
        print(f"🌐 유튜브 URL 감지. 영상 다운로드 중...")
        temp_downloaded_file = os.path.join(BASE_DIR, f"temp_yt_{int(time.time())}.mp4")
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best[height<=1080]/best",
            "-o", temp_downloaded_file,
            "--no-part",
            "--no-continue",
            "--no-playlist",
            video_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print(f"❌ 유튜브 영상 다운로드 실패: {res.stderr}")
                return
            video_path = temp_downloaded_file
            print(f"✓ 유튜브 영상 다운로드 완료.")
        except Exception as e:
            print(f"❌ yt-dlp 실행 오류: {e}")
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다: {video_path}")
        if temp_downloaded_file and os.path.exists(temp_downloaded_file):
            os.remove(temp_downloaded_file)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(1, int(fps * sample_interval_sec))

    frame_idx = 0
    saved_count = 0
    discarded_blur = 0
    discarded_duplicate = 0
    discarded_no_match = 0

    saved_session_encodings = []
    start_timestamp = int(time.time())

    while cap.isOpened() and saved_count < max_faces:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        # 1️⃣ 선명도 (Blur) 검사
        blur_score = calc_blur_score(frame)
        if blur_score < blur_threshold:
            discarded_blur += 1
            continue

        # 2️⃣ 조명/대조 정규화 적용 후 RGB 변환
        norm_frame = apply_lighting_normalization(frame)
        rgb_frame = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2RGB)

        # 3️⃣ 얼굴 탐지 & 특징점 추출
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            continue

        candidate_encs = face_recognition.face_encodings(rgb_frame, face_locations)

        matched = False
        for i, cand_enc in enumerate(candidate_encs):
            # 앵커 이미지와의 유사도 비교
            distances = face_recognition.face_distance(anchor_encs, cand_enc)
            min_dist = min(distances) if len(distances) > 0 else 1.0

            if min_dist <= tolerance:
                # 얼굴 크기 검사 (최소 70px)
                top, right, bottom, left = face_locations[i]
                if (right - left) < 70 or (bottom - top) < 70:
                    continue

                # 4️⃣ 다변화 중복 검사 (Diversity Sampler)
                # 이전에 추출한 프레임들과 각도가 너무 비슷하면 버림
                if saved_session_encodings:
                    diversity_dists = face_recognition.face_distance(saved_session_encodings, cand_enc)
                    min_div_dist = min(diversity_dists)
                    if min_div_dist < min_diversity_dist:
                        discarded_duplicate += 1
                        continue

                filename = f"video_{start_timestamp}_{saved_count+1:03d}.jpg"
                save_path = os.path.join(output_dir, filename)

                cv2.imwrite(save_path, frame)
                saved_session_encodings.append(cand_enc)
                saved_count += 1
                matched = True
                print(f"  ✓ [{saved_count}/{max_faces}] 합격! (선명도: {blur_score:.1f}, 앵커거리: {min_dist:.3f}) -> {filename}")
                break

        if not matched:
            discarded_no_match += 1

    cap.release()

    if temp_downloaded_file and os.path.exists(temp_downloaded_file):
        try:
            os.remove(temp_downloaded_file)
        except Exception:
            pass

    print(f"\n==================================================")
    print(f"🎉 동영상 프레임 고품질 수집 완료!")
    print(f"  - 저장된 고품질 프레임: {saved_count}장")
    print(f"  - 흐려서 제거: {discarded_blur}개")
    print(f"  - 중복 포즈/각도 제거(Diversity Filter): {discarded_duplicate}개")
    print(f"  - 멤버 불일치 제거: {discarded_no_match}개")
    print(f"==================================================")

    # 128-D Descriptors 오프라인 빌더 실행
    import build_descriptors
    build_descriptors.build()

def main():
    if not check_dependencies():
        return

    parser = argparse.ArgumentParser(description="동영상 고품질 다변화 얼굴 추출 필터")
    parser.add_argument("--video", type=str, required=True, help="동영상 파일 경로 또는 유튜브 URL")
    parser.add_argument("--member", type=str, required=True, choices=[m["id"] for m in MEMBERS], help="대상 멤버 ID")
    parser.add_argument("--blur", type=float, default=100.0, help="최소 선명도 (기본값: 100.0)")
    parser.add_argument("--tolerance", type=float, default=0.48, help="얼굴 일치 엄격도 (기본값: 0.48)")
    parser.add_argument("--interval", type=float, default=0.5, help="추출 간격 (초, 기본값: 0.5초)")
    parser.add_argument("--max", type=int, default=15, help="최대 추출장수 (기본값: 15장)")
    parser.add_argument("--diversity", type=float, default=0.22, help="중복 포즈 차단 컷오프 (기본값: 0.22)")

    args = parser.parse_args()
    process_video(args.video, args.member, args.blur, args.tolerance, args.interval, args.max, args.diversity)

if __name__ == "__main__":
    main()

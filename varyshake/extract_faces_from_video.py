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

import numpy as np

def apply_concert_white_balance(img_cv2):
    """Gray-World White Balance: 콘서트 무대 조명(빨강/파랑/보라) 색 편향 자동 중화"""
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

    # 3단계: 색상 채널(a, b) 극단값 평활화
    a = cv2.normalize(a, None, 100, 155, cv2.NORM_MINMAX)
    b = cv2.normalize(b, None, 100, 155, cv2.NORM_MINMAX)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def align_face(rgb_img, face_landmarks=None):
    """양쪽 눈 좌표 기반 얼굴 수평 회전 정렬 (Face Alignment)"""
    import cv2
    import face_recognition

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
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                   (left_eye_center[1] + right_eye_center[1]) / 2.0)

    h, w = rgb_img.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    return cv2.warpAffine(rgb_img, M, (w, h), flags=cv2.INTER_CUBIC)

def l2_normalize(vec):
    """벡터 L2 정규화 (단위 벡터 변환)"""
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

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
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"InsightFace 로드 경고: {e}")
        return None

def compute_vector_distances(anchors, candidate_vec):
    """L2-normalized 벡터들 간의 유클리드 거리 배열 계산"""
    anchors_np = np.array(anchors, dtype=np.float32)
    cand_np = np.array(candidate_vec, dtype=np.float32)
    return np.linalg.norm(anchors_np - cand_np, axis=1)

def load_member_anchors(member_id, insight_app=None):
    import cv2
    try:
        import face_recognition
    except ImportError:
        face_recognition = None

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

            extracted_vec = None
            if insight_app is not None:
                faces = insight_app.get(img_cv2)
                if faces and len(faces) >= 1:
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    if hasattr(faces[0], 'normed_embedding') and faces[0].normed_embedding is not None:
                        extracted_vec = faces[0].normed_embedding
                    elif hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                        extracted_vec = l2_normalize(faces[0].embedding)

            if extracted_vec is None and face_recognition is not None:
                norm_cv2 = apply_lighting_normalization(img_cv2)
                rgb_img = cv2.cvtColor(norm_cv2, cv2.COLOR_BGR2RGB)
                aligned_rgb = align_face(rgb_img)
                face_encs = face_recognition.face_encodings(aligned_rgb)
                if not face_encs:
                    face_encs = face_recognition.face_encodings(rgb_img)
                if face_encs:
                    extracted_vec = l2_normalize(face_encs[0])

            if extracted_vec is not None:
                encodings.append(extracted_vec)
        except Exception:
            pass

    vec_dim = len(encodings[0]) if encodings else 0
    print(f"  ✓ {len(encodings)}개의 정렬된 기준 얼굴 {vec_dim}-D 특징점 로드 완료.")
    return encodings

def calc_blur_score(image_cv2):
    import cv2
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_video(video_path, member_id, blur_threshold=100.0, tolerance=0.45, sample_interval_sec=0.5, max_faces=15, min_diversity_dist=0.25):
    import cv2
    try:
        import face_recognition
    except ImportError:
        face_recognition = None

    insight_app = init_insightface_app()
    if insight_app is not None:
        print("[*] InsightFace 512D ArcFace SOTA Engine 활성화됨")
        # 512D ArcFace용 기본 tolerance 조정 (기존 128D 0.5~0.6 -> 512D 0.40~0.48)
        if tolerance == 0.05 or tolerance < 0.2:
            tolerance = 0.45
    else:
        print("[*] dlib 128D Engine Fallback 활성화됨")
        if tolerance > 0.3:
            tolerance = 0.55

    anchor_encs = load_member_anchors(member_id, insight_app)
    if not anchor_encs:
        print(f"❌ '{member_id}' 멤버의 기존 앵커(기준) 이미지가 필요합니다.")
        return

    output_dir = os.path.join(MEMBERS_DIR, member_id)
    os.makedirs(output_dir, exist_ok=True)

    vec_dim = len(anchor_encs[0]) if anchor_encs else 128

    print(f"\n🎬 동영상 처리 시작: {video_path}")
    print(f"   - 대상 멤버: {member_id}")
    print(f"   - AI 특징점 엔진 차원: {vec_dim}-D")
    print(f"   - 최소 선명도(Blur Score): {blur_threshold}")
    print(f"   - 엄격도 Threshold(Tolerance): {tolerance}")
    print(f"   - 중복 포즈 차단(Diversity Distance): {min_diversity_dist}")
    print(f"   - 조명 정규화 & Face Alignment 적용됨")

    # Handle YouTube URL automatically
    temp_downloaded_file = None
    if video_path.startswith("http://") or video_path.startswith("https://"):
        import subprocess
        print(f"🌐 유튜브 URL 감지. 영상 다운로드 중...")
        temp_downloaded_file = os.path.join(BASE_DIR, f"temp_yt_{int(time.time())}.mp4")
        cmd = [
            "yt-dlp",
            "-f", "bv*[height<=1080]+ba/bv*[height<=1080]/b[height<=1080]/bestvideo[height<=1080]/best",
            "-o", temp_downloaded_file,
            "--no-part",
            "--no-continue",
            "--no-playlist",
            video_path
        ]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            last_line_was_progress = False
            for line in iter(p.stdout.readline, ''):
                if not line:
                    break
                line_str = line.strip()
                if not line_str:
                    continue
                if "[download]" in line_str:
                    sys.stdout.write(f"\r[유튜브 다운로드] {line_str}                ")
                    sys.stdout.flush()
                    last_line_was_progress = True
                else:
                    if last_line_was_progress:
                        sys.stdout.write("\n")
                        last_line_was_progress = False
                    print(f"[yt-dlp] {line_str}")
            p.stdout.close()
            ret_code = p.wait()
            if last_line_was_progress:
                sys.stdout.write("\n")
            if ret_code != 0:
                print(f"❌ 유튜브 영상 다운로드 실패 (종료 코드: {ret_code})")
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * sample_interval_sec))
    duration_sec = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0

    if total_frames > 0:
        print(f"📹 영상 정보: 총 {total_frames} 프레임 ({duration_sec:.1f}초, {fps:.1f} FPS)")
    print(f"🔍 동영상 얼굴 학습 분석 진행 중...\n")

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

        # 실시간 CMD 진행률 출력 (커서 갱신)
        if total_frames > 0:
            progress_pct = (frame_idx / total_frames) * 100
            sys.stdout.write(f"\r[동영상 학습 진행률] {frame_idx}/{total_frames} 프레임 ({progress_pct:.1f}%) | 수집된 얼굴: {saved_count}/{max_faces}장    ")
        else:
            sys.stdout.write(f"\r[동영상 학습 진행률] {frame_idx} 프레임 처리 중 | 수집된 얼굴: {saved_count}/{max_faces}장    ")
        sys.stdout.flush()

        if frame_idx % frame_interval != 0:
            continue

        # 1️⃣ 선명도 (Blur) 검사
        blur_score = calc_blur_score(frame)
        if blur_score < blur_threshold:
            discarded_blur += 1
            continue

        candidate_encs = []
        face_boxes = []

        # 2️⃣ InsightFace 512-D SCRFD/ArcFace 추출
        if insight_app is not None:
            try:
                faces = insight_app.get(frame)
                for f in faces:
                    bbox = f.bbox.astype(int) # top, right, bottom, left
                    w_box = bbox[2] - bbox[0]
                    h_box = bbox[3] - bbox[1]
                    if w_box >= 40 and h_box >= 40:
                        vec = f.normed_embedding if (hasattr(f, 'normed_embedding') and f.normed_embedding is not None) else l2_normalize(f.embedding)
                        candidate_encs.append(vec)
                        face_boxes.append(bbox)
            except Exception:
                pass

        # 3️⃣ Fallback to face_recognition if no candidate_encs found
        if not candidate_encs and face_recognition is not None:
            norm_frame = apply_lighting_normalization(frame)
            rgb_frame = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                aligned_frame = align_face(rgb_frame)
                raw_encs = face_recognition.face_encodings(aligned_frame)
                if not raw_encs:
                    raw_encs = face_recognition.face_encodings(rgb_frame, face_locations)
                for enc in raw_encs:
                    candidate_encs.append(l2_normalize(enc))

        matched = False
        for i, cand_enc in enumerate(candidate_encs):
            # 앵커 이미지와의 유사도 비교 (Normalized Euclidean Distance)
            distances = compute_vector_distances(anchor_encs, cand_enc)
            min_dist = float(min(distances)) if len(distances) > 0 else 1.0

            if min_dist <= tolerance:
                # 4️⃣ 다변화 중복 검사 (Diversity Sampler)
                if saved_session_encodings:
                    diversity_dists = compute_vector_distances(saved_session_encodings, cand_enc)
                    min_div_dist = float(min(diversity_dists))
                    if min_div_dist < min_diversity_dist:
                        discarded_duplicate += 1
                        continue

                filename = f"video_{start_timestamp}_{saved_count+1:03d}.jpg"
                save_path = os.path.join(output_dir, filename)

                cv2.imwrite(save_path, frame)
                saved_session_encodings.append(cand_enc)
                saved_count += 1
                matched = True
                sys.stdout.write("\n")
                print(f"  ✓ [{saved_count}/{max_faces}] 합격! (선명도: {blur_score:.1f}, 앵커거리: {min_dist:.3f}) -> {filename}")
                break

        if not matched:
            discarded_no_match += 1

    sys.stdout.write("\n")
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

    # 오프라인 빌더 실행
    import build_descriptors
    build_descriptors.build()

def main():
    if not check_dependencies():
        return

    parser = argparse.ArgumentParser(description="동영상 고품질 다변화 얼굴 추출 필터")
    parser.add_argument("--video", type=str, required=True, help="동영상 파일 경로 또는 유튜브 URL")
    parser.add_argument("--member", type=str, required=True, choices=[m["id"] for m in MEMBERS], help="대상 멤버 ID")
    parser.add_argument("--blur", type=float, default=100.0, help="최소 선명도 (기본값: 100.0)")
    parser.add_argument("--tolerance", type=float, default=0.05, help="얼굴 일치 엄격도 (기본값: 0.05, 95%+ 일치)")
    parser.add_argument("--interval", type=float, default=0.5, help="추출 간격 (초, 기본값: 0.5초)")
    parser.add_argument("--max", type=int, default=15, help="최대 추출장수 (기본값: 15장)")
    parser.add_argument("--diversity", type=float, default=0.22, help="중복 포즈 차단 컷오프 (기본값: 0.22)")

    args = parser.parse_args()
    process_video(args.video, args.member, args.blur, args.tolerance, args.interval, args.max, args.diversity)

if __name__ == "__main__":
    main()

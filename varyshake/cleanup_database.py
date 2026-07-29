"""
NCT 127 Face Database Smart Garbage Cleaner & Serializer v2.0
=============================================================
1. 얼굴이 미감지된 배경/손상 이미지는 영구 삭제 (Delete)
2. 타 멤버 얼굴로 밝혀진 이미지는 _trash 폴더로 이동
3. 백댄서/관중/애매한 인물은 _review 폴더로 이동 (사용자 직접 확인용)
4. 살아남은 정상 이미지는 파일명을 순차적 정렬 (image_001.jpg, image_002.jpg...)
"""

import os
import sys
import shutil
import argparse
import time
import glob
import numpy as np

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from members_config import MEMBERS, MEMBERS_DIR, BASE_DIR

def l2_normalize(vec):
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calc_cosine_distance(vecA, vecB):
    vA = l2_normalize(vecA)
    vB = l2_normalize(vecB)
    dot = np.dot(vA, vB)
    return 1.0 - float(np.clip(dot, -1.0, 1.0))

def init_insightface():
    """InsightFace 512-D ArcFace 엔진 초기화"""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"[!] InsightFace 로드 실패: {e}")
        return None

def calc_blur_score(img_cv2):
    """이미지 선명도 점수 계산 (Laplacian Variance)"""
    import cv2
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def load_all_member_anchors(insight_app):
    """모든 멤버의 시드 앵커 특징점(512-D) 사전 로드"""
    import cv2
    anchors = {m["id"]: [] for m in MEMBERS}

    print("[*] 7명 멤버의 시드 기준 특징점(Anchor Vectors) 추출 중...")
    for m in MEMBERS:
        member_id = m["id"]
        member_dir = os.path.join(MEMBERS_DIR, member_id)
        if not os.path.exists(member_dir):
            continue

        files = [
            f for f in sorted(os.listdir(member_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            and not f.startswith('.') and not f.startswith('_')
        ]

        for fname in files[:30]:  # 각 멤버당 상위 30장의 고품질 시드 사용
            fpath = os.path.join(member_dir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                faces = insight_app.get(img)
                if faces:
                    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    vec = target_face.normed_embedding if (hasattr(target_face, 'normed_embedding') and target_face.normed_embedding is not None) else l2_normalize(target_face.embedding)
                    anchors[member_id].append(vec)
            except Exception:
                pass
        print(f"  ✓ [{m['eng']}]: 기준 앵커 {len(anchors[member_id])}개 확보")

    return anchors

def smart_cleanup_database(insight_app, execute=False):
    import cv2

    anchors = load_all_member_anchors(insight_app)

    stats = {
        "deleted_no_face": 0,
        "moved_trash_other_member": 0,
        "moved_review_ambiguous": 0,
        "kept_valid": 0,
        "renamed_total": 0
    }

    print("\n==========================================================")
    print("  SMART 512-D DATABASE PURIFICATION & SORTING")
    print("==========================================================")

    for m in MEMBERS:
        member_id = m["id"]
        eng_name = m["eng"]
        member_dir = os.path.join(MEMBERS_DIR, member_id)

        if not os.path.exists(member_dir):
            continue

        trash_dir = os.path.join(member_dir, "_trash")
        review_dir = os.path.join(member_dir, "_review")

        if execute:
            os.makedirs(trash_dir, exist_ok=True)
            os.makedirs(review_dir, exist_ok=True)

        image_files = [
            f for f in sorted(os.listdir(member_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            and not f.startswith('.') and not f.startswith('_')
        ]

        print(f"\n[+] [{eng_name}] 총 {len(image_files)}장 정밀 분류 중...")

        valid_files_to_keep = []

        for fname in image_files:
            fpath = os.path.join(member_dir, fname)

            try:
                img = cv2.imread(fpath)

                # 1. 파일 손상 또는 얼굴 미감지 -> 영구 삭제
                if img is None:
                    print(f"  [DELETE] 손상된 파일: {fname}")
                    if execute:
                        os.remove(fpath)
                    stats["deleted_no_face"] += 1
                    continue

                faces = insight_app.get(img)
                if not faces or len(faces) == 0:
                    print(f"  [DELETE] 얼굴 미감지 (배경 이미지): {fname}")
                    if execute:
                        os.remove(fpath)
                    stats["deleted_no_face"] += 1
                    continue

                # 가장 큰 얼굴 특징점 추출
                target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                query_vec = target_face.normed_embedding if (hasattr(target_face, 'normed_embedding') and target_face.normed_embedding is not None) else l2_normalize(target_face.embedding)

                # 7명 전체 멤버와의 평균 Cosine Distance 계산
                member_distances = {}
                for target_mid, ref_vecs in anchors.items():
                    if not ref_vecs:
                        continue
                    dists = [calc_cosine_distance(query_vec, rvec) for rvec in ref_vecs]
                    dists.sort()
                    avg_k_dist = sum(dists[:5]) / min(5, len(dists))
                    member_distances[target_mid] = avg_k_dist

                if not member_distances:
                    valid_files_to_keep.append(fpath)
                    continue

                # 가장 유사한 멤버 판별
                best_match_id = min(member_distances, key=member_distances.get)
                best_dist = member_distances[best_match_id]
                own_dist = member_distances.get(member_id, 1.0)

                # 2. 다른 멤버의 얼굴로 판명된 경우 -> _trash 이동
                if best_match_id != member_id and best_dist < 0.38 and best_dist < (own_dist - 0.08):
                    print(f"  [TRASH] 타 멤버({best_match_id.upper()}) 얼굴 감지: {fname} (dist: {best_dist:.3f})")
                    if execute:
                        shutil.move(fpath, os.path.join(trash_dir, fname))
                    stats["moved_trash_other_member"] += 1

                # 3. 7명 멤버 어디에도 속하지 않는 백댄서/애매한 인물 -> _review 이동
                elif own_dist > 0.44 and best_dist > 0.42:
                    print(f"  [REVIEW] 백댄서/애매한 인물 (검토 필요): {fname} (dist: {own_dist:.3f})")
                    if execute:
                        shutil.move(fpath, os.path.join(review_dir, fname))
                    stats["moved_review_ambiguous"] += 1

                # 4. 검증 통과 (본인 얼굴)
                else:
                    valid_files_to_keep.append(fpath)
                    stats["kept_valid"] += 1

            except Exception as e:
                print(f"  [!] 처리 오류 ({fname}): {e}")

        # 4. 살아남은 정상 이미지 순차 정렬 및 리네이밍 (image_001.jpg, image_002.jpg...)
        if execute and valid_files_to_keep:
            print(f"  [SORT] [{eng_name}] 검증 통과 {len(valid_files_to_keep)}장 순차 파일명 정렬 중...")
            
            # 임시 리네이밍 (충돌 방지)
            temp_paths = []
            for idx, old_path in enumerate(valid_files_to_keep, 1):
                ext = os.path.splitext(old_path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                    ext = '.jpg'
                temp_name = f"__tmp_{idx:04d}{ext}"
                temp_path = os.path.join(member_dir, temp_name)
                try:
                    os.rename(old_path, temp_path)
                    temp_paths.append((temp_path, ext))
                except Exception as e:
                    print(f"    [!] 임시 파일명 변경 실패: {e}")

            # 최종 정렬 리네이밍
            for idx, (tmp_path, ext) in enumerate(temp_paths, 1):
                final_name = f"image_{idx:03d}{ext}"
                final_path = os.path.join(member_dir, final_name)
                try:
                    os.rename(tmp_path, final_path)
                    stats["renamed_total"] += 1
                except Exception as e:
                    print(f"    [!] 최종 파일명 변경 실패: {e}")

    print("\n==========================================================")
    print("  정화 및 재정렬 완료 요약")
    print(f"  - 삭제된 배경/무얼굴 이미지: {stats['deleted_no_face']}장")
    print(f"  - _trash 이동 (타 멤버 얼굴): {stats['moved_trash_other_member']}장")
    print(f"  - _review 이동 (애매한/백댄서): {stats['moved_review_ambiguous']}장")
    print(f"  - 검증 통과 (본인 얼굴): {stats['kept_valid']}장")
    print(f"  - 정렬 및 리네이밍 완료: {stats['renamed_total']}장")
    print("==========================================================")

def main():
    parser = argparse.ArgumentParser(description="NCT 127 Smart Face Database Cleaner & Serializer")
    parser.add_argument("--execute", action="store_true", help="실제로 파일 삭제, _trash/_review 이동 및 순차 정렬 실행")
    args = parser.parse_args()

    insight_app = init_insightface()
    if not insight_app:
        print("[!] InsightFace 엔진을 로드할 수 없어 작업을 취소합니다.")
        return

    smart_cleanup_database(insight_app, execute=args.execute)

    if args.execute:
        print("\n[*] descriptors.json 및 members.json 데이터베이스 자동 재빌드 중...")
        try:
            import build_descriptors
            build_descriptors.build()
        except Exception as e:
            print(f"[!] 재빌드 오류: {e}")

if __name__ == "__main__":
    main()


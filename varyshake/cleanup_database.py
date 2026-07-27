"""
NCT 127 Face Database Garbage Cleaner v1.0
===========================================
멤버별 이미지 폴더를 스캔하여 얼굴이 감지되지 않거나 품질이 낮은
쓰레기 이미지를 자동으로 _trash 폴더로 이동시킵니다.

사용법:
  python cleanup_database.py                    # 전체 멤버 스캔 (dry-run 미리보기)
  python cleanup_database.py --execute          # 실제로 쓰레기 이미지 이동
  python cleanup_database.py --member doyoung   # 특정 멤버만 스캔
  python cleanup_database.py --restore          # _trash에서 복원
"""

import os
import sys
import shutil
import argparse
import time

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

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

def init_insightface():
    """InsightFace 엔진 초기화"""
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

def scan_member_images(member_id, insight_app, blur_threshold=50.0, min_face_size=30):
    """멤버 폴더의 모든 이미지를 스캔하여 쓰레기 이미지 목록을 반환"""
    import cv2
    try:
        import face_recognition
    except ImportError:
        face_recognition = None

    member_dir = os.path.join(MEMBERS_DIR, member_id)
    if not os.path.exists(member_dir):
        return [], []

    image_files = [
        f for f in sorted(os.listdir(member_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        and not f.startswith('.')
    ]

    trash_list = []   # (filepath, reason) 쓰레기 이미지
    keep_list = []    # (filepath,) 정상 이미지

    for fname in image_files:
        fpath = os.path.join(member_dir, fname)

        try:
            img = cv2.imread(fpath)
            if img is None:
                trash_list.append((fpath, "파일 손상 (읽기 불가)"))
                continue

            h, w = img.shape[:2]

            # 1. 너무 작은 이미지 (< 50x50)
            if h < 50 or w < 50:
                trash_list.append((fpath, f"이미지 너무 작음 ({w}x{h})"))
                continue

            # 2. 선명도 검사
            blur = calc_blur_score(img)
            if blur < blur_threshold:
                trash_list.append((fpath, f"흐릿한 이미지 (blur={blur:.1f} < {blur_threshold})"))
                continue

            # 3. 얼굴 감지 검사
            face_found = False

            if insight_app is not None:
                try:
                    faces = insight_app.get(img)
                    for f in faces:
                        bbox = f.bbox.astype(int)
                        fw = bbox[2] - bbox[0]
                        fh = bbox[3] - bbox[1]
                        if fw >= min_face_size and fh >= min_face_size:
                            face_found = True
                            break
                except Exception:
                    pass

            if not face_found and face_recognition is not None:
                try:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb)
                    if locs:
                        face_found = True
                except Exception:
                    pass

            if not face_found:
                trash_list.append((fpath, "얼굴 미감지"))
                continue

            # 통과
            keep_list.append((fpath,))

        except Exception as e:
            trash_list.append((fpath, f"처리 오류: {str(e)[:50]}"))

    return trash_list, keep_list

def move_to_trash(trash_list, member_id):
    """쓰레기 이미지를 _trash 폴더로 이동"""
    member_dir = os.path.join(MEMBERS_DIR, member_id)
    trash_dir = os.path.join(member_dir, "_trash")
    os.makedirs(trash_dir, exist_ok=True)

    moved = 0
    for fpath, reason in trash_list:
        try:
            fname = os.path.basename(fpath)
            dest = os.path.join(trash_dir, fname)
            shutil.move(fpath, dest)
            moved += 1
        except Exception as e:
            print(f"  [!] 이동 실패: {fname} - {e}")

    return moved

def restore_from_trash(member_id):
    """_trash 폴더에서 이미지를 복원"""
    member_dir = os.path.join(MEMBERS_DIR, member_id)
    trash_dir = os.path.join(member_dir, "_trash")

    if not os.path.exists(trash_dir):
        print(f"  [-] {member_id}: _trash 폴더 없음")
        return 0

    files = [f for f in os.listdir(trash_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    restored = 0
    for fname in files:
        src = os.path.join(trash_dir, fname)
        dest = os.path.join(member_dir, fname)
        try:
            shutil.move(src, dest)
            restored += 1
        except Exception as e:
            print(f"  [!] 복원 실패: {fname} - {e}")

    # _trash 폴더가 비었으면 삭제
    if os.path.exists(trash_dir) and not os.listdir(trash_dir):
        os.rmdir(trash_dir)

    return restored

def empty_trash(member_id):
    """_trash 폴더를 영구 삭제"""
    member_dir = os.path.join(MEMBERS_DIR, member_id)
    trash_dir = os.path.join(member_dir, "_trash")

    if not os.path.exists(trash_dir):
        return 0

    files = [f for f in os.listdir(trash_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    count = len(files)
    shutil.rmtree(trash_dir, ignore_errors=True)
    return count

def main():
    parser = argparse.ArgumentParser(description="NCT 127 Face Database Garbage Cleaner")
    parser.add_argument("--member", type=str, default=None, help="특정 멤버만 스캔 (예: doyoung)")
    parser.add_argument("--execute", action="store_true", help="실제로 쓰레기 이미지를 _trash 폴더로 이동")
    parser.add_argument("--restore", action="store_true", help="_trash 폴더에서 이미지 복원")
    parser.add_argument("--empty-trash", action="store_true", help="_trash 폴더 영구 삭제")
    parser.add_argument("--blur", type=float, default=50.0, help="최소 선명도 점수 (기본값: 50.0)")
    args = parser.parse_args()

    print("==========================================================")
    print("  NCT 127 DATABASE GARBAGE CLEANER v1.0")
    print("==========================================================")

    # 복원 모드
    if args.restore:
        print("\n[*] 복원 모드: _trash 폴더에서 이미지를 복원합니다.\n")
        targets = [m for m in MEMBERS if args.member is None or m["id"] == args.member]
        total_restored = 0
        for m in targets:
            restored = restore_from_trash(m["id"])
            if restored > 0:
                print(f"  [v] {m['eng']}: {restored}장 복원 완료")
            total_restored += restored
        print(f"\n총 {total_restored}장 복원 완료!")
        return

    # 영구 삭제 모드
    if args.empty_trash:
        print("\n[*] 영구 삭제 모드: _trash 폴더를 완전히 삭제합니다.\n")
        targets = [m for m in MEMBERS if args.member is None or m["id"] == args.member]
        total_deleted = 0
        for m in targets:
            deleted = empty_trash(m["id"])
            if deleted > 0:
                print(f"  [v] {m['eng']}: {deleted}장 영구 삭제 완료")
            total_deleted += deleted
        print(f"\n총 {total_deleted}장 영구 삭제 완료!")
        return

    # 스캔 모드
    import cv2
    print("\n[*] InsightFace AI 엔진 초기화 중...")
    insight_app = init_insightface()

    if insight_app:
        print("[OK] InsightFace 512D 엔진 로드 완료\n")
    else:
        print("[!] InsightFace 사용 불가. face_recognition fallback 사용\n")

    targets = [m for m in MEMBERS if args.member is None or m["id"] == args.member]

    grand_total_trash = 0
    grand_total_keep = 0
    all_trash = {}

    for m in targets:
        member_id = m["id"]
        eng_name = m["eng"]

        print(f"[+] [{eng_name}] 스캔 중...")
        trash_list, keep_list = scan_member_images(member_id, insight_app, args.blur)

        all_trash[member_id] = trash_list
        grand_total_trash += len(trash_list)
        grand_total_keep += len(keep_list)

        if trash_list:
            print(f"  [!] 쓰레기 이미지: {len(trash_list)}장 발견")
            for fpath, reason in trash_list[:5]:
                print(f"      - {os.path.basename(fpath)}: {reason}")
            if len(trash_list) > 5:
                print(f"      ... 외 {len(trash_list) - 5}장")
        print(f"  [v] 정상 이미지: {len(keep_list)}장")

    print(f"\n==================================================")
    print(f"  스캔 결과 요약")
    print(f"  - 정상 이미지: {grand_total_keep}장")
    print(f"  - 쓰레기 이미지: {grand_total_trash}장")
    print(f"==================================================")

    if grand_total_trash == 0:
        print("\n[OK] 쓰레기 이미지가 없습니다. 데이터베이스가 깨끗합니다!")
        return

    if args.execute:
        print(f"\n[*] 쓰레기 이미지 {grand_total_trash}장을 _trash 폴더로 이동합니다...")
        total_moved = 0
        for m in targets:
            member_id = m["id"]
            trash_list = all_trash.get(member_id, [])
            if trash_list:
                moved = move_to_trash(trash_list, member_id)
                total_moved += moved
                print(f"  [v] {m['eng']}: {moved}장 -> _trash 이동 완료")

        print(f"\n[OK] 총 {total_moved}장이 _trash 폴더로 이동되었습니다.")
        print("     복원하려면: python cleanup_database.py --restore")
        print("     영구 삭제:  python cleanup_database.py --empty-trash")

        # 자동으로 descriptors.json 재빌드
        print("\n[*] descriptors.json 재빌드 중...")
        try:
            import build_descriptors
            build_descriptors.build()
        except Exception as e:
            print(f"[!] 재빌드 오류: {e}")
    else:
        print(f"\n[!] 위 {grand_total_trash}장은 미리보기입니다. 실제로 이동하려면:")
        print(f"    python cleanup_database.py --execute")

if __name__ == "__main__":
    main()

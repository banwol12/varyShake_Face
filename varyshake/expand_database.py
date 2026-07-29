import os
import glob
import urllib.request
import urllib.parse
import re
import ssl
import time

try:
    import face_recognition
except ImportError:
    print("CRITICAL ERROR: 'face-recognition' library is not installed.")
    print("Please install CMake and the face_recognition library by running:")
    print("  brew install cmake")
    print("  pip3 install face-recognition")
    exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Bypass SSL certificates verification (common on macOS)
ssl._create_default_https_context = ssl._create_unverified_context

# The 7 members specified by the user
MEMBERS = [
    {"id": "johnny", "eng": "Johnny", "kor": "쟈니"},
    {"id": "taeyong", "eng": "Taeyong", "kor": "태용"},
    {"id": "yuta", "eng": "Yuta", "kor": "유타"},
    {"id": "doyoung", "eng": "Doyoung", "kor": "도영"},
    {"id": "jaehyun", "eng": "Jaehyun", "kor": "재현"},
    {"id": "jungwoo", "eng": "Jungwoo", "kor": "정우"},
    {"id": "haechan", "eng": "Haechan", "kor": "해찬"}
]

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "members")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

import numpy as np

def align_face_img(rgb_img):
    """양쪽 눈 좌표 기반 얼굴 수평 회전 정렬 (Face Alignment)"""
    if not HAS_CV2:
        return rgb_img
    import cv2
    try:
        landmarks_list = face_recognition.face_landmarks(rgb_img)
        if not landmarks_list or 'left_eye' not in landmarks_list[0] or 'right_eye' not in landmarks_list[0]:
            return rgb_img
        landmarks = landmarks_list[0]
        left_eye_center = np.array(landmarks['left_eye']).mean(axis=0)
        right_eye_center = np.array(landmarks['right_eye']).mean(axis=0)
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                       (left_eye_center[1] + right_eye_center[1]) / 2.0)
        h, w = rgb_img.shape[:2]
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        return cv2.warpAffine(rgb_img, M, (w, h), flags=cv2.INTER_CUBIC)
    except Exception:
        return rgb_img

def l2_norm(vec):
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 0 else v

def load_member_anchors(member_id):
    """Loads all user-uploaded images in the member directory as 'golden truth' templates."""
    member_dir = os.path.join(BASE_DIR, member_id)
    if not os.path.exists(member_dir):
        return []
        
    all_files = glob.glob(os.path.join(member_dir, "images-*")) + glob.glob(os.path.join(member_dir, "images.*"))
    for f in os.listdir(member_dir):
        if not f.startswith("auto_") and not f.startswith("temp_") and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            full_path = os.path.join(member_dir, f)
            if full_path not in all_files:
                all_files.append(full_path)
                
    encodings = []
    print(f"Analyzing {len(all_files)} templates for '{member_id}'...")
    for path in all_files:
        try:
            image = face_recognition.load_image_file(path)
            aligned = align_face_img(image)
            face_encs = face_recognition.face_encodings(aligned)
            if not face_encs:
                face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(l2_norm(face_encs[0]))
            else:
                print(f"  Warning: No face detected in template: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Error reading template {os.path.basename(path)}: {e}")
            
    return encodings

def download_and_verify_candidates(member, anchor_encs, max_additions=15):
    member_id = member["id"]
    eng_name = member["eng"]
    kor_name = member["kor"]
    
    query = f"NCT {eng_name} solo portrait face"
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
    
    print(f"Searching web for similar faces of {eng_name}...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        print(f"  Failed to query search: {e}")
        return
        
    image_urls = re.findall(r'murl&quot;:&quot;(http[s]?://.*?)&quot;', html)
    print(f"  Found {len(image_urls)} candidates on search page. Beginning verification...")
    
    member_dir = os.path.join(BASE_DIR, member_id)
    success_count = 0
    download_idx = 100
    
    for img_url in image_urls:
        if success_count >= max_additions:
            break
            
        try:
            req_img = urllib.request.Request(img_url, headers=HEADERS)
            with urllib.request.urlopen(req_img, timeout=4) as img_resp:
                content_type = img_resp.headers.get("content-type", "")
                if "image" not in content_type:
                    continue
                    
                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"
                elif "gif" in content_type:
                    continue
                    
                temp_path = os.path.join(member_dir, f"temp_{download_idx}{ext}")
                with open(temp_path, "wb") as f:
                    f.write(img_resp.read())
                    
                try:
                    if HAS_CV2:
                        img_cv2 = cv2.imread(temp_path)
                        if img_cv2 is not None:
                            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                            if blur_score < 90.0:
                                print(f"  ✗ Discarded: Image too blurry (blur score: {blur_score:.1f} < 90.0).")
                                os.remove(temp_path)
                                continue

                    candidate_image = face_recognition.load_image_file(temp_path)
                    aligned_cand = align_face_img(candidate_image)
                    candidate_encs = face_recognition.face_encodings(aligned_cand)
                    if not candidate_encs:
                        candidate_encs = face_recognition.face_encodings(candidate_image)
                    
                    if len(candidate_encs) == 1:
                        cand_vec = l2_norm(candidate_encs[0])
                        distances = face_recognition.face_distance(anchor_encs, cand_vec)
                        min_dist = min(distances) if len(distances) > 0 else 1.0
                        if min_dist <= 0.05:
                            final_path = os.path.join(member_dir, f"auto_{download_idx}{ext}")
                            os.rename(temp_path, final_path)
                            anchor_encs.append(cand_vec)
                            success_count += 1
                            download_idx += 1
                            print(f"  ✓ Added match #{success_count}: {os.path.basename(final_path)} (distance: {min_dist:.3f})")
                        else:
                            os.remove(temp_path)
                    else:
                        os.remove(temp_path)
                except Exception:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception:
            pass
            
    print(f"Finished. Added {success_count} verified images for {eng_name}.")

def main():
    print("==========================================================")
    print("  NCT 127 DATABASE EXPANSION: BIOMETRIC VERIFIER v1.0     ")
    print("==========================================================")
    
    for member in MEMBERS:
        member_id = member["id"]
        eng_name = member["eng"]
        
        print(f"\nAnalyzing manual templates for {eng_name}...")
        anchors = load_member_anchors(member_id)
        
        if not anchors:
            print(f"No manual templates found in public/members/{member_id}/! Please add photos first.")
            continue
            
        print(f"Successfully loaded {len(anchors)} anchor templates.")
        # Scrape and add up to 15 verified images per member
        download_and_verify_candidates(member, anchors, max_additions=15)
        # Sleep 5 seconds between members to cool down search engines
        time.sleep(5.0)
        
    print("\nBiometric database expansion completed!")
    try:
        import build_descriptors
        build_descriptors.build()
    except Exception as e:
        print(f"Error building descriptors: {e}")

if __name__ == "__main__":
    main()

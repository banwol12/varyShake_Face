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

def load_member_anchors(member_id):
    """Loads all user-uploaded images in the member directory as 'golden truth' templates."""
    member_dir = os.path.join(BASE_DIR, member_id)
    if not os.path.exists(member_dir):
        return []
        
    # Search for all user uploaded files (files named images-x.jpeg or any other manual image files)
    all_files = glob.glob(os.path.join(member_dir, "images-*")) + glob.glob(os.path.join(member_dir, "images.*"))
    # Also include any files that are not named 'auto_' or 'temp_'
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
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
            else:
                print(f"  Warning: No face detected in template: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Error reading template {os.path.basename(path)}: {e}")
            
    return encodings

def download_and_verify_candidates(member, anchor_encs, max_additions=15):
    member_id = member["id"]
    eng_name = member["eng"]
    kor_name = member["kor"]
    
    # Query Bing Images (NCT + English Name + portrait)
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
    download_idx = 100 # start at index 100 to avoid overwriting user files
    
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
                    
                # Analyze the face in downloaded image
                try:
                    candidate_image = face_recognition.load_image_file(temp_path)
                    candidate_encs = face_recognition.face_encodings(candidate_image)
                    
                    if len(candidate_encs) == 1:
                        # Compare to all our anchor templates
                        # tolerance=0.48 is a strict distance threshold (standard is 0.60)
                        # This ensures the face is mathematically identical or highly similar
                        matches = face_recognition.compare_faces(anchor_encs, candidate_encs[0], tolerance=0.48)
                        match_count = sum(matches)
                        
                        # We require at least 30% of the manual templates to match
                        min_matches_required = max(1, int(len(anchor_encs) * 0.3))
                        
                        if match_count >= min_matches_required:
                            final_path = os.path.join(member_dir, f"auto_{download_idx}{ext}")
                            os.rename(temp_path, final_path)
                            print(f"  ✓ VERIFIED! Saved auto_{download_idx}{ext} (matched {match_count}/{len(anchor_encs)} templates)")
                            success_count += 1
                            download_idx += 1
                            time.sleep(0.35)
                        else:
                            print(f"  ✗ Discarded: Face similarity below strict matching requirements ({match_count}/{len(anchor_encs)} matched).")
                            os.remove(temp_path)
                    else:
                        print(f"  ✗ Discarded: Image contains {len(candidate_encs)} faces (must be exactly 1 face).")
                        os.remove(temp_path)
                except Exception as e_analysis:
                    # Clean up temp file on analysis failure
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception as e:
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
    # Regenerate the index JSON
    import sys
    os.system(f'"{sys.executable}" generate_db_json.py')

if __name__ == "__main__":
    main()

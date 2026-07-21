import os
import urllib.request
import urllib.parse
import re
import ssl
import time
import shutil

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

# Complete set of all folders that should exist. Any other folders will be deleted.
ALLOWED_IDS = {m["id"] for m in MEMBERS}

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "members")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_old_members():
    print("Cleaning up old and obsolete member folders...")
    if not os.path.exists(BASE_DIR):
        return
        
    for name in os.listdir(BASE_DIR):
        dir_path = os.path.join(BASE_DIR, name)
        if os.path.isdir(dir_path):
            if name not in ALLOWED_IDS:
                print(f"  Deleting obsolete member folder: {name}")
                shutil.rmtree(dir_path)

def download_member_images(member, max_images=40):
    member_id = member["id"]
    eng_name = member["eng"]
    kor_name = member["kor"]
    
    # Simple, high-density query
    query = f"NCT {eng_name}"
    
    print(f"\n==================================================")
    print(f"Downloading images for {eng_name} ({kor_name})...")
    print(f"Query: '{query}'")
    print(f"==================================================")
    
    dest_dir = os.path.join(BASE_DIR, member_id)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
    image_urls = []
    
    # Retry mechanism with 15-second cooldown if rate-limited
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')
            
            image_urls = re.findall(r'murl&quot;:&quot;(http[s]?://.*?)&quot;', html)
            
            if len(image_urls) >= 15:
                print(f"  Attempt {attempt + 1}: Success! Found {len(image_urls)} image URLs.")
                break
            else:
                print(f"  Attempt {attempt + 1}: Only found {len(image_urls)} URLs (likely rate-limited).")
                print("  Sleeping 15 seconds for cooldown...")
                time.sleep(15)
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            print("  Sleeping 15 seconds for cooldown...")
            time.sleep(15)
            
    if not image_urls:
        print(f"No image URLs found for {eng_name}. Trying fallback...")
        fallback_query = f"{kor_name} NCT"
        url = f"https://www.bing.com/images/search?q={urllib.parse.quote(fallback_query)}"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')
            image_urls = re.findall(r'murl&quot;:&quot;(http[s]?://.*?)&quot;', html)
            print(f"  Fallback found {len(image_urls)} URLs.")
        except Exception as e_fallback:
            print(f"  Fallback search failed: {e_fallback}")
            
    success_count = 0
    for idx, img_url in enumerate(image_urls):
        if success_count >= max_images:
            break
            
        print(f"[{success_count+1}/{max_images}] Fetching: {img_url}")
        try:
            req_img = urllib.request.Request(img_url, headers=HEADERS)
            with urllib.request.urlopen(req_img, timeout=5) as img_resp:
                content_type = img_resp.headers.get("content-type", "")
                if "image" not in content_type:
                    print("  Skipped: URL response is not an image.")
                    continue
                
                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"
                elif "gif" in content_type:
                    continue  # Skip gifs
                    
                file_path = os.path.join(dest_dir, f"image_{success_count+1}{ext}")
                with open(file_path, "wb") as f:
                    f.write(img_resp.read())
                print(f"  Saved to: public/members/{member_id}/image_{success_count+1}{ext}")
                success_count += 1
                
                # Sleep to prevent aggressive server hitting
                time.sleep(0.3)
        except Exception as e:
            print(f"  Failed: {e}")
            
    print(f"Completed downloads for {eng_name}. Successfully saved {success_count} images.")

def main():
    print("NCT 127 Member Image Scraper (Custom Bing Engine) Initializing...")
    print(f"Target Directory: {BASE_DIR}")
    
    # Clean old member folders (Taeil, Mark, Winwin)
    clean_old_members()
    
    for member in MEMBERS:
        download_member_images(member, max_images=40)
        # Sleep 10 seconds between members to fully clear rate limits
        print("Waiting 10 seconds before next query to prevent rate limits...")
        time.sleep(10.0)
        
    print("\nAll member image downloads and restructuring completed!")

if __name__ == "__main__":
    main()

import os
import requests

# Files to download from face-api.js weights directory
MODEL_FILES = [
    # SSD Mobilenet V1 (Default Face Detector)
    "ssd_mobilenetv1_model-weights_manifest.json",
    "ssd_mobilenetv1_model-shard1",
    "ssd_mobilenetv1_model-shard2",  # Added missing shard
    
    # Tiny Face Detector (Faster alternative for webcam)
    "tiny_face_detector_model-weights_manifest.json",
    "tiny_face_detector_model-shard1",
    
    # Face Landmark 68 (For finding face positions)
    "face_landmark_68_model-weights_manifest.json",
    "face_landmark_68_model-shard1",
    
    # Face Recognition (For generating 128-dimensional face embedding vector)
    "face_recognition_model-weights_manifest.json",
    "face_recognition_model-shard1",
    "face_recognition_model-shard2"
]

BASE_URL = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/"
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "models")

def main():
    print("face-api.js Models Downloader Initializing...")
    print(f"Destination Directory: {DEST_DIR}")
    os.makedirs(DEST_DIR, exist_ok=True)
    
    for filename in MODEL_FILES:
        url = f"{BASE_URL}{filename}"
        dest_path = os.path.join(DEST_DIR, filename)
        
        # If file already exists and is not empty, skip downloading
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"Already exists (skipped): {filename}")
            continue
            
        print(f"Downloading: {filename} from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Successfully downloaded: {filename}")
            else:
                print(f"  Failed to download: {filename} (HTTP Status {response.status_code})")
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
            
    print("\nModel downloads finished!")

if __name__ == "__main__":
    main()

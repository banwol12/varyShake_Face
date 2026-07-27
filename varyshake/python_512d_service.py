"""
NCT 127 512-D InsightFace ArcFace Real-time Recognition Service v1.0
=====================================================================
Runs InsightFace 512-D ArcFace model in RAM and provides high-speed,
SOTA face recognition for the web application & TouchDesigner.
"""

import os
import sys
import json
import base64
import time
import math
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTORS_JSON = os.path.join(BASE_DIR, "public", "descriptors.json")

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
    cos_sim = float(np.clip(dot, -1.0, 1.0))
    return 1.0 - cos_sim

print("==========================================================")
print("  INSIGHTFACE 512-D ARCFACE REAL-TIME SERVICE v1.0       ")
print("==========================================================")
print("[*] Initializing InsightFace 512D ArcFace Engine...")

import insightface
from insightface.app import FaceAnalysis

insight_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
insight_app.prepare(ctx_id=0, det_size=(640, 640))
print("[OK] InsightFace 512D ArcFace Engine loaded into RAM successfully!")

# Load pre-calculated 512-D descriptors
labeled_descriptors = []

def load_descriptors():
    global labeled_descriptors
    if os.path.exists(DESCRIPTORS_JSON):
        try:
            with open(DESCRIPTORS_JSON, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                labeled_descriptors = []
                for item in raw_data:
                    label = item["label"]
                    vecs = [np.array(d, dtype=np.float32) for d in item["descriptors"]]
                    labeled_descriptors.append({"label": label, "descriptors": vecs})
            print(f"[OK] Loaded 512-D descriptors for {len(labeled_descriptors)} members from descriptors.json!")
        except Exception as e:
            print(f"[!] Error loading descriptors.json: {e}")
    else:
        print("[!] descriptors.json not found!")

load_descriptors()

class ArcFaceServiceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress verbose HTTP request logging to keep console clean
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/reload':
            load_descriptors()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reloaded"}).encode('utf-8'))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "active", "engine": "InsightFace ArcFace 512-D"}).encode('utf-8'))

    def do_POST(self):
        if self.path == '/recognize-512d':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                image_b64 = data.get('image', '')
                top_k = data.get('topK', 3)
                
                if ',' in image_b64:
                    image_b64 = image_b64.split(',')[1]

                img_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_cv2 is None:
                    raise ValueError("Could not decode image")

                # Extract 512-D vector using InsightFace ArcFace
                faces = insight_app.get(img_cv2)
                
                if not faces:
                    res = {"matched": False, "label": "unknown", "distance": 1.0, "confidence": 0}
                else:
                    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    query_vec = target_face.normed_embedding if (hasattr(target_face, 'normed_embedding') and target_face.normed_embedding is not None) else l2_normalize(target_face.embedding)

                    # Top-K 512-D Cosine Distance Matcher
                    best_label = "unknown"
                    min_dist = 1.0

                    for ld in labeled_descriptors:
                        label = ld["label"]
                        dists = [calc_cosine_distance(query_vec, ref_vec) for ref_vec in ld["descriptors"]]
                        dists.sort()
                        k = min(top_k, len(dists))
                        if k > 0:
                            avg_dist = sum(dists[:k]) / k
                            if avg_dist < min_dist:
                                min_dist = avg_dist
                                best_label = label

                    confidence = int(max(0, (1 - min_dist) * 100))
                    res = {
                        "matched": True,
                        "label": best_label,
                        "distance": float(min_dist),
                        "confidence": confidence,
                        "vectorDim": len(query_vec)
                    }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

def run(port=5001):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ArcFaceServiceHandler)
    print(f"[OK] 512-D ArcFace Service running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run(5001)

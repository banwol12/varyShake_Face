import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBERS_DIR = os.path.join(BASE_DIR, "public", "members")
OUTPUT_JSON = os.path.join(BASE_DIR, "public", "members.json")

# Restrict to only the 7 active members requested by the user
MEMBERS = {
    "johnny": {"name": "Johnny", "korName": "쟈니"},
    "taeyong": {"name": "Taeyong", "korName": "태용"},
    "yuta": {"name": "Yuta", "korName": "유타"},
    "doyoung": {"name": "Doyoung", "korName": "도영"},
    "jaehyun": {"name": "Jaehyun", "korName": "재현"},
    "jungwoo": {"name": "Jungwoo", "korName": "정우"},
    "haechan": {"name": "Haechan", "korName": "해찬"}
}

def main():
    print("Generating members.json database file...")
    if not os.path.exists(MEMBERS_DIR):
        print(f"Directory {MEMBERS_DIR} does not exist.")
        return
        
    members_data = []
    
    # Iterate through folders in public/members
    for member_id in sorted(os.listdir(MEMBERS_DIR)):
        member_path = os.path.join(MEMBERS_DIR, member_id)
        if not os.path.isdir(member_path):
            continue
            
        # Ignore members not in the allowed list
        if member_id not in MEMBERS:
            continue
            
        # Scan for images
        images = []
        for filename in sorted(os.listdir(member_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                images.append(f"/members/{member_id}/{filename}")
                
        if images:
            info = MEMBERS[member_id]
            members_data.append({
                "id": member_id,
                "name": info["name"],
                "korName": info["korName"],
                "images": images
            })
            print(f"  {info['name']}: Found {len(images)} images.")
        else:
            print(f"  {member_id}: No images found (skipped).")
            
    # Write to members.json
    db = {"members": members_data}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
        
    print(f"Database file created successfully at: public/members.json")

if __name__ == "__main__":
    main()

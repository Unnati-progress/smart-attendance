import os
import cv2
import uuid
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_faces")
EMBED_DIR = os.path.join(BASE_DIR, "arcface_attendance")
ATTENDANCE_EXCEL = os.path.join(BASE_DIR, "Final names of cse.xlsx")

EMB_PATH = os.path.join(EMBED_DIR, "known_embeddings.npy")
NAME_PATH = os.path.join(EMBED_DIR, "known_names.npy")

os.makedirs(UNKNOWN_DIR, exist_ok=True)

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(1280, 1280))

known_embeddings = np.load(EMB_PATH)
known_names = np.load(NAME_PATH, allow_pickle=True)

# temporary memory for unknown embeddings
UNKNOWN_MEMORY = {}


def save_embeddings():
    np.save(EMB_PATH, known_embeddings)
    np.save(NAME_PATH, known_names)


def cosine_similarity(known_embs, face_emb):
    return np.dot(known_embs, face_emb) / (
        np.linalg.norm(known_embs, axis=1) * np.linalg.norm(face_emb)
    )


def mark_attendance_excel(present_names):
    df = pd.read_excel(ATTENDANCE_EXCEL)
    df.columns = df.columns.str.strip()

    name_column = "Names"

    if name_column not in df.columns:
        return f"Column {name_column} not found. Columns are {list(df.columns)}"

    today = datetime.now().strftime("%Y-%m-%d")

    if today not in df.columns:
        df[today] = "Absent"

    present_names = [str(x).strip() for x in present_names]

    df[today] = df[name_column].apply(
        lambda x: "Present" if str(x).strip() in present_names else "Absent"
    )

    df.to_excel(ATTENDANCE_EXCEL, index=False)
    return "Attendance marked successfully"


def process_attendance(image_path, threshold=0.40):
    global known_embeddings, known_names, UNKNOWN_MEMORY

    img = cv2.imread(image_path)

    if img is None:
        return {"error": "Image could not be read"}

    faces = face_app.get(img)

    results = []
    unknown_faces = []
    present_students = set()
    unknown_count = 0

    for face in faces:
        emb = face.embedding
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()

        h, w, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        status = "unknown"
        name = ""
        score = 0.0

        if len(known_embeddings) > 0:
            sims = cosine_similarity(known_embeddings, emb)
            best_idx = int(np.argmax(sims))
            score = float(sims[best_idx])

            if score >= threshold:
                name = str(known_names[best_idx]).strip()
                status = "known"
                present_students.add(name)

        if status == "unknown":
            unknown_count += 1
            unknown_id = f"unknown_{uuid.uuid4().hex[:8]}"
            name = f"Unknown_{unknown_count}"

            crop = img[y1:y2, x1:x2]
            crop_filename = f"{unknown_id}.jpg"
            crop_path = os.path.join(UNKNOWN_DIR, crop_filename)

            cv2.imwrite(crop_path, crop)

            UNKNOWN_MEMORY[unknown_id] = {
                "embedding": emb,
                "crop_path": crop_path
            }

            unknown_faces.append({
                "unknown_id": unknown_id,
                "label": name,
                "image_url": f"/unknown_faces/{crop_filename}",
                "bbox": [x1, y1, x2, y2],
                "score": round(score, 4)
            })

        results.append({
            "name": name,
            "status": status,
            "score": round(score, 4),
            "bbox": [x1, y1, x2, y2]
        })

    attendance_status = mark_attendance_excel(list(present_students))

    return {
        "total_students_detected": len(faces),
        "known_students_count": len(present_students),
        "unknown_students_count": unknown_count,
        "present_students": list(present_students),
        "attendance_status": attendance_status,
        "unknown_faces": unknown_faces,
        "results": results
    }


def add_unknown_to_dataset(unknown_id, student_name):
    global known_embeddings, known_names, UNKNOWN_MEMORY

    student_name = student_name.strip()

    if student_name == "":
        return {"message": "Skipped unknown student"}

    if unknown_id not in UNKNOWN_MEMORY:
        return {"error": "unknown_id not found. Run /predict again."}

    emb = UNKNOWN_MEMORY[unknown_id]["embedding"]
    crop_path = UNKNOWN_MEMORY[unknown_id]["crop_path"]

    student_folder = os.path.join(DATASET_DIR, student_name)
    os.makedirs(student_folder, exist_ok=True)

    save_name = f"{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    final_crop_path = os.path.join(student_folder, save_name)

    shutil.copy(crop_path, final_crop_path)

    known_embeddings = np.vstack([known_embeddings, emb])
    known_names = np.append(known_names, student_name)

    save_embeddings()

    del UNKNOWN_MEMORY[unknown_id]

    return {
        "message": "Student added successfully",
        "student_name": student_name,
        "saved_image": final_crop_path,
        "updated_embeddings": len(known_embeddings)
    }
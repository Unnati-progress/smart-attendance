from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os

from attendance_logic import process_attendance, add_unknown_to_dataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_images")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_faces")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

app.mount("/unknown_faces", StaticFiles(directory=UNKNOWN_DIR), name="unknown_faces")


class AddUnknownRequest(BaseModel):
    unknown_id: str
    name: str


@app.get("/")
def home():
    return {"message": "Smart Attendance Backend is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_attendance(file_path)
    return result


@app.post("/add-unknown")
def add_unknown(data: AddUnknownRequest):
    result = add_unknown_to_dataset(data.unknown_id, data.name)
    return result
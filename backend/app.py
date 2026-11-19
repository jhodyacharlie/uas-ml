# fastapi buat upload file, error, cors
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cv2
import json
import numpy as np
import tensorflow as tf

from cnn import bersihkan, UKURAN, MODEL_PATH, LABEL_PATH

app = FastAPI()

# Mengizinkan React mengakses backend
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Memuat model dan daftar identitas penulis
model = tf.keras.models.load_model(MODEL_PATH)

# Baca daftar nama penulis dari json biar sesuai hasil prediksi
with open(LABEL_PATH, "r", encoding="utf-8") as f:
  kelas = json.load(f)["kelas"]

@app.get("/api")
def home():
  return {"message": "Backend klasifikasi CNN aktif"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  # Membaca gambar yang dikirim React
  contents = await file.read()

  # Ubah bytes jadi array abis itu di decode jadi gambar grayscale
  img_array = np.frombuffer(contents, np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

  if img is None:
    raise HTTPException(
      status_code=400,
      detail="Gambar tidak valid"
    )

  # Preprocessing sama seperti saat training
  img_bersih = bersihkan(img)

  if img_bersih is None:
    raise HTTPException(
      status_code=400,
      detail="Tulisan tidak berhasil dideteksi"
    )

  img_resize = cv2.resize(
    img_bersih,
    (UKURAN, UKURAN),
    interpolation=cv2.INTER_AREA
  )

  # Bentuk gambar: (1, 128, 128, 1)
  X = img_resize[np.newaxis, ..., np.newaxis]

  # Prediksi menggunakan CNN
  probabilitas = model.predict(X, verbose=0)[0]

  # Ambil 3 prediksi dengan probabilitas tertinggi
  urutan = np.argsort(probabilitas)[::-1][:3]

  hasil = [
    {
      "penulis": kelas[int(i)],
      "confidence": float(probabilitas[i] * 100)
    }

    for i in urutan
  ]

  return {
    "prediksi": hasil[0]["penulis"],
    "confidence": hasil[0]["confidence"],
    "top_3": hasil
  }
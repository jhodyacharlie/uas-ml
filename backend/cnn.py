# Ngehitung kode hash dari gambar
# Kalo ada gambar yg bener2 sama, MD5 nya juga sama
# Yg sama itu dihapus biar gk ada data duplikat
import hashlib

# Simpan file ke json
import json

# Hitung waktu
import time

# Kelola tempat2 folder
from pathlib import Path

# Ngeproses gambar
import cv2

# Kelola data numerik ama array
import numpy as np

# Ngebuat model
import tensorflow as tf

# Bagi dataset jadi training sama uji
from sklearn.model_selection import train_test_split

# Buat evaluasi model
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  precision_score,
  recall_score,
  f1_score,
)

BASE_DIR = Path(__file__).resolve().parent

# Letakkan folder MLD di dalam folder backend.
DATASET_DIR = BASE_DIR / "MLD"

# Pilih jumlah penulisnya
JUMLAH_KELAS = 15

# Ukuran foto hasil preprocessing
UKURAN = 128

# Ukuran data uji
TEST_SIZE = 0.20

# Biar pengacakannya bisa dapet yg sama terus
RANDOM_STATE = 42

# Epoch = Model belajar brp kali
EPOCHS = 60

# Brp banyak gambar yg diproses di 1 batch
BATCH_SIZE = 32

# Folder buat model dan daftar nama penulis yang kepilih
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cnn.keras"
LABEL_PATH = MODEL_DIR / "kelas.json"

EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp"}


# Fungsi buat ngebersihin gambar tulisan
def bersihkan(img):
  if img is None:
    return None

  # Pisahin tulisan sama latar belakangnya
  # gaussian blur ngebuat gambar agak buram buat ngurangin noise
  # sisanya ngubah jadi gambar hitam putih murni
  # thresh otsu buat mencari batas terang gelap yang paling pas tanpa nebak2
  # thresh binary inv ngebalik warna jadi kertas hitam tinta putih
  # itu karena opencv ngeproses objek harus pake warna putih
  # indeks 1 buat ngambil gambar, 0 buat nilai ambang batas
  tinta = cv2.threshold(
    cv2.GaussianBlur(img, (3, 3), 0),
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )[1]

  h, w = tinta.shape


  # k_h dan k_v buat kernel bentuk persegi panjang buat deteksi garis horizontal ama vertikal
  k_h = cv2.getStructuringElement(
    cv2.MORPH_RECT, (min(max(40, w // 3), w), 1)
  )

  k_v = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, min(max(30, h // 2), h))
  )

  # Deteksi ama gabungin kedua garis
  garis = (
    cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k_h) | cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k_v)
  )

  # Hapus garis dari tinta
  tinta = cv2.bitwise_and(
    tinta,
    cv2.bitwise_not(
      cv2.dilate(garis, np.ones((3, 3), np.uint8))
    )
  )

  # Cari koordinat sisa tinta
  ys, xs = np.where(tinta > 0)

  # Kalo tintanya dikit, di skip
  if len(ys) < 50:
    return None

  # Perlebar area seleksi tulisan biar hurufnya gk kepotong kaku
  area = cv2.dilate(tinta, np.ones((5, 5), np.uint8)) > 0

  # Pertahanin tekstur asli di area tulisan abis itu lainnya jadi putih
  hasil = np.where(area, img, 255).astype(np.uint8)

  # Hitung koordinat batas atas bawah kiri kanan ama paddingnya 5-6 px
  y1 = max(int(ys.min()) - 5, 0)
  y2 = min(int(ys.max()) + 6, h)
  x1 = max(int(xs.min()) - 5, 0)
  x2 = min(int(xs.max()) + 6, w)

  # Potong gambar pas di area tulisan
  return hasil[y1:y2, x1:x2]



def ambil_folder_penulis():
  if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
      f"Folder dataset tidak ditemukan: {DATASET_DIR}\n"
      "Pastikan folder MLD berada di dalam folder backend."
    )

  # Urutin folder
  folder = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_dir() and "original" not in p.name.lower()
  )

  if JUMLAH_KELAS is not None:
    if JUMLAH_KELAS < 2:
      raise ValueError("JUMLAH_KELAS minimal 2.")
    folder = folder[:JUMLAH_KELAS]

  if len(folder) < 2:
    raise ValueError(
      "Minimal harus ada 2 folder penulis di dalam folder MLD."
    )

  return folder


def baca_dataset():
  folder_penulis = ambil_folder_penulis()

  print("Folder penulis yang digunakan:")
  for folder in folder_penulis:
    print("-", folder.name)

  data_gambar = []
  label_gambar = []
  hash_terpakai = set()

  jumlah_duplikat = 0
  jumlah_gagal_baca = 0
  jumlah_gagal_bersih = 0

  for folder in folder_penulis:
    label = folder.name

    # Ngeliat semua file di dalam folder
    for path_gambar in sorted(folder.rglob("*")):
      if not path_gambar.is_file():
        continue

      if "original" in path_gambar.name.lower():
        continue

      if path_gambar.suffix.lower() not in EKSTENSI_GAMBAR:
        continue

      try:
        isi_file = path_gambar.read_bytes()
      except OSError:
        jumlah_gagal_baca += 1
        continue

      # Hapus gambar yang isi filenya sama
      md5 = hashlib.md5(isi_file).hexdigest()
      if md5 in hash_terpakai:
        jumlah_duplikat += 1
        continue

      # Ubah byte file jadi matriks gambar grayscale
      img = cv2.imdecode(
        np.frombuffer(isi_file, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
      )

      if img is None:
        jumlah_gagal_baca += 1
        continue

      img_bersih = bersihkan(img)

      if img_bersih is None:
        jumlah_gagal_bersih += 1
        continue

      # Ganti ukuran jadi ukuran yang udh ditentuin
      img_resize = cv2.resize(
        img_bersih,
        (UKURAN, UKURAN),
        interpolation=cv2.INTER_AREA
      )

      # Simpen hash, data, ama label yg udh lewat semua uji
      hash_terpakai.add(md5)
      data_gambar.append(img_resize)
      label_gambar.append(label)

  if not data_gambar:
    raise ValueError(
      "Tidak ada gambar yang berhasil diproses. "
      "Periksa isi folder MLD dan format gambarnya."
    )

  # Ubah list gambar ama label jadi array
  X = np.array(data_gambar, dtype=np.uint8)
  label_asli = np.array(label_gambar)

  # Nama kelas diubah menjadi angka 0, 1, 2, dll
  kelas = sorted(np.unique(label_asli).tolist())
  peta_label = {nama: nomor for nomor, nama in enumerate(kelas)}
  y = np.array([peta_label[nama] for nama in label_asli], dtype=np.int32)

  print("\nRingkasan dataset:")
  print("Jumlah gambar:", len(X))
  print("Jumlah kelas yang berhasil dipakai:", len(kelas))
  print("Jumlah duplikat dilewati:", jumlah_duplikat)
  print("Gambar gagal dibaca:", jumlah_gagal_baca)
  print("Gambar gagal dibersihkan:", jumlah_gagal_bersih)

  print("\nJumlah gambar per penulis:")
  for nama in kelas:
    print(f"{nama}: {np.sum(label_asli == nama)}")

  # Cek tiap kelas punya minimal 2 gambar ato gk
  jumlah_per_kelas = np.bincount(y)
  kelas_kurang = [
    kelas[i] for i, jumlah in enumerate(jumlah_per_kelas)
    if jumlah < 2
  ]
  if kelas_kurang:
    raise ValueError(
      "Setiap penulis minimal harus memiliki 2 gambar. "
      f"Kelas yang kurang: {kelas_kurang}"
    )

  return X, y, kelas



def buat_cnn(jumlah_kelas):

  # Fungsi buat bikin 1 blok ekstraksi fitur
  def blok(jumlah_filter):
    return [
      # Deteksi pola gambar pake filter
      tf.keras.layers.Conv2D(
        jumlah_filter, 3, padding="same", activation="relu"
      ),
      # Bikin proses training lebih stabil dan cepet
      tf.keras.layers.BatchNormalization(),
      # Kecilin dimensi gambar biar fokus di fitur penting
      tf.keras.layers.MaxPooling2D(),
    ]

  # Susun model CNN dari input sampe output secara berurut
  model = tf.keras.Sequential([
    # nentuin ukuran input gambar grayscale
    tf.keras.layers.Input(shape=(UKURAN, UKURAN, 1)),
    # normalisasi nilai dari 0 - 255 jadi 0.0 - 1.0
    tf.keras.layers.Rescaling(1.0 / 255.0),

    # ekstraksi fitur dasar (garis/tepi) dengan 32 filter
    *blok(32),
    # ekstraksi pola menengah pake 64 filter
    *blok(64),
    # ekstraksi pola kompleks pake 128 filter
    *blok(128),
    # ekstraksi pola lebih kompleks pake 128 filter
    *blok(128),

    # Ubah matriks 2d dari konvolusi jadi vektor 1d
    tf.keras.layers.Flatten(),
    # Mengkombinasi fitur
    tf.keras.layers.Dense(256, activation="relu"),
    # Matiin 50% neuron acak pas training biar gk overfitting
    tf.keras.layers.Dropout(0.5),
    # probabilitas prediksi masing2 penulis
    tf.keras.layers.Dense(jumlah_kelas, activation="softmax"),
  ])

  # Konfigurasi model
  model.compile(
    # optimizer adam pake learning ratenya 0.001
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # fungsi loss buat klasifikasi multi kelas
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
  )

  return model


def main():
  X, y, kelas = baca_dataset()

  # Tambahkan channel grayscale: (jumlah_gambar, 128, 128, 1)
  X = X[..., np.newaxis]

  indeks = np.arange(len(y))

  indeks_train, indeks_test = train_test_split(
    indeks,
    test_size=TEST_SIZE,
    # seed acak biar pembagian data konsisten
    random_state=RANDOM_STATE,
    # memastikan proporsi uji sama latih itu sama
    stratify=y
  )

  X_train = X[indeks_train]
  X_test = X[indeks_test]
  y_train = y[indeks_train]
  y_test = y[indeks_test]

  model = buat_cnn(len(kelas))

  # Matiin lebih awal kalo akurasi validasi gk membaik
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    restore_best_weights=True
  )

  print("\nMulai training CNN...")

  # Mulai ngitung waktu buat ngelatih model
  waktu_mulai = time.perf_counter()

  model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
  )

  # Hasil akhir waktu ngelatih model
  waktu_training = time.perf_counter() - waktu_mulai

  # Prediksi pada data test
  probabilitas = model.predict(X_test, verbose=0)
  y_prediksi = np.argmax(probabilitas, axis=1)

  print("\n=== HASIL EVALUASI CNN ===")
  print("Accuracy :", round(accuracy_score(y_test, y_prediksi), 4))
  print(
    "Precision:",
    round(precision_score(
      y_test, y_prediksi, average="macro", zero_division=0
    ), 4)
  )
  print(
    "Recall   :",
    round(recall_score(
      y_test, y_prediksi, average="macro", zero_division=0
    ), 4)
  )
  print(
    "F1-score :",
    round(f1_score(
      y_test, y_prediksi, average="macro", zero_division=0
    ), 4)
  )

  print("\nClassification report:")
  print(classification_report(
    y_test,
    y_prediksi,
    labels=np.arange(len(kelas)),
    target_names=kelas,
    zero_division=0
  ))

  print(f"Waktu training CNN: {waktu_training:.2f} detik")

  # Simpan model dan urutan kelas untuk dipakai oleh backend API
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  model.save(MODEL_PATH)

  # Simpan urutan daftar nama penulis
  with open(LABEL_PATH, "w", encoding="utf-8") as file:
    json.dump({"kelas": kelas}, file, ensure_ascii=False, indent=4)

  print("\nModel berhasil disimpan:")
  print(MODEL_PATH)
  print("Daftar kelas disimpan:")
  print(LABEL_PATH)


if __name__ == "__main__":
  main()
import { useState, type ChangeEvent } from "react";

interface PredictResult {
  penulis: string;
  confidence: number;
}

interface PredictResponse {
  prediksi: string;
  confidence: number;
  top_3: PredictResult[];
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const pilihGambar = (e: ChangeEvent<HTMLInputElement>) => {
    const gambar = e.target.files?.[0];

    if (!gambar) return;

    if (!["image/jpeg", "image/png"].includes(gambar.type)) {
      setError("Gunakan gambar JPG atau PNG.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(gambar);
    setPreview(URL.createObjectURL(gambar));
    setResult(null);
    setError("");
  };

  const predictCNN = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediksi gagal.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Terjadi kesalahan."
      );
    } finally {
      setLoading(false);
    }
  };

  const deletePic = () => {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  };

  return (
    <main className="min-h-screen bg-[#141313] px-4 py-10">
      <div className="mx-auto max-w-2xl">

        <h1 className="text-3xl font-bold text-white mb-4">
          Klasifikasi Tulisan Tangan
        </h1>

        <div className=" rounded-2xl border border-white/8 bg-white/3 p-6 shadow-sm text-white">

          <p className="mb-4 font-semibold">
            Masukkan Gambar
          </p>

          <label className="flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-gray-200/10 hover:bg-white/5">
            <span className="font-semibold">
              Klik untuk memilih gambar
            </span>

            <span className="mt-1 text-sm">
              Format JPG atau PNG
            </span>

            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={pilihGambar}
              className="hidden"
            />
          </label>

          {preview && (
            <div className="mt-6 text-white">
              <p className="mb-3 font-semibold">
                Preview
              </p>

              <img
                src={preview}
                alt="Preview tulisan tangan"
                className="mx-auto max-h-80 rounded-xl border object-contain"
              />

              <p className="mt-2 text-center text-sm text-white/70">
                {file?.name}
              </p>

              <div className="mt-4 flex gap-3">
                <button
                  onClick={deletePic}
                  className="flex-1 rounded-lg border-2 border-red-500/40 py-3 font-semibold text-red-500 hover:bg-red-500/5 cursor-pointer"
                >
                  Hapus
                </button>

                <button
                  onClick={predictCNN}
                  disabled={loading}
                  className="flex-1 rounded-lg border-2 border-blue-500/40 hover:bg-blue-500/5 py-3 font-semibold text-white disabled:opacity-50 cursor-pointer"
                >
                  {loading ? "Memproses..." : "Prediksi"}
                </button>
              </div>
            </div>
          )}

          {error && (
            <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-600">
              {error}
            </p>
          )}

          {result && (
            <div className="rounded-2xl border border-orange-200/8 bg-[#241B17]/80 backdrop-blur-xl p-6 shadow-lg shadow-black/20 text-white mt-6">

              <p className="text-sm font-medium">
                Hasil
              </p>

              <h2 className="mt-2 text-2xl font-bold">
                {result.prediksi}
              </h2>

              <p className="mt-1 text-sm">
                Skor keyakinan: <span className="font-semibold">{result.confidence.toFixed(2)}%</span>
              </p>

                <p className="mb-3 font-semibold text-white mt-6">
                  3 Prediksi Teratas
                </p>

              {result.top_3.map((item, index) => (
                <div key={item.penulis} className="mb-3">
                  <div className="mb-1 flex justify-between text-sm">
                    <span className="font-medium">
                      {index + 1}. {item.penulis}
                    </span>

                    <span className="">
                      {item.confidence.toFixed(2)}%
                    </span>
                  </div>

                  <div className="h-2 overflow-hidden rounded-full bg-black/30">
                    <div
                      className="h-full rounded-full bg-green-500"
                      style={{
                        width: `${item.confidence}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

export default App;

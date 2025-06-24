# SVD-Based Image Compression

Proyek ini mengimplementasikan kompresi citra digital menggunakan metode **Singular Value Decomposition (SVD)**. Sistem dirancang untuk mengompres gambar berwarna atau grayscale, menghitung efisiensi kompresi secara matematis, serta mengevaluasi kualitas hasil kompresi menggunakan metrik **MSE** dan **PSNR**.

## 📁 Struktur Direktori

```

src/
├── main.py                  # Program utama (interface / CLI / eksekusi kompresi)
├── utils.py                 # Modul fungsi kompresi, evaluasi, dan visualisasi
├── requirements.txt         # Daftar dependensi proyek

````

## ⚙️ Dependensi

Proyek ini dibangun menggunakan Python dengan beberapa pustaka eksternal:

- `streamlit`
- `numpy`
- `Pillow`
- `matplotlib`

## ▶️ Cara Menjalankan Aplikasi

Pastikan semua dependensi telah terinstal dengan perintah:

```bash
pip install -r requirements.txt
````

Untuk membuka antarmuka interaktif berbasis web menggunakan Streamlit, gunakan salah satu perintah berikut:
```bash
streamlit run src/main.py
````
atau jika menggunakan Python module flag:
```bash
python -m streamlit run src/main.py
````
Setelah dijalankan, aplikasi akan terbuka di browser secara otomatis pada alamat lokal (biasanya http://localhost:8501).

NB: Jika folder src/ berada di lokasi berbeda, sesuaikan path sesuai struktur direktori proyek.

## Fitur Utama

* Kompresi citra menggunakan dekomposisi SVD per channel RGB.
* Mendukung gambar grayscale dan berwarna.
* Penentuan jumlah singular values (`k`) berdasarkan rasio kompresi.
* Visualisasi hasil: gambar asli vs. hasil kompresi.
* Evaluasi kualitas hasil kompresi menggunakan:

  * **Mean Squared Error (MSE)**
  * **Peak Signal-to-Noise Ratio (PSNR)**
* Statistik kompresi seperti:

  * Jumlah elemen data sebelum dan sesudah
  * Persentase penghematan ruang
  * Nilai `k` rata-rata dan presentasenya terhadap maksimum

## Tujuan Proyek

Proyek ini bertujuan untuk menunjukkan penerapan konsep **Aljabar Linier**, khususnya **dekomposisi matriks**, dalam konteks pemrosesan citra digital dan efisiensi penyimpanan data.

## Catatan

* Kompresi tidak selalu menghasilkan file dengan ukuran lebih kecil secara fisik (tergantung format penyimpanan seperti PNG), namun secara matematis ukuran matriks penyimpanan berkurang signifikan.
* Nilai `k` yang lebih kecil menghasilkan kompresi lebih tinggi, tetapi menurunkan kualitas visual.

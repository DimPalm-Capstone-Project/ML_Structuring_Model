# Sistem Pengenalan Telapak Tangan 🖐️

Sistem pengenalan biometrik berbasis telapak tangan yang menggunakan ROI box extraction dengan MediaPipe. Sistem ini diintegrasikan dengan arsitektur Siamese Neural Network untuk melakukan embedding dan recognition telapak tangan.

## 📑 Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Visualisasi Preprocessing](#visualisasi-preprocessing)
- [Tahapan Preprocessing](#tahapan-preprocessing)
- [Arsitektur Model](#arsitektur-model)
- [Hasil Model](#hasil-model)
- [Cara Instalasi](#cara-instalasi)
- [Limitasi](#limitasi)
- [Best Practice](#best-practice)
- [Pengembangan ke Depan](#pengembangan-ke-depan)

## 🚀 Fitur Utama

- Deteksi landmark tangan otomatis menggunakan MediaPipe
- Ekstraksi ROI (Region of Interest) dinamis dengan auto-rotate
- Quality check otomatis untuk kecerahan dan ketajaman gambar
- Preprocessing komprehensif dengan visualisasi di setiap tahap
- Data augmentation otomatis untuk peningkatan dataset
- Pengenalan telapak tangan menggunakan Siamese Neural Network
- Sistem embedding database untuk penyimpanan dan pencocokan
- Visualisasi proses untuk keperluan debugging dan analisis

## 📊 Visualisasi Preprocessing

### 1. Deteksi Landmark dan ROI Extraction

### 2. Quality Check

### 3. Image Enhancement

### 4. Data Augmentation

<img src="./result/PreprocessedVisual.png" width="800">

## 🔄 Tahapan Preprocessing

1. **Input dan Validasi Awal**

   - Resize gambar ke ukuran standar (1280x720)
   - Validasi kualitas gambar input

2. **Deteksi dan Ekstraksi**

   - Deteksi landmark tangan dengan MediaPipe
   - Ekstraksi ROI telapak tangan menggunakan dynamic box
   - Auto-rotate untuk standardisasi orientasi

3. **Quality Check**

   - Evaluasi kecerahan (threshold: 100-180)
   - Analisis ketajaman (minimum threshold: 8)
   - Visualisasi metrik kualitas

4. **Image Enhancement**

   - Konversi ke grayscale
   - Penghilangan bayangan menggunakan teknik morfologi
   - Normalisasi kontras dengan CLAHE
   - Gamma correction

5. **Standardisasi**
   - Resize ke ukuran 128x128
   - Normalisasi nilai pixel

## 🏗️ Arsitektur Model

- **Base Network:**
  ```
  Input (128x128x1)
  ↓
  Conv2D(64, 10x10) + ReLU
  MaxPooling2D
  ↓
  Conv2D(128, 7x7) + ReLU
  MaxPooling2D
  ↓
  Conv2D(128, 4x4) + ReLU
  MaxPooling2D
  ↓
  Conv2D(256, 4x4) + ReLU
  ↓
  Flatten
  Dense(4096, sigmoid)
  ```

## 📈 Hasil Model

### 1. Embedding Visualization

```
[Gambar 5: Embedding Space]
- Visualisasi t-SNE dari embedding telapak tangan
- Clustering berdasarkan identitas
```

### 2. Recognition Results

```
[Gambar 6: Recognition Examples]
Case 1: Match Found
- Query Image
- Best Match
- Similarity Score: 0.92

Case 2: No Match
- Query Image
- Closest Match
- Similarity Score: 0.45 (Below Threshold)
```

### 3. Model Metrics

- Accuracy: XX%
- Precision: XX%
- Recall: XX%
- F1-Score: XX%

## ⚙️ Cara Instalasi

1. Clone repository:

```bash
git clone https://github.com/DimPalm-Capstone-Project/ML_Structuring_Model.git
cd ML_Structuring_Model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements:

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
tensorflow>=2.12.0
matplotlib>=3.7.0
```

## ⚠️ Limitasi

1. **Pencahayaan**

   - Membutuhkan pencahayaan yang cukup (brightness: 100-180)
   - Sensitif terhadap bayangan berlebih

2. **Posisi Tangan**

   - Telapak tangan harus terbuka penuh
   - Sudut pengambilan harus relatif tegak lurus

3. **Hardware**
   - Membutuhkan kamera dengan minimal 720p
   - Proses preprocessing membutuhkan CPU/RAM yang memadai

## 💡 Best Practice

1. **Pengambilan Gambar**

   - Gunakan pencahayaan merata
   - Posisikan tangan tegak lurus dengan kamera
   - Pastikan telapak tangan terbuka penuh
   - Hindari background yang kompleks

2. **Preprocessing**

   - Selalu cek visualisasi quality check
   - Gunakan augmentasi data untuk variasi dataset
   - Simpan hasil intermediate untuk debugging

3. **Deployment**
   - Implementasikan sistem caching untuk embedding
   - Gunakan batch processing untuk optimasi
   - Monitor resource usage secara berkala

## 🔮 Pengembangan ke Depan

1. **Peningkatan Akurasi**

   - Implementasi multi-scale feature extraction
   - Integrasi dengan teknik attention
   - Pengembangan data augmentation yang lebih advanced

2. **Optimasi**

   - Implementasi model quantization
   - Pengembangan lite version untuk mobile
   - Optimasi preprocessing pipeline

3. **Fitur Tambahan**

   - Integrasi dengan sistem anti-spoofing
   - Pengembangan GUI untuk visualisasi real-time
   - Implementasi sistem multi-modal recognition

4. **Keamanan**
   - Pengembangan enkripsi untuk embedding database
   - Implementasi secure API endpoints
   - Pengembangan audit trail system

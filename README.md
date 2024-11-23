# Sistem Pengenalan Telapak Tangan

Sistem pengenalan biometrik berbasis telapak tangan menggunakan arsitektur Siamese Neural Network dengan kemampuan embedding dan recognition.

## Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Arsitektur](#arsitektur)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Performa](#performa)
- [Struktur Proyek](#struktur-proyek)
- [Limitasi](#limitasi)
- [Best Practices](#best-practices)

## Fitur Utama
- Pengenalan telapak tangan menggunakan Siamese Neural Network untuk mendeteksi kemiripan dan Identitas
- Dukungan untuk pengujian batch
- Analisis similarity dengan threshold yang dapat dikonfigurasi
- Top-5 matches untuk setiap gambar
- Penambahan data dinamis tanpa retraining
- Model sudah dikonversi ke TFLite untuk mobile

## Arsitektur
- **Model**: Siamese Neural Network
- **Input Size**: 128x128 piksel (grayscale)
- **Embedding Size**: 4096 dimensi
- **Similarity Metric**: L1 Distance
- **Format Model**:
  - TensorFlow (.h5)
  - TFLite (.tflite)

## Instalasi

### Dependensi
```bash
pip install tensorflow==2.15.1 opencv-python numpy
```

### Model Files
> **Penting**: File model tidak disertakan dalam repository karena ukurannya yang besar.
> 
> Sebelum menjalankan program, pastikan file model berikut sudah ada di direktori yang sesuai:
> - `src/models/palm_print_siamese_model.h5` (Model TensorFlow)
> - `src/models/palm_print_model.tflite` (Model TFLite)

### Struktur Direktori
```
palm-recognizition/
├── src/
│   ├── models/
│   │   ├── palm_print_siamese_model.h5
│   │   └── palm_print_model.tflite
│   ├── test_trained_images.py
│   ├── test_new_image.py
│   └── add_and_test_new_data.py
└── data/
    ├── raw/     # Data training
    └── test/    # Gambar untuk pengujian
```

##  Penggunaan

### 1. Pengujian Data Training
```bash
python src/test_trained_images.py
```
Validasi performa model pada dataset training dengan multiple threshold.

### 2. Pengujian Gambar Baru
```bash
python src/test_new_image.py
```
Menguji gambar baru terhadap database yang ada.

### 3. Menambah Data Baru
```bash
python src/add_and_test_new_data.py
```
Menambahkan data baru ke database dengan ekstraksi embedding otomatis.

## Performa

### Metrik Dataset Training
| Metrik | Nilai |
|--------|-------|
| Akurasi | 95.31% |
| Precision | 0.97 |
| Recall | 0.95 |
| F1-Score | 0.95 |

### Threshold Pengujian
| Threshold | Penggunaan |
|-----------|------------|
| 0.85 | Toleran, cocok untuk pengujian awal |
| 0.90 | Seimbang antara akurasi dan recall |
| 0.95 | Ketat, meminimalkan false positives |

##  Struktur Proyek

### Komponen Utama
1. **Preprocessing Gambar**
   - Konversi ke grayscale
   - Resize (128x128)
   - Normalisasi nilai piksel

2. **Ekstraksi Fitur**
   - Embedding dimensi 4096
   - Normalisasi embedding
   - Komputasi similarity (L1)

3. **Manajemen Database**
   - Format JSON
   - Penambahan data dinamis
   - Backup otomatis

## Limitasi
1. Memerlukan kualitas gambar yang baik
2. Sensitif terhadap pencahayaan
3. Performa optimal pada gambar telapak tangan yang jelas
4. Threshold perlu disesuaikan berdasarkan use case

## Best Practices
1. Gunakan gambar dengan resolusi minimal 128x128
2. Harus menggunakan pencahayaan yang cukup
3. Telapak tangan harus terlihat jelas
4. Lakukan pengujian dengan multiple threshold
5. Backup database secara berkala

## Catatan Pengembangan
- Model dapat dikonversi ke TFLite untuk deployment mobile
- Mendukung penambahan data baru tanpa retraining
- Sistem backup otomatis untuk database
- Logging komprehensif untuk debugging


## 📄 Lisensi
[MIT License](LICENSE)
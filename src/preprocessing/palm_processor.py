"""
File: palm_preprocessing.py

Flow Preprocessing:
1. Input gambar -> Deteksi landmark tangan dengan MediaPipe
2. Ekstrak ROI (Region of Interest) telapak tangan dan lakukan croping
3. Konversi ke grayscale dan hilangkan bayangan 
4. Resize ke ukuran standard
5. Augmentasi data (rotasi, scaling, brightness, contrast)
6. Simpan ke dynamic folder

Library yang digunakan:
- cv2: Digunakan untuk operasi pengolahan citra seperti:
  - Membaca/menulis gambar
  - Konversi color space (RGB/BGR/Grayscale)
  - Operasi morphologi untuk menghilangkan bayangan
  - Normalisasi dan enhance contrast

- numpy: Digunakan untuk:
  - Operasi array pada citra
  - Kalkulasi statistik (mean, max, min)
  - Manipulasi matriks untuk transformasi gambar

- mediapipe: Digunakan untuk:
  - Deteksi landmark tangan
  - Mendapatkan koordinat titik-titik penting telapak tangan

- typing: Digunakan untuk:
  - Type hints parameter dan return value
  - Meningkatkan readability dan maintainability kode

- os & glob: Digunakan untuk:
  - Operasi filesystem (buat folder, simpan file)
  - Pattern matching untuk mencari file

- logging: Digunakan untuk:
  - Tracking proses preprocessing  
  - Debugging dan error handling
"""

# Import library yang dibutuhkan
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional, Union
import os
from glob import glob
import logging

# Setup logging untuk tracking proses & debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PalmPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Inisialisasi preprocessing telapak tangan.

        Flow:
        1. Set ukuran target output (128,128)
        2. Setup MediaPipe untuk deteksi landmark tangan
        3. Konfigurasi detektor (mode statis, 1 tangan, confidence 50%)

        Args:
            target_size: Tuple (width, height) untuk ukuran akhir gambar
        """
        # Simpan ukuran target untuk resize di akhir preprocessing
        self.target_size = target_size

        # Inisialisasi MediaPipe hands untuk deteksi landmark
        self.mp_hands = mp.solutions.hands

        # Utilitas untuk visualisasi landmark (opsional, untuk debugging)
        self.mp_drawing = mp.solutions.drawing_utils

        # Buat instance detektor tangan dengan konfigurasi optimal
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # Mode gambar statis
            max_num_hands=1,  # Deteksi maksimal 1 tangan
            min_detection_confidence=0.5,  # Threshold confidence 50%
        )

    def preprocess_image(
        self, image_path: Union[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Fungsi utama untuk preprocessing gambar telapak tangan.

        Flow:
        1. Load & validasi gambar (dari path/array)
        2. Deteksi landmark -> cari posisi tangan
        3. Ekstrak ROI -> potong area telapak tangan
        4. Convert grayscale -> hapus bayangan
        5. Resize -> ukuran standard

        Args:
            image_path: Path file gambar atau numpy array

        Returns:
            array: Gambar hasil preprocessing atau None jika gagal
        """
        try:
            # Step 1: Load gambar
            if isinstance(image_path, str):
                image = cv2.imread(image_path)  # Baca dari file
                if image is None:
                    raise ValueError(f"Gagal baca gambar: {image_path}")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert ke RGB
            else:
                image_rgb = image_path  # Gunakan array langsung

            # Step 2: Deteksi landmark tangan
            landmarks = self._detect_hand_landmarks(image_rgb)
            if landmarks is None:
                logger.warning("Tidak ada tangan terdeteksi")
                return None

            # Step 3: Ekstrak area telapak tangan
            roi, _ = self._extract_palm_roi(image_rgb, landmarks)
            if roi is None:
                logger.warning("Gagal ekstrak area telapak")
                return None

            # Step 4: Proses ke grayscale
            processed_roi = self._convert_to_grayscale(roi)
            if processed_roi is None:
                logger.warning("Gagal konversi grayscale")
                return None

            # Step 5: Resize ke ukuran standard
            final_image = self._resize_roi(processed_roi)
            if final_image is None:
                logger.warning("Gagal resize gambar")
                return None

            return final_image

        except Exception as e:
            logger.error(f"Error dalam preprocessing: {str(e)}")
            return None

    def _detect_hand_landmarks(self, image_rgb: np.ndarray) -> Optional[object]:
        """
        Deteksi landmark/titik penting pada tangan menggunakan MediaPipe.

        Flow:
        1. Proses gambar dengan MediaPipe Hands
        2. Cek hasil deteksi landmark
        3. Ambil landmark tangan pertama (jika ada)

        Args:
            image_rgb: Array gambar format RGB

        Returns:
            object: Landmark tangan pertama yang terdeteksi
            None: Jika tidak ada tangan terdeteksi
        """
        # Proses gambar untuk deteksi landmark
        results = self.hands.process(image_rgb)

        # Cek & return landmark tangan pertama jika ada
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]  # Ambil hanya tangan pertama

        return None  # Return None jika tidak ada tangan

    def _extract_palm_roi(
        self, image_rgb: np.ndarray, hand_landmarks: object
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Ekstrak Region of Interest (ROI) dari area telapak tangan.

        Flow:
        1. Validasi landmark tangan
        2. Ambil koordinat titik-titik pangkal jari (landmark 1,5,9,13,17)
        3. Hitung pusat & ukuran area telapak
        4. Tentukan koordinat kotak ROI
        5. Potong gambar sesuai ROI

        Args:
            image_rgb: Array gambar RGB
            hand_landmarks: Objek landmark dari MediaPipe

        Returns:
            Tuple berisi:
            - ROI gambar telapak tangan
            - Koordinat ROI (x1,y1,x2,y2)
        """
        # Validasi input
        if hand_landmarks is None:
            return None, None

        # Ambil dimensi gambar
        h, w, _ = image_rgb.shape

        # Index landmark untuk pangkal jari (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
        palm_center_indices = [1, 5, 9, 13, 17]  # Landmark pangkal tiap jari
        palm_points = []

        # Kumpulkan koordinat pangkal jari
        for idx in palm_center_indices:
            landmark = hand_landmarks.landmark[idx]
            # Konversi koordinat relatif (0-1) ke pixel
            x, y = int(landmark.x * w), int(landmark.y * h)
            palm_points.append((x, y))

        # Hitung pusat telapak tangan
        center_x = int(np.mean([p[0] for p in palm_points]))
        center_y = (
            int(np.mean([p[1] for p in palm_points])) + 100
        )  # Offset 100px ke bawah

        # Hitung ukuran area telapak
        palm_width = max([p[0] for p in palm_points]) - min([p[0] for p in palm_points])
        palm_height = max([p[1] for p in palm_points]) - min(
            [p[1] for p in palm_points]
        )
        roi_size = int(max(palm_width, palm_height) * 0.8)  # Ambil 80% ukuran maksimal

        # Tentukan koordinat ROI dengan batas aman
        x1 = max(0, center_x - roi_size // 2)  # Batas kiri
        y1 = max(0, center_y - roi_size // 2)  # Batas atas
        x2 = min(w, x1 + roi_size)  # Batas kanan
        y2 = min(h, y1 + roi_size)  # Batas bawah

        # Pastikan ROI berbentuk persegi
        roi_size = min(x2 - x1, y2 - y1)
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        # Potong gambar sesuai ROI
        roi = image_rgb[y1:y2, x1:x2]

        return roi, (x1, y1, x2, y2)

    def _convert_to_grayscale(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Konversi ROI ke grayscale dan perbaikan kualitas gambar.

        Flow:
        1. Konversi RGB ke Grayscale
        2. Perbaikan kontras dengan CLAHE
        3. Hilangkan bayangan dengan morphology
        4. Normalisasi intensitas piksel
        5. Koreksi gamma

        Args:
            roi: Array gambar RGB area telapak

        Returns:
            array: Gambar grayscale yang telah diproses
            None: Jika input None
        """
        # Validasi input
        if roi is None:
            return None

        # Step 1: Konversi ke grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Step 2: Perbaikan kontras pertama
        clahe = cv2.createCLAHE(
            clipLimit=2.0,  # Batasi kontras untuk mengurangi noise
            tileGridSize=(8, 8),  # Ukuran grid untuk adaptif lokal
        )
        gray = clahe.apply(gray)

        # Step 3: Hilangkan bayangan
        # Dilasi untuk memperbesar area terang
        dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
        # Blur untuk estimasi background
        bg_img = cv2.medianBlur(dilated, 25)
        # Kurangkan background untuk hilangkan bayangan
        diff_img = 255 - cv2.absdiff(gray, bg_img)

        # Step 4: Perbaikan kontras kedua
        diff_img = clahe.apply(diff_img)

        # Step 5: Normalisasi nilai piksel ke range 10-245
        normalized = cv2.normalize(
            diff_img,
            None,
            alpha=10,  # Nilai minimum
            beta=245,  # Nilai maksimum
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )

        # Step 6: Koreksi gamma untuk perbaikan brightness
        gamma = 0.8  # Gamma < 1 membuat gambar lebih terang
        normalized = np.array(255 * (normalized / 255) ** gamma, dtype="uint8")

        return normalized

    def _resize_roi(self, enhanced_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Ubah ukuran ROI ke target size yang ditentukan.

        Flow:
        1. Validasi input
        2. Resize gambar ke ukuran target (default 128x128)

        Args:
            enhanced_roi: Array gambar yang sudah dienhance

        Returns:
            array: Gambar hasil resize sesuai target_size
            None: Jika input None

        Notes:
            - Menggunakan INTER_AREA karena bagus untuk pengecilan gambar
            - target_size sudah ditentukan di __init__ (default 128x128)
        """
        # Validasi input
        if enhanced_roi is None:
            return None

        # Resize dengan interpolasi INTER_AREA
        # INTER_AREA bagus untuk shrinking karena anti-aliasing
        return cv2.resize(
            enhanced_roi,
            self.target_size,  # (width, height) dari __init__
            interpolation=cv2.INTER_AREA,
        )

    def generate_augmentations(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Buat versi augmentasi dari gambar telapak tangan untuk memperkaya dataset.

        Flow Augmentasi:
        1. Simpan gambar original
        2. Rotasi ±10 derajat
        3. Scaling 1.1x
        4. Penyesuaian brightness ±25
        5. Penyesuaian contrast ±10%

        Args:
            image: Gambar telapak hasil preprocessing

        Returns:
            dict: Dictionary berisi semua versi augmentasi
                    Format: {'tipe_augmentasi': gambar_array}
        """
        # Inisialisasi dictionary & ukuran
        augmented = {}
        height, width = image.shape

        # Step 1: Simpan gambar original
        augmented["original"] = image

        # Step 2: Augmentasi Rotasi (±10°)
        for angle in [-10, 10]:
            # Hitung matrix rotasi dari pusat gambar
            M = cv2.getRotationMatrix2D(
                center=(width / 2, height / 2),  # Putar dari tengah
                angle=angle,  # Sudut rotasi
                scale=1,  # Skala tetap
            )
            # Aplikasikan rotasi dengan refleksi di border
            rotated = cv2.warpAffine(
                image,
                M,
                (width, height),
                borderMode=cv2.BORDER_REFLECT,  # Refleksi di pinggir
            )
            augmented[f"rotate_{angle}°"] = rotated

        # Step 3: Augmentasi Scaling (1.1x)
        scale = 1.1
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Perbesar gambar
        scaled = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        # Crop bagian tengah agar ukuran sama
        start_y = (scaled.shape[0] - height) // 2
        start_x = (scaled.shape[1] - width) // 2
        scaled = scaled[start_y : start_y + height, start_x : start_x + width]
        augmented["scale_1.1"] = scaled

        # Step 4: Augmentasi Brightness (±25)
        for beta in [-25, 25]:  # beta: brightness
            label = "darker" if beta < 0 else "brighter"
            adjusted = cv2.convertScaleAbs(
                image, alpha=1.0, beta=beta  # Kontras tetap  # Ubah brightness
            )
            augmented[f"{label}_10%"] = adjusted

        # Step 5: Augmentasi Contrast (±10%)
        for alpha in [0.9, 1.1]:  # alpha: contrast
            label = "lower" if alpha < 1 else "higher"
            adjusted = cv2.convertScaleAbs(
                image, alpha=alpha, beta=0  # Ubah kontras  # Brightness tetap
            )
            augmented[f"contrast_{label}_10%"] = adjusted

        return augmented

    def save_augmented_images(
        self,
        augmented_dict: Dict[str, np.ndarray],
        base_dir: str = "/CAPSTONE-PROJECT/data",
    ) -> str:
        """
        Simpan hasil augmentasi gambar ke dalam folder terstruktur.

        Flow:
        1. Buat folder 'aug' jika belum ada
        2. Generate ID person baru (001, 002, dst)
        3. Buat folder untuk person tersebut
        4. Simpan semua versi augmentasi

        Args:
            augmented_dict: Dictionary hasil augmentasi
            base_dir: Direktori utama penyimpanan data

        Returns:
            str: ID person yang baru dibuat (format: '001')
        """
        # Step 1: Persiapkan direktori augmentasi
        temp_dir = os.path.join(base_dir, "aug")
        os.makedirs(temp_dir, exist_ok=True)  # Buat folder jika belum ada

        # Step 2: Generate ID person baru
        existing_folders = glob(os.path.join(temp_dir, "person_*"))
        person_id = f"{(len(existing_folders) + 1):03d}"  # Format: 001, 002, ...

        # Step 3: Buat folder untuk person baru
        save_dir = os.path.join(temp_dir, f"person_{person_id}")
        os.makedirs(save_dir, exist_ok=True)

        # Step 4: Simpan setiap versi augmentasi
        for idx, (_, image) in enumerate(augmented_dict.items(), 1):
            # Format nama file: data_001_1.jpg, data_001_2.jpg, dst
            filename = f"data_{person_id}_{idx}.jpg"
            # Simpan gambar
            cv2.imwrite(os.path.join(save_dir, filename), image)
            # Log untuk tracking
            logger.info(f"Saved: {filename}")

        return person_id

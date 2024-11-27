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
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PalmPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128), initial_size: Tuple[int, int] = (1280, 720)):
        """Inisialisasi palm preprocessor dengan visualisasi"""
        self.target_size = target_size
        self.initial_size = initial_size  # Ukuran awal sebelum deteksi
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
        )

    def _initial_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize gambar ke ukuran initial sebelum deteksi landmark"""
        h, w = image.shape[:2]
        target_h, target_w = self.initial_size
        
        # Hitung aspect ratio
        aspect = w / h
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Image lebih lebar
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            # Image lebih tinggi
            new_h = target_h
            new_w = int(target_h * aspect)
            
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Visualisasi hasil resize
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Size: {image.shape[:2]}")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        plt.title(f"Resized: {resized.shape[:2]}")
        plt.axis("off")
        plt.show()
        
        return resized
    
    def load_and_display_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load dan tampilkan gambar asli"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
            
            # Resize gambar sebelum konversi ke RGB
            resized_image = self._initial_resize(image)
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            return image_rgb
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def _detect_hand_landmarks(
        self, image_rgb: np.ndarray
    ) -> Optional[Tuple[object, np.ndarray]]:
        """Deteksi dan visualisasi landmark tangan dengan auto-rotate"""

        def is_upside_down(landmarks):
            """Cek apakah gambar tangan terbalik"""
            wrist = landmarks.landmark[0]  # pergelangan
            middle_finger_tip = landmarks.landmark[12]  # ujung jari tengah
            return wrist.y < middle_finger_tip.y

        # Proses gambar awal
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            image_to_process = image_rgb.copy()

            # Cek apakah gambar terbalik
            if is_upside_down(landmarks):
                # Rotate 180 derajat
                height, width = image_rgb.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
                image_to_process = cv2.warpAffine(
                    image_rgb, rotation_matrix, (width, height)
                )

                # Proses ulang gambar yang sudah dirotasi
                results = self.hands.process(image_to_process)
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]

            # Visualisasi landmark
            image_with_landmarks = image_to_process.copy()
            self.mp_drawing.draw_landmarks(
                image_with_landmarks, landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            plt.figure(figsize=(8, 8))
            plt.imshow(image_with_landmarks)
            plt.title("Deteksi Landmark Tangan")
            plt.axis("off")
            plt.show()

            return landmarks, image_to_process

        logger.warning("Tidak ada landmark tangan yang terdeteksi!")
        return None, image_rgb

    def _extract_palm_roi(
        self, image_rgb: np.ndarray, landmarks: object
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        try:
            if landmarks is None:
                return None, None

            h, w = image_rgb.shape[:2]
            palm_center_indices = [1, 5, 9, 13, 17]
            palm_points = []

            for idx in palm_center_indices:
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                palm_points.append((x, y))

            center_x = int(np.mean([p[0] for p in palm_points]))
            center_y = int(np.mean([p[1] for p in palm_points])) + 90

            # Hitung angle untuk kemiringan ke kiri
            p1 = landmarks.landmark[5]  # pinky base
            p2 = landmarks.landmark[17]  # thumb base
            angle = -np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x)) + 90  # untuk miring kiri

            # Kontrol ukuran ROI
            palm_width = max([p[0] for p in palm_points]) - min([p[0] for p in palm_points])
            palm_height = max([p[1] for p in palm_points]) - min([p[1] for p in palm_points])
            
            min_size = 280
            max_size = 295
            base_size = int(max(palm_width, palm_height) * 1.0)
            roi_size = max(min_size, min(base_size, max_size))

            # Buat points untuk box
            x1 = max(0, center_x - roi_size // 2)
            y1 = max(0, center_y - roi_size // 2)
            x2 = min(w, x1 + roi_size)
            y2 = min(h, y1 + roi_size)

            roi_size = min(x2 - x1, y2 - y1)
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            # Create rotated box points
            rect_pts = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            # Rotate points around center
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            ones = np.ones(shape=(len(rect_pts), 1))
            points_ones = np.hstack([rect_pts, ones])
            rotated_pts = M.dot(points_ones.T).T
            rotated_pts = rotated_pts.astype(np.int32)

            # Extract ROI (non-rotated untuk processing)
            roi = image_rgb[y1:y2, x1:x2]

            # Visualisasi dengan box miring
            img_with_roi = image_rgb.copy()
            cv2.polylines(img_with_roi, [rotated_pts], True, (0, 255, 0), 2)

            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(img_with_roi)
            plt.title("Telapak Tangan dengan Kotak ROI")
            plt.axis("off")
            plt.subplot(122)
            plt.imshow(roi)
            plt.title(f"ROI Terekstrak {roi.shape[:2]}")
            plt.axis("off")
            plt.show()

            return roi, (x1, y1, x2, y2)

        except Exception as e:
            logger.error(f"Error dalam ekstraksi ROI: {str(e)}")
            return None, None

    def _convert_to_grayscale(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Konversi ke grayscale dengan visualisasi proses"""
        if roi is None:
            return None

        # Konversi ke grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # CLAHE pertama
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Penghilangan bayangan
        dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)

        # CLAHE kedua
        clahe_final = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        diff_img = clahe_final.apply(diff_img)

        # Normalisasi
        normalized = cv2.normalize(
            diff_img,
            None,
            alpha=15,
            beta=240,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )

        # Gamma correction
        gamma = 1.0
        normalized = np.array(255 * (normalized / 255) ** gamma, dtype="uint8")

        # Visualisasi
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(roi)
        plt.title("ROI Original (RGB)")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(gray, cmap="gray")
        plt.title("ROI Grayscale")
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(normalized, cmap="gray")
        plt.title("Grayscale Tanpa Bayangan")
        plt.axis("off")

        plt.show()

        return normalized

    def _resize_roi(self, enhanced_roi: np.ndarray) -> Optional[np.ndarray]:
        """Resize ROI dengan visualisasi"""
        if enhanced_roi is None:
            return None

        resized = cv2.resize(
            enhanced_roi, self.target_size, interpolation=cv2.INTER_AREA
        )

        # Visualisasi
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(enhanced_roi, cmap="gray")
        plt.title(f"ROI Enhanced {enhanced_roi.shape}")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(resized, cmap="gray")
        plt.title(f"ROI Resized {resized.shape}")
        plt.axis("off")
        plt.show()

        return resized

    def generate_augmentations(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate augmentasi"""
        augmented = {}
        height, width = image.shape

        # Original
        augmented["original"] = image

        # Rotasi
        angles = [-15, -10, -5, 5, 10, 15]
        for angle in angles:
            M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            rotated = cv2.warpAffine(
                image, M, (width, height), borderMode=cv2.BORDER_REFLECT
            )
            augmented[f"rotate_{angle}°"] = rotated

        # Scaling
        scales = [0.95, 1.05, 1.1]
        for scale in scales:
            new_width = int(width * scale)
            new_height = int(height * scale)
            scaled = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )

            if scale > 1:
                start_y = (scaled.shape[0] - height) // 2
                start_x = (scaled.shape[1] - width) // 2
                scaled = scaled[start_y : start_y + height, start_x : start_x + width]
            else:
                pad_y = (height - scaled.shape[0]) // 2
                pad_x = (width - scaled.shape[1]) // 2
                scaled = cv2.copyMakeBorder(
                    scaled, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT
                )

            augmented[f"scale_{scale}"] = scaled

        # Brightness
        brightnesses = [-25, -12, 12, 25]
        for beta in brightnesses:
            label = "darker" if beta < 0 else "brighter"
            intensity = "5%" if abs(beta) < 15 else "10%"
            adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
            augmented[f"{label}_{intensity}"] = adjusted

        # Contrast
        contrasts = [1.05, 1.1, 1.15, 1.2]
        for alpha in contrasts:
            label = "lower" if alpha < 1 else "higher"
            intensity = "5%" if abs(alpha - 1) < 0.1 else "10%"
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented[f"contrast_{label}_{intensity}"] = adjusted

        # Kombinasi augmentasi
        for angle in [-5, 5]:
            for beta in [-12, 12]:
                M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                rotated = cv2.warpAffine(
                    image, M, (width, height), borderMode=cv2.BORDER_REFLECT
                )
                adjusted = cv2.convertScaleAbs(rotated, alpha=1.0, beta=beta)
                label = "darker" if beta < 0 else "brighter"
                augmented[f"rotate_{angle}°_{label}_5%"] = adjusted

        # Visualisasi
        plt.figure(figsize=(20, 15))
        total_imgs = len(augmented)
        cols = 4
        rows = (total_imgs + cols - 1) // cols

        for idx, (aug_type, aug_image) in enumerate(augmented.items(), 1):
            plt.subplot(rows, cols, idx)
            plt.imshow(aug_image, cmap="gray")
            plt.title(f"Augmentation:\n{aug_type}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"\nTotal augmented images generated: {len(augmented)}")
        for idx, aug_type in enumerate(augmented.keys(), 1):
            print(f"{idx}. {aug_type}")

        return augmented

    def save_augmented_images(
        self,
        augmented_dict: Dict[str, np.ndarray],
        base_dir: str = "/CAPSTONE-PROJECT/data",
    ) -> str:
        """Simpan hasil augmentasi"""
        temp_dir = os.path.join(base_dir, "aug")
        os.makedirs(temp_dir, exist_ok=True)

        existing_folders = glob(os.path.join(temp_dir, "person_*"))
        person_id = f"{(len(existing_folders) + 1):03d}"

        save_dir = os.path.join(temp_dir, f"person_{person_id}")
        os.makedirs(save_dir, exist_ok=True)

        for idx, (aug_type, image) in enumerate(augmented_dict.items(), 1):
            filename = f"data_{person_id}_{idx}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), image)
            logger.info(f"Saved: {filename} ({aug_type})")

        logger.info(f"\nSaved {len(augmented_dict)} images in: person_{person_id}")
        return person_id

    def preprocess_image(
        self, image_path: Union[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Main preprocessing function with visualization"""
        try:
            # Load dan tampilkan gambar
            if isinstance(image_path, str):
                image_rgb = self.load_and_display_image(image_path)
            else:
                # Jika input adalah array, resize dulu
                image_rgb = cv2.cvtColor(self._initial_resize(image_path), cv2.COLOR_BGR2RGB)

            if image_rgb is None:
                return None

            # Deteksi landmark
            landmarks, processed_image = self._detect_hand_landmarks(image_rgb)
            if landmarks is None:
                return None

            # Ekstrak ROI menggunakan processed_image
            roi, _ = self._extract_palm_roi(processed_image, landmarks)
            if roi is None:
                return None

            # Konversi ke grayscale
            processed_roi = self._convert_to_grayscale(roi)
            if processed_roi is None:
                return None

            # Resize
            final_image = self._resize_roi(processed_roi)
            if final_image is None:
                return None

            return final_image

        except Exception as e:
            logger.error(f"Error dalam preprocessing: {str(e)}")
            return None

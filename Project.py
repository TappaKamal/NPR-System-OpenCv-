"""
number_plate_recognition.py

A single-file, production-style Number Plate Recognition (NPR) system using OpenCV + pytesseract.
This file is intentionally long (~500+ lines) and contains:
- Configuration management
- Robust logging
- Multiple preprocessing strategies
- Candidate region detection with heuristics
- OCR with multiple configs & post-processing
- Image augmentation utilities for testing
- Batch processing and video (webcam/file) support
- CLI interface
- Basic performance timing and report generation
- Helpful docstrings and inline comments

Usage (examples):
    python number_plate_recognition.py --input img_1.jpg --output_dir output --save True
    python number_plate_recognition.py --video src=0 --display True
    python number_plate_recognition.py --batch_dir dataset/images --save True

Requirements:
    pip install opencv-python pytesseract numpy matplotlib pillow
    Tesseract OCR engine must be installed separately (system-level).

Author: Team NPR (V Sripathi Sanjeeva, Hari Krishna, Kamal Hussain, R Lokeswar)
Date: 2025
"""

import os
import sys
import cv2
import time
import json
import glob
import argparse
import logging
import numpy as np
import pytesseract
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import Counter
from PIL import Image, ImageOps, ImageEnhance

# ---------------------------
# Basic configuration / paths
# ---------------------------

@dataclass
class Config:
    # I/O
    input_path: str = "img_1.jpg"           # default single image
    batch_dir: Optional[str] = None         # directory for batch processing
    output_dir: str = "output"              # where to save annotated images
    save_output: bool = True                # save annotated output images
    display: bool = False                   # show intermediate and final images
    video_source: Optional[str] = None      # "0" for webcam, or path to video file

    # Preprocessing
    bilateral_filter_d: int = 11
    bilateral_filter_sigma_color: int = 17
    bilateral_filter_sigma_space: int = 17
    canny_thresh1: int = 30
    canny_thresh2: int = 200
    morph_kernel_size: Tuple[int, int] = (5, 5)

    # Contour filtering heuristics
    min_plate_w: int = 80
    min_plate_h: int = 20
    aspect_ratio_min: float = 2.0
    aspect_ratio_max: float = 6.0
    max_candidate_contours: int = 80

    # OCR
    ocr_whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ocr_psm_modes: List[int] = field(default_factory=lambda: [7, 8, 11])
    ocr_oem: int = 3
    ocr_min_len: int = 4

    # Postprocessing
    allow_space_in_plate: bool = False
    cleanup_chars: str = " \n\t-:;"

    # Augmentation / debug
    augmentation_enabled: bool = True
    debug_save_candidates: bool = False

    # Advanced / performance
    max_workers: int = 4

    # Logging
    log_level: str = "INFO"

    def to_dict(self):
        return self.__dict__


# ---------------------------
# Logging utilities
# ---------------------------

def setup_logger(level="INFO") -> logging.Logger:
    logger = logging.getLogger("NPR")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


# ---------------------------
# Utility helpers
# ---------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_imwrite(path: str, image) -> bool:
    try:
        cv2.imwrite(path, image)
        return True
    except Exception:
        return False


def normalize_text(text: str, cfg: Config) -> str:
    if text is None:
        return ""
    # Remove unexpected characters and keep whitelist
    cleaned = "".join(ch for ch in text.upper() if ch in cfg.ocr_whitelist)
    return cleaned


# ---------------------------
# Augmentation utilities
# ---------------------------

def random_brightness_contrast(image: np.ndarray, brightness=0.2, contrast=0.2) -> np.ndarray:
    """Apply random brightness and contrast variations."""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if np.random.rand() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        factor = 1.0 + (np.random.rand() - 0.5) * brightness * 2
        img = enhancer.enhance(factor)
    if np.random.rand() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        factor = 1.0 + (np.random.rand() - 0.5) * contrast * 2
        img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def add_gaussian_noise(image: np.ndarray, mean=0, var=10) -> np.ndarray:
    """Add Gaussian noise to an image."""
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype("float32")
    noisy = image.astype("float32") + gauss
    noisy = np.clip(noisy, 0, 255).astype("uint8")
    return noisy


def augment_image(image: np.ndarray, n_variants: int = 6) -> List[np.ndarray]:
    """Generate augmentation variants for the input image for robust detection testing."""
    variants = []
    for i in range(n_variants):
        img = image.copy()
        if np.random.rand() < 0.6:
            img = random_brightness_contrast(img, brightness=0.5, contrast=0.5)
        if np.random.rand() < 0.4:
            img = add_gaussian_noise(img, var=np.random.randint(5, 40))
        if np.random.rand() < 0.3:
            # slight blur sometimes
            k = np.random.choice([0, 1, 3])
            if k > 0:
                img = cv2.GaussianBlur(img, (k*2+1, k*2+1), 0)
        variants.append(img)
    return variants


# ---------------------------
# Core NPR pipeline
# ---------------------------

class NPRProcessor:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    # -----------------------
    # Preprocessing methods
    # -----------------------
    def to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise_bilateral(self, gray: np.ndarray) -> np.ndarray:
        d = self.cfg.bilateral_filter_d
        sc = self.cfg.bilateral_filter_sigma_color
        ss = self.cfg.bilateral_filter_sigma_space
        return cv2.bilateralFilter(gray, d, sc, ss)

    def auto_canny(self, image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        self.logger.debug(f"Auto Canny thresholds: lower={lower}, upper={upper}")
        return cv2.Canny(image, lower, upper)

    def canny_edges(self, gray: np.ndarray) -> np.ndarray:
        # Allow fallback to manual thresholds in cfg; use auto if not sensible
        try:
            edges = cv2.Canny(gray, self.cfg.canny_thresh1, self.cfg.canny_thresh2)
        except Exception:
            edges = self.auto_canny(gray)
        return edges

    def morphological_close(self, edges: np.ndarray) -> np.ndarray:
        kx, ky = self.cfg.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return closed

    # -----------------------
    # Candidate extraction
    # -----------------------
    def find_candidate_contours(self, closed: np.ndarray) -> List[np.ndarray]:
        contours_info = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_info) == 3:
            _, contours, _ = contours_info
        else:
            contours, _ = contours_info
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[: self.cfg.max_candidate_contours]

    def contour_filter(self, contour: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h != 0 else 0.0
        if w < self.cfg.min_plate_w or h < self.cfg.min_plate_h:
            return None
        if aspect_ratio < self.cfg.aspect_ratio_min or aspect_ratio > self.cfg.aspect_ratio_max:
            return None
        return (x, y, w, h)

    # -----------------------
    # OCR & postprocessing
    # -----------------------
    def ocr_from_image(self, plate_image: np.ndarray) -> Optional[str]:
        # We'll try several OCR preprocessing combos and PSM modes to maximize chance
        gray_plate = plate_image
        if plate_image.ndim == 3:
            gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # try raw, threshold, and inverted threshold
        results = []
        try:
            _, th = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th_inv = cv2.bitwise_not(th)
        except Exception:
            th = gray_plate
            th_inv = gray_plate

        # build configs
        for psm in self.cfg.ocr_psm_modes:
            config = f"--psm {psm} --oem {self.cfg.ocr_oem} -c tessedit_char_whitelist={self.cfg.ocr_whitelist}"
            try:
                text_raw = pytesseract.image_to_string(gray_plate, config=config).strip()
                text_th = pytesseract.image_to_string(th, config=config).strip()
                text_th_inv = pytesseract.image_to_string(th_inv, config=config).strip()
            except Exception as e:
                self.logger.debug(f"OCR error with config {config}: {e}")
                text_raw = ""
                text_th = ""
                text_th_inv = ""

            for t in (text_raw, text_th, text_th_inv):
                if t:
                    normalized = normalize_text(t, self.cfg)
                    if len(normalized) >= self.cfg.ocr_min_len:
                        results.append(normalized)

        if not results:
            return None

        # choose the most common result (robust against minor variation)
        chosen = Counter(results).most_common(1)[0][0]
        return chosen

    # -----------------------
    # Plate detection pipeline for a single image
    # -----------------------
    def detect_plates_in_image(self, image: np.ndarray, image_name: str = "image") -> Dict:
        """
        Detect candidate plates in the image, run OCR on them, and return a dict:
        {
            "image_name": ...,
            "detections": [
                {"bbox": (x,y,w,h), "plate_text": "ABC1234", "score": None, "candidate_image": np.ndarray}
            ],
            "annotated_image": np.ndarray (BGR)
        }
        """
        out = {
            "image_name": image_name,
            "detections": [],
            "annotated_image": image.copy()
        }
        try:
            gray = self.to_gray(image)
            deno = self.denoise_bilateral(gray)
            edges = self.canny_edges(deno)
            closed = self.morphological_close(edges)
            contours = self.find_candidate_contours(closed)

            candidates = []
            for cnt in contours:
                bbox = self.contour_filter(cnt)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                # Expand region slightly to better capture characters
                pad_w = int(0.03 * w)
                pad_h = int(0.05 * h)
                xs = max(0, x - pad_w)
                ys = max(0, y - pad_h)
                xe = min(image.shape[1], x + w + pad_w)
                ye = min(image.shape[0], y + h + pad_h)
                plate_crop = image[ys:ye, xs:xe]
                candidates.append((bbox, plate_crop))

            # If augmentation enabled, also try on augmented variants
            augmented_results = []
            if self.cfg.augmentation_enabled:
                variants = augment_image(image, n_variants=4)
            else:
                variants = []

            # OCR on candidates
            for bbox, plate_crop in candidates:
                plate_text = self.ocr_from_image(plate_crop)
                if plate_text:
                    out["detections"].append({
                        "bbox": bbox,
                        "plate_text": plate_text,
                        "candidate_image": plate_crop
                    })
                    x, y, w, h = bbox
                    cv2.rectangle(out["annotated_image"], (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(out["annotated_image"], plate_text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

            # Try detecting in augmented variants (rarely finds new ones, but helpful)
            for v in variants:
                gray_v = self.to_gray(v)
                deno_v = self.denoise_bilateral(gray_v)
                edges_v = self.canny_edges(deno_v)
                closed_v = self.morphological_close(edges_v)
                contours_v = self.find_candidate_contours(closed_v)
                for cnt in contours_v:
                    bbox_v = self.contour_filter(cnt)
                    if bbox_v is None:
                        continue
                    x, y, w, h = bbox_v
                    pad_w = int(0.03 * w)
                    pad_h = int(0.05 * h)
                    xs = max(0, x - pad_w)
                    ys = max(0, y - pad_h)
                    xe = min(v.shape[1], x + w + pad_w)
                    ye = min(v.shape[0], y + h + pad_h)
                    plate_crop_v = v[ys:ye, xs:xe]
                    plate_text_v = self.ocr_from_image(plate_crop_v)
                    if plate_text_v:
                        # Map coordinates back to original image size if same size - here assume same
                        out["detections"].append({
                            "bbox": (x, y, w, h),
                            "plate_text": plate_text_v,
                            "candidate_image": plate_crop_v
                        })
                        cv2.rectangle(out["annotated_image"], (x, y), (x + w, y + h), (0, 128, 255), 2)
                        cv2.putText(out["annotated_image"], plate_text_v, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 128, 255), 2)

            # Deduplicate detections by text and bbox proximity
            deduped = {}
            final_detections = []
            for d in out["detections"]:
                text = d["plate_text"]
                if text in deduped:
                    # choose larger bbox or skip
                    existing = deduped[text]
                    ex_bbox = existing["bbox"]
                    if (d["bbox"][2] * d["bbox"][3]) > (ex_bbox[2] * ex_bbox[3]):
                        deduped[text] = d
                else:
                    deduped[text] = d
            for k, v in deduped.items():
                final_detections.append(v)
            out["detections"] = final_detections

            # Optionally save candidate crops for debugging
            if self.cfg.debug_save_candidates and out["detections"]:
                debug_dir = os.path.join(self.cfg.output_dir, "candidates")
                ensure_dir(debug_dir)
                for i, d in enumerate(out["detections"]):
                    crop_path = os.path.join(debug_dir, f"{image_name}_candidate_{i}_{d['plate_text']}.jpg")
                    try:
                        cv2.imwrite(crop_path, d["candidate_image"])
                    except Exception:
                        pass

        except Exception as e:
            self.logger.error(f"Error during detection on {image_name}: {e}")
            self.logger.debug(traceback.format_exc())

        return out

    # -----------------------
    # Utilities for video / webcam
    # -----------------------
    def process_video_stream(self, source=0, max_frames: Optional[int] = None):
        """
        Process a video stream (webcam or file). Yields detection dicts per frame.
        source: integer for camera index or string path to video file.
        """
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Unable to open video source: {source}")
            return

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video stream or error.")
                    break
                frame_count += 1
                detections = self.detect_plates_in_image(frame, image_name=f"frame_{frame_count}")
                yield detections
                if self.display:
                    # RESERVED: attribute, not used; use cfg.display instead
                    pass
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()

    # -----------------------
    # Batch processing helper
    # -----------------------
    def process_image_file(self, input_path: str) -> Dict:
        self.logger.info(f"Processing image: {input_path}")
        image_name = os.path.splitext(os.path.basename(input_path))[0]
        image = cv2.imread(input_path)
        if image is None:
            self.logger.error(f"Failed to load image: {input_path}")
            return {"image_name": image_name, "detections": [], "annotated_image": None}
        start = time.time()
        out = self.detect_plates_in_image(image, image_name=image_name)
        end = time.time()
        out["processing_time_sec"] = end - start
        if self.cfg.save_output and out["annotated_image"] is not None:
            ensure_dir(self.cfg.output_dir)
            out_path = os.path.join(self.cfg.output_dir, f"{image_name}_annotated_{timestamp()}.jpg")
            safe_imwrite(out_path, out["annotated_image"])
            out["output_path"] = out_path
        return out


# ---------------------------
# CLI and running logic
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Number Plate Recognition (OpenCV + Tesseract) - Single file")
    parser.add_argument("--input", type=str, default="img_1.jpg", help="Input image path")
    parser.add_argument("--batch_dir", type=str, default=None, help="Directory containing images for batch processing")
    parser.add_argument("--video", type=str, default=None, help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--save", type=lambda x: (str(x).lower() == "true"), default=True, help="Save annotated output images (True/False)")
    parser.add_argument("--display", type=lambda x: (str(x).lower() == "true"), default=False, help="Display images during processing")
    parser.add_argument("--debug_save_candidates", type=lambda x: (str(x).lower() == "true"), default=False, help="Save candidate plate crops for debugging")
    parser.add_argument("--no_augment", dest="augment_off", action="store_true", help="Disable augmentation variants")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--ocr_psm", type=str, default=None, help="Comma-separated PSM modes to try (e.g., 7,8,11)")
    parser.add_argument("--ocr_min_len", type=int, default=4, help="Minimum length of OCR result to consider")
    parser.add_argument("--max_candidates", type=int, default=80, help="Maximum contours to consider")
    parser.add_argument("--run_demo", action="store_true", help="Run demo pipeline on sample images if available")
    args = parser.parse_args()
    return args


def build_config_from_args(args) -> Config:
    cfg = Config()
    cfg.input_path = args.input
    cfg.batch_dir = args.batch_dir
    cfg.output_dir = args.output_dir
    cfg.save_output = args.save
    cfg.display = args.display
    if args.video:
        cfg.video_source = args.video
    cfg.debug_save_candidates = args.debug_save_candidates
    cfg.augmentation_enabled = not args.augment_off
    cfg.log_level = args.log_level
    if args.ocr_psm:
        try:
            cfg.ocr_psm_modes = [int(x.strip()) for x in args.ocr_psm.split(",")]
        except Exception:
            pass
    cfg.ocr_min_len = args.ocr_min_len
    cfg.max_candidate_contours = args.max_candidates
    return cfg


# ---------------------------
# Small helper: print summary
# ---------------------------

def print_summary(results: List[Dict], logger: logging.Logger):
    total_images = len(results)
    total_detections = sum(len(r.get("detections", [])) for r in results)
    avg_time = np.mean([r.get("processing_time_sec", 0.0) for r in results]) if results else 0.0
    logger.info(f"Processed images: {total_images}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Average processing time (sec): {avg_time:.3f}")

    # Frequency of plate strings
    all_plates = []
    for r in results:
        for d in r.get("detections", []):
            all_plates.append(d["plate_text"])
    if all_plates:
        freq = Counter(all_plates)
        logger.info("Top recognized plates:")
        for plate, count in freq.most_common(10):
            logger.info(f"  {plate}  x{count}")


# ---------------------------
# Example / demo images generator (optional)
# ---------------------------

def generate_demo_images(out_dir: str, n=3):
    """
    Create simple demo images containing synthetic license-like rectangles
    This is a fallback if user doesn't have images handy. Useful for testing the pipeline.
    """
    ensure_dir(out_dir)
    created = []
    for i in range(n):
        canvas = np.zeros((320, 640, 3), dtype=np.uint8) + 255  # white background
        # Draw a vehicle shape (rectangle)
        cv2.rectangle(canvas, (50, 120), (590, 260), (200, 200, 200), -1)
        # Add a plate rectangle
        plate_x1 = 220
        plate_y1 = 180
        plate_x2 = 420
        plate_y2 = 210
        cv2.rectangle(canvas, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), -1)
        # add plate text
        plate_text = f"TN{10+i}AB{100+i}"
        cv2.putText(canvas, plate_text, (plate_x1+8, plate_y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        fname = os.path.join(out_dir, f"demo_{i+1}.jpg")
        cv2.imwrite(fname, canvas)
        created.append(fname)
    return created


# ---------------------------
# MAIN
# ---------------------------

def main():
    args = parse_args()
    cfg = build_config_from_args(args)
    logger = setup_logger(cfg.log_level)
    cfg.output_dir = os.path.abspath(cfg.output_dir)
    ensure_dir(cfg.output_dir)

    # Show config in debug
    logger.debug(json.dumps(cfg.to_dict(), indent=2))

    # init processor
    processor = NPRProcessor(cfg, logger)

    results = []

    try:
        if args.run_demo:
            demo_dir = os.path.join(cfg.output_dir, "demo_images")
            demo_files = generate_demo_images(demo_dir, n=4)
            logger.info(f"Generated demo images: {demo_files}")
            for f in demo_files:
                res = processor.process_image_file(f)
                results.append(res)

        # Batch directory mode
        if cfg.batch_dir:
            images = sorted(glob.glob(os.path.join(cfg.batch_dir, "*.*")))
            logger.info(f"Found {len(images)} files in batch dir {cfg.batch_dir}")
            for img_path in images:
                res = processor.process_image_file(img_path)
                results.append(res)

        # Video mode
        elif cfg.video_source is not None:
            # Process as video stream (webcam or file)
            vs = cfg.video_source
            logger.info(f"Processing video source: {vs}")
            # allow a limited loop here: we'll process until user interrupts or end of file
            # For the CLI run, we'll process frames and show detection summary per frame; also save annotated frames optionally.
            cap = cv2.VideoCapture(int(vs) if vs.isdigit() else vs)
            if not cap.isOpened():
                logger.error("Unable to open video source.")
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video or cannot fetch frame.")
                        break
                    frame_idx += 1
                    out = processor.detect_plates_in_image(frame, image_name=f"video_frame_{frame_idx}")
                    out["processing_time_sec"] = 0.0  # could measure per-frame
                    results.append(out)
                    # Display
                    if cfg.display and out["annotated_image"] is not None:
                        cv2.imshow("NPR - Video", out["annotated_image"])
                        # Press q to exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    # save per-frame annotated image if requested
                    if cfg.save_output and out["annotated_image"] is not None:
                        ensure_dir(cfg.output_dir)
                        out_path = os.path.join(cfg.output_dir, f"video_frame_{frame_idx}_annotated.jpg")
                        safe_imwrite(out_path, out["annotated_image"])
            finally:
                cap.release()
                cv2.destroyAllWindows()

        # Single image mode
        else:
            logger.info(f"Processing single image: {cfg.input_path}")
            res = processor.process_image_file(cfg.input_path)
            results.append(res)
            # Display if requested
            if cfg.display and res.get("annotated_image") is not None:
                cv2.imshow("NPR - Result", res["annotated_image"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.debug(traceback.format_exc())

    # Print summary and save brief json report
    print_summary(results, logger)
    report_path = os.path.join(cfg.output_dir, f"report_{timestamp()}.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, default=lambda o: "<not serializable>", indent=2)
        logger.info(f"Saved report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

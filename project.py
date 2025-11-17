import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display, Javascript
from google.colab import output
from base64 import b64decode

# ===============================
# Webcam capture for Colab
# ===============================
def capture_photo(filename='photo.jpg', quality=0.9):
    js = f"""
    async function takePhoto() {{
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = '📸 Capture Photo';
        div.appendChild(capture);
        document.body.appendChild(div);

        const video = document.createElement('video');
        video.style.display = 'block';
        document.body.appendChild(video);

        const stream = await navigator.mediaDevices.getUserMedia({{video: true}});
        video.srcObject = stream;
        await video.play();

        await new Promise(resolve => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        stream.getTracks().forEach(track => track.stop());
        video.remove();
        div.remove();

        const dataUrl = canvas.toDataURL('image/jpeg', {quality});
        google.colab.kernel.invokeFunction('notebook.getPhoto', [dataUrl], {{}});
    }}
    takePhoto();
    """
    display(Javascript(js))
    image_data = {}

    def _get_photo(data_url):
        image_data['data'] = data_url

    output.register_callback('notebook.getPhoto', _get_photo)
    print("📷 Please click 'Capture Photo' in the popup window.")

    while 'data' not in image_data:
        time.sleep(0.1)

    img_bytes = b64decode(image_data['data'].split(',')[1])
    with open(filename, 'wb') as f:
        f.write(img_bytes)

    return filename

# ===============================
# License Plate Detection Function
# ===============================
def detect_plate_number(image_path, display_images=True, save_output=False,
                        output_name="detected_output.jpg"):

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or invalid path.")
        return None

    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edges = cv2.Canny(gray, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    detected_plates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if 2 <= aspect_ratio <= 6 and w > 80 and h > 20:
            plate_image = gray[y:y + h, x:x + w]

            if display_images:
                plt.imshow(plate_image, cmap="gray")
                plt.title("Candidate Plate Region")
                plt.axis("off")
                plt.show()

            _, thresh = cv2.threshold(
                plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresh_inv = cv2.bitwise_not(thresh)

            ocr_configs = [
                "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ]

            plate_text = None

            for config in ocr_configs:
                texts = [
                    pytesseract.image_to_string(plate_image, config=config).strip(),
                    pytesseract.image_to_string(thresh, config=config).strip(),
                    pytesseract.image_to_string(thresh_inv, config=config).strip()
                ]
                texts = [t for t in texts if len(t) >= 4]

                if texts:
                    plate_text = texts[0]
                    break

            if plate_text:
                detected_plates.append(plate_text)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(image, plate_text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if display_images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Plates")
        plt.axis("off")
        plt.show()

    if save_output:
        cv2.imwrite(output_name, image)

    return detected_plates

# ===============================
# Example Usage
# ===============================

# filename = capture_photo()   # For webcam in Colab
filename = "img_1.jpg"         # Static test image

plates = detect_plate_number(filename, display_images=True, save_output=True)
print("Detected Plate Numbers:", plates)

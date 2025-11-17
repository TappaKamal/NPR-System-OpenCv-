Number Plate Recognition (NPR) Using OpenCV & Tesseract OCR

This project implements an automated Number Plate Recognition system using OpenCV for image preprocessing and Tesseract OCR for extracting alphanumeric text from detected license plates.

The system works on static images and supports webcam capture inside Google Colab.

🚀 Features

Detect vehicle number plates using:

Grayscale conversion

Bilateral filtering

Canny edge detection

Morphological operations

Contour extraction & aspect ratio filtering

Extract alphanumeric text using Tesseract OCR

Display detected number plates with bounding boxes

Save processed output images

Colab webcam capture support

🛠️ Technologies Used

Python 3.x

OpenCV

Pytesseract

NumPy

Matplotlib

Google Colab JavaScript (optional for webcam)


📦 Number-Plate-Recognition
 ┣ 📜 main.py
 ┣ 📜 README.md
 ┣ 📜 img_1.jpg
 ┗ 📂 output/
pip install opencv-python pytesseract numpy matplotlib
sudo apt install tesseract-ocr

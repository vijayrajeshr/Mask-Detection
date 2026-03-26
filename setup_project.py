import urllib.request
import os


# Haar Cascade for face detection
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
# Lightweight Mask Detection Model
MODEL_URL = "https://github.com/balajisrinivas/Face-Mask-Detection/raw/master/mask_detector.model"

def download_file(url, filename):
    print(f"[INFO] Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"[INFO] Downloaded {filename} successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}. Error: {e}")
        print(f"[TIPS] Try to manually download from: {url}")

if __name__ == "__main__":
    try:
        # Ensure Haar Cascade is present
        if not os.path.exists("haarcascade_frontalface_default.xml"):
            download_file(HAAR_CASCADE_URL, "haarcascade_frontalface_default.xml")
        else:
            print("[INFO] Haar Cascade already exists.")

        # Ensure Mask Detector Model is present
        if not os.path.exists("mask_detector.h5"):
            download_file(MODEL_URL, "mask_detector.h5")
        else:
            print("[INFO] mask_detector.h5 already exists.")

        print("[INFO] Setup complete. You can now run 'python main.py'.")
    except KeyboardInterrupt:
        print("\n[INFO] Download interrupted by user.")

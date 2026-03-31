import urllib.request
import os

# Where to get the Haar Cascade and the Model
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
MODEL_URL = "https://github.com/balajisrinivas/Face-Mask-Detection/raw/master/mask_detector.model"

def get_file(url, filename):
    print(f"Downloading {filename}... this might take a bit")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Finished downloading {filename}!")
    except Exception as e:
        print(f"Error: Could not download {filename}. {e}")
        print(f"You might need to download it manually from: {url}")

if __name__ == "__main__":
    # Check for the Haar Cascade file
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        get_file(HAAR_CASCADE_URL, "haarcascade_frontalface_default.xml")
    else:
        print("Haar Cascade file is already here.")

    # Check for the Mask Model file
    if not os.path.exists("mask_detector.h5"):
        get_file(MODEL_URL, "mask_detector.h5")
    else:
        print("Model file (mask_detector.h5) is already here.")

    print("Setup is done! Now run 'python main.py' to start.")

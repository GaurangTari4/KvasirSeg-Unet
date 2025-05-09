import os
import zipfile
import urllib.request

# ✅ Updated dataset URL and filename
URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
DEST_DIR = "data"
ZIP_NAME = "kvasir-seg.zip"

def download_and_extract():
    os.makedirs(DEST_DIR, exist_ok=True)
    zip_path = os.path.join(DEST_DIR, ZIP_NAME)

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)
    print("Download and extraction completed!")

if __name__ == "__main__":
    download_and_extract()

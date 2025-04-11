import os
import zipfile
import urllib.request

def download_and_extract_kvasir(dataset_path="kvasir-seg", zip_path="kvasir-seg.zip"):
    if not os.path.exists(zip_path):
        print("Downloading Kvasir-SEG dataset...")
        urllib.request.urlretrieve("https://datasets.simula.no/downloads/kvasir-seg.zip", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    print("Download and extraction completed!")
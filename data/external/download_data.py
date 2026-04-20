import os
import subprocess
import logging
from pathlib import Path
import zipfile

DATA_DIR = Path("data/external")
KAGGLE_DATASETS = [
    "zynicide/wine-reviews",        
    "datasnaek/youtube-new"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_kaggle_dataset(dataset_name):
    try:
        logging.info(f"⬇️ Downloading: {dataset_name}")

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_name,
            "-p",
            str(DATA_DIR)
        ]

        subprocess.run(cmd, check=True)

        logging.info(f"✅ Downloaded: {dataset_name}")

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to download {dataset_name}: {e}")

def extract_all():
    for file in DATA_DIR.glob("*.zip"):
        try:
            logging.info(f"📦 Extracting: {file.name}")

            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)

            file.unlink()  
            logging.info(f"✅ Extracted & removed: {file.name}")

        except Exception as e:
            logging.error(f"❌ Extraction failed: {file.name} - {e}")

def validate_download():
    files = list(DATA_DIR.glob("*"))
    if not files:
        logging.warning("⚠️ No files found after download.")
    else:
        logging.info(f"📊 Total files in external/: {len(files)}")

def main():
    logging.info("🚀 Starting data pipeline...")

    for dataset in KAGGLE_DATASETS:
        download_kaggle_dataset(dataset)

    extract_all()
    validate_download()

    logging.info("🎉 Data pipeline completed successfully.")

if __name__ == "__main__":
    main()

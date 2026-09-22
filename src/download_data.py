from pathlib import Path
import kagglehub

DATASET = "rmisra/imdb-spoiler-dataset"
RAW_DATA_DIRECTORY = Path("data/raw")

def download_data():
    # Make data/raw directory if doesn't exist yet
    RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        DATASET,
        output_dir=RAW_DATA_DIRECTORY
    )

    print(f"dataset downloaded to {path} ")

if __name__ == "__main__":
    download_data()
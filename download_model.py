import os
import requests
from tqdm import tqdm

def download_model():
    # Replace YOUR_FILE_ID with the actual file ID from your Google Drive link
    FILE_ID = "YOUR_FILE_ID"  # You'll need to replace this
    MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "multi_condition_detector.pt")

    if os.path.exists(MODEL_PATH):
        print("Model file already exists.")
        return

    print("Downloading model file...")
    session = requests.Session()
    
    # First request to get the confirmation token
    response = session.get(MODEL_URL, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            # Get the confirmation token
            token = value
            # Make the actual download request with the token
            response = session.get(MODEL_URL, params={'confirm': token}, stream=True)
            break

    total_size = int(response.headers.get('content-length', 0))

    with open(MODEL_PATH, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

    print("Model download completed!")

if __name__ == "__main__":
    download_model() 
import gdown


def download_model(output_path):
    # Construct the gdown URL
    url = f"https://drive.google.com/file/d/19zJX28yx8rtON0HNYYho-I5dk1uGl0qj/view?usp=sharing"
    # Download the file
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    print("Model downloaded successfully and saved as:", output_path)

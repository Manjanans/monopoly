import pandas as pd
import gdown
from io import BytesIO
import ssl
import tempfile

def extract_transform_data() -> pd.DataFrame:
    ssl._create_default_https_context = ssl._create_unverified_context
    file_id = '120szsMnVIMgze6B-KboYJVjfVBy5UwRN'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    # Download to memory (BytesIO object)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        gdown.download(download_url, temp_file.name, quiet=False)

        # Read the downloaded file as an Excel file
        data = pd.read_excel(temp_file.name, sheet_name='Transición de Negocio')
    return data


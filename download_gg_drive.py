import gdown

url = "https://drive.google.com/file/d/1AtZ0pgU-odE4aRQmTtI1-8HlGXZb8X9l/view?usp=sharing"
url = url.split('/')[5]
url = 'https://drive.google.com/uc?id=' + url

output = 'download.zip'
gdown.download(url, output, quiet=False)

import zipfile
with zipfile.ZipFile('download.zip', 'r') as zip_ref:
    zip_ref.extractall('')

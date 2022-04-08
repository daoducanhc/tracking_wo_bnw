import gdown

url = "https://drive.google.com/file/d/1UjBqFmej1y3c-1kAUwi27jBr-kwTpAKZ/view?usp=sharing"
url = url.split('/')[5]
url = 'https://drive.google.com/uc?id=' + url

output = 'download.zip'
gdown.download(url, output, quiet=False)

import zipfile
with zipfile.ZipFile('download.zip', 'r') as zip_ref:
    zip_ref.extractall('output/mot_17')

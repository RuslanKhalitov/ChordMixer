import requests
import zipfile
import os
import shutil

if __name__ == "__main__":
    url = 'https://storage.googleapis.com/chordmixerdata/genbank_benchmark.zip'
    r = requests.get(url, allow_redirects=True)
    open(f'./genbank_benckmark.zip', 'wb').write(r.content)
    os.makedirs('genbank', exist_ok=True)
    shutil.unpack_archive('genbank_benckmark.zip', 'genbank')
    os.remove("genbank_benckmark.zip") 

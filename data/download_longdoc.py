import requests
import zipfile
import os
import shutil

if __name__ == "__main__":
    url = 'https://storage.googleapis.com/chordmixerdata/longdoc.zip'
    r = requests.get(url, allow_redirects=True)
    open(f'./longdoc.zip', 'wb').write(r.content)
    os.makedirs('longdoc', exist_ok=True)
    shutil.unpack_archive('longdoc.zip', 'longdoc')
    os.remove("longdoc.zip")

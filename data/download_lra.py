import requests
import zipfile
import os
import shutil
import subprocess

# After downloading please run "tar -xvf lra_release.gz"

if __name__ == "__main__":
    url = 'https://storage.googleapis.com/long-range-arena/lra_release.gz'
    r = requests.get(url, allow_redirects=True)
    open(f'./lra_release.gz', 'wb').write(r.content)
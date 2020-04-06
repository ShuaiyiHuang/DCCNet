# download_datasets.py from WeakAlign Rocco et al. CVPR2018

from os.path import exists, join, basename, dirname, splitext
from os import makedirs, remove, rename
from six.moves import urllib
import tarfile
import zipfile
import requests
import sys
import click


def download_and_uncompress(url, dest=None, chunk_size=1024, replace="ask",
                            label="Downloading {dest_basename} ({size:.2f}MB)"):
    dest = dest or "./" + url.split("/")[-1]
    dest_dir = dirname(dest)
    if not exists(dest_dir):
        makedirs(dest_dir)
    if exists(dest):
        if (replace is False
                or replace == "ask"
                and not click.confirm("Replace {}?".format(dest))):
            return
    # download file
    with open(dest, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.write("{:.1%}".format(dl / total_length))
                sys.stdout.flush()
    sys.stdout.write("\n")
    # uncompress
    if dest.endswith("zip"):
        file = zipfile.ZipFile(dest, 'r')
    elif dest.endswith("tar"):
        file = tarfile.open(dest, 'r')
    elif dest.endswith("tar.gz"):
        file = tarfile.open(dest, 'r:gz')
    else:
        return dest

    print("Extracting data...")
    file.extractall(dest_dir)
    file.close()

    return dest


def download_PF_willow(dest="./proposal-flow-willow"):
    print("Fetching PF Willow dataset ")
    url = "http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip"
    file_path = join(dest, basename(url))
    download_and_uncompress(url, file_path)

    print('Downloading image pair list \n')
    url = "http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_pf.csv"
    file_path = join(dest, basename(url))
    download_and_uncompress(url, file_path)

def download_TSS(dest="./tss"):
    print("Fetching TSS dataset ")
    url = "http://www.hci.iis.u-tokyo.ac.jp/datasets/data/JointCorrCoseg/TSS_CVPR2016.zip"
    file_path = join(dest, basename(url))
    download_and_uncompress(url, file_path)

    print('Downloading image pair list \n')
    url = "http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_tss.csv"
    file_path = join(dest, basename(url))
    download_and_uncompress(url, file_path)

if __name__ == '__main__':
    download_PF_willow()

    download_TSS()
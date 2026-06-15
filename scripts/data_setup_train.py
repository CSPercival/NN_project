import os
import config.paths as paths

def download():
    os.makedirs(paths.DATA_DIR, exist_ok=True)

    for url in paths.COCO_FILES_TRAIN.values():
        os.system(f"wget -c -P {paths.DATA_DIR} {url}")


def extract():
    # Wypakowywanie archiwów (pomija istniejące pliki dzięki -n)
    os.system(f"unzip -n -q {paths.DATA_DIR}/train2017.zip -d {paths.DATA_DIR}")
    os.system(f"unzip -n -q {paths.DATA_DIR}/annotations_trainval2017.zip -d {paths.DATA_DIR}")


def data_setup():
    download()
    extract()
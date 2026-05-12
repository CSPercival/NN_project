import os
import config.paths as paths

def download():
    os.makedirs(paths.DATA_DIR, exist_ok=True)

    for url in paths.COCO_FILES.values():
        os.system(f"wget -P {paths.DATA_DIR} {url}")


def extract():
    os.system(f"unzip -q {paths.DATA_DIR}/val2017.zip -d {paths.DATA_DIR}")
    os.system(f"unzip -q {paths.DATA_DIR}/annotations_trainval2017.zip -d {paths.DATA_DIR}")


def setup():
    download()
    extract()

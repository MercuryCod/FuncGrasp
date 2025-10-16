import os
import os.path as op
import zipfile
import subprocess
from glob import glob


def unzip(zip_p, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(out_dir,)


def unzip_split_files(first_file_path, dst_path):
    try:
        subprocess.run(["7z", "x", "-o" + dst_path, first_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {str(e)}")


def main():
    """src dir = oakink_dir/zipped/"""
    """dst dir = oakink_dir/"""
    # Support DATA_ROOT, OAKINK_PATH, or OAKINK_DIR (legacy)
    data_root = os.environ.get('DATA_ROOT', '/workspace/data')
    oakink_dir = os.environ.get("OAKINK_DIR") or os.environ.get("OAKINK_PATH") or os.path.join(data_root, 'OakInk')
    assert oakink_dir is not None, "Please set environment variable 'OAKINK_DIR', 'OAKINK_PATH', or 'DATA_ROOT'"

    zipped_dir = os.path.join(oakink_dir, "zipped")
    fnames = glob(os.path.join(zipped_dir, "**/*"), recursive=True)
    fnames = [f for f in fnames if os.path.isfile(f)]  # remove dir names

    oakbase_zip = None
    oakink_image_zips = []
    oakink_shape_zips = []

    for fn in fnames:
        if not (".zip" in fn or ".z0" in fn):
            continue
        # Skip any stream-related archives entirely (we do not use or restore stream_release_v2)
        if (
            "stream_release_v2" in fn
            or "/stream_zipped/" in fn
            or "/image/stream/" in fn
        ):
            continue
        if "OakBase" in fn:
            oakbase_zip = fn
        elif "/image/" in fn:
            oakink_image_zips.append(fn)
        elif "/shape/" in fn:
            oakink_shape_zips.append(fn)
        else:
            print(f"Unknown zip: {fn}")

    out_dir = oakink_dir
    os.makedirs(out_dir, exist_ok=True)

    # unzip OakBase.zip
    print(f"Unzipping {oakbase_zip} to {out_dir}")
    unzip(oakbase_zip, out_dir)

    # unzip OakInk-Shape files
    for zip_p in oakink_shape_zips:
        out_p = op.join(out_dir, "shape")
        print(f"Unzipping {zip_p} to {out_p}")
        unzip(zip_p, out_p)

    # unzip OakInk-Image files
    for zip_p in oakink_image_zips:
        # Only unzip non-stream image archives (e.g., obj, anno)
        if (
            "stream_release_v2_combined.zip" in zip_p
            or "/stream_zipped/" in zip_p
            or "/image/stream/" in zip_p
            or "stream_release_v2" in zip_p
        ):
            continue
        out_p = op.join(out_dir, "image")
        print(f"Unzipping {zip_p} to {out_p}")
        unzip(zip_p, out_p)




if __name__ == "__main__":
    main()
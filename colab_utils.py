"""Shared data setup for the two Colab notebooks.

UDnet_Baseline_Colab.ipynb (original UDnet) and UDnetU_Colab.ipynb (UDnet-U)
both call these functions with the same seed, so they train on the same images
and are scored on the same 90 UIEB test images.
"""
import os
import sys
import csv
import json
import time
import random
import shutil
import subprocess
import urllib.request
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# name: (Google Drive id, RAR password published on https://li-chongyi.github.io/proj_benchmark.html,
#        expected image count, minimum file size in MB)
UIEB_FILES = {
    "raw-890":        ("12W_kkblc2Vryb9zHQ6BfGQ_NKUfXYk13", "1234567", 890, 600),
    "reference-890":  ("1cA-8CzajnVEL4feBRKdBxjEe6hwql6Z7", "8901234", 890, 750),
    "challenging-60": ("1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4", "5678901",  60,  29),
}
AUTHOR_W_ID = "1SsOoMmO1V2cuYDoFIWfQT_MPmx5lmYZu"   # UDnet.pth, link from the repo README

# The official EUVP Google Drive folder now returns 404. This public Hugging Face copy holds
# the EUVP "Underwater Dark" training pairs (5550); only the good-quality side is used here.
EUVP_URL = "https://huggingface.co/datasets/Ken1053/EUVP/resolve/main/data/train-00000-of-00001.parquet"
EUVP_GOOD_COLUMN = "edited_image"


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)])


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
def _big_enough(path, min_mb):
    return os.path.exists(path) and os.path.getsize(path) > min_mb * 1e6


def fetch_gdrive(fid, out, min_mb):
    if _big_enough(out, min_mb):
        print("cached:", os.path.basename(out))
        return out
    import gdown
    try:
        gdown.download(id=fid, output=out, quiet=False)
    except Exception as e:
        print("gdown:", e)
    if not _big_enough(out, min_mb):
        raise RuntimeError("download failed: {} -- open https://drive.google.com/file/d/{}/view, download it "
                           "yourself and put it at {}".format(os.path.basename(out), fid, out))
    return out


def fetch_url(url, out, min_mb):
    if _big_enough(out, min_mb):
        print("cached:", os.path.basename(out))
        return out
    print("downloading", url)
    urllib.request.urlretrieve(url, out + ".part")
    os.replace(out + ".part", out)
    if not _big_enough(out, min_mb):
        raise RuntimeError("download failed or truncated: " + url)
    return out


def _rar_tool():
    if shutil.which("unrar"):
        return "unrar"
    subprocess.run("apt-get -qq update && apt-get -qq install -y unrar", shell=True)
    if shutil.which("unrar"):
        return "unrar"
    subprocess.run("apt-get -qq install -y p7zip-full p7zip-rar", shell=True)
    if shutil.which("7z"):
        return "7z"
    raise RuntimeError("no rar extractor found (unrar / 7z)")


def prepare_uieb(uieb_dir, cache_dir):
    """Download + extract UIEB (encrypted RARs) into uieb_dir/<name>/."""
    os.makedirs(uieb_dir, exist_ok=True)
    tool = None
    for name, (fid, pw, n_expect, min_mb) in UIEB_FILES.items():
        dest = os.path.join(uieb_dir, name)
        if count_images(dest) == n_expect:
            print("{}: already extracted ({} images)".format(name, n_expect))
            continue
        rar = fetch_gdrive(fid, os.path.join(cache_dir, name + ".rar"), min_mb)
        tool = tool or _rar_tool()
        cmd = (["unrar", "x", "-o+", "-idq", "-p" + pw, rar, uieb_dir + "/"] if tool == "unrar"
               else ["7z", "x", "-y", "-p" + pw, "-o" + uieb_dir, rar])
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = count_images(dest)
        if r.returncode != 0 or n != n_expect:
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise RuntimeError("could not extract {}.rar ({}/{} images). Check the password on "
                               "https://li-chongyi.github.io/proj_benchmark.html or delete the file "
                               "and download again".format(name, n, n_expect))
        print("{}: extracted {} images".format(name, n))


def author_weights(cache_dir):
    return fetch_gdrive(AUTHOR_W_ID, os.path.join(cache_dir, "UDnet.pth"), 60)


def prepare_clean_domain(cache_dir, out_dir):
    """Extract EUVP good-quality images (unpaired clean domain Y) into out_dir."""
    if count_images(out_dir) > 0:
        print("clean domain: already extracted ({} images)".format(count_images(out_dir)))
        return
    import pyarrow.parquet as pq
    pqf = fetch_url(EUVP_URL, os.path.join(cache_dir, "euvp_dark_train.parquet"), 150)
    os.makedirs(out_dir, exist_ok=True)
    col = pq.read_table(pqf, columns=[EUVP_GOOD_COLUMN]).column(0).to_pylist()
    for i, cell in enumerate(col):
        data = cell["bytes"]
        ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
        with open(os.path.join(out_dir, "euvp_{:05d}{}".format(i, ext)), "wb") as f:
            f.write(data)
    print("clean domain: extracted {} images".format(len(col)))


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
def make_split(uieb_dir, split_path, seed, n_train):
    """800/90 split of UIEB, saved once and reused by every notebook and run."""
    raw = list_images(os.path.join(uieb_dir, "raw-890"))
    ref = list_images(os.path.join(uieb_dir, "reference-890"))
    assert raw == ref, "raw and reference file names differ"
    if os.path.exists(split_path):
        split = json.load(open(split_path))
        print("using existing split", split_path)
    else:
        order = raw[:]
        random.Random(seed).shuffle(order)
        split = {"train": sorted(order[:n_train]), "test": sorted(order[n_train:])}
        json.dump(split, open(split_path, "w"), indent=0)
        print("new split saved to", split_path)
    assert not set(split["train"]) & set(split["test"]), "train and test overlap"
    return split


def nested_subset(names, n, seed):
    """First n of one fixed shuffle, so subset(200) < subset(400) < subset(800)."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    if n > len(order):
        raise ValueError("asked for {} images, only {} available".format(n, len(order)))
    return sorted(order[:n])


def link_all(names, src, dst):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for n in names:
        try:
            os.symlink(os.path.join(src, n), os.path.join(dst, n))
        except OSError:                      # e.g. Windows without symlink rights
            shutil.copy(os.path.join(src, n), os.path.join(dst, n))


def build_data_dirs(uieb_dir, data_dir, split, n_subset, seed):
    """data/train/image (subset of the train split), data/test/{input,reference}, data/challenging-60."""
    train = nested_subset(split["train"], n_subset, seed)
    link_all(train, os.path.join(uieb_dir, "raw-890"), os.path.join(data_dir, "train", "image"))
    link_all(split["test"], os.path.join(uieb_dir, "raw-890"), os.path.join(data_dir, "test", "input"))
    link_all(split["test"], os.path.join(uieb_dir, "reference-890"), os.path.join(data_dir, "test", "reference"))
    ch = os.path.join(uieb_dir, "challenging-60")
    link_all(list_images(ch), ch, os.path.join(data_dir, "challenging-60"))
    for d in ["train/image", "test/input", "test/reference", "challenging-60"]:
        print("{:<16s} {} images".format(d, count_images(os.path.join(data_dir, d))))


def build_clean_dir(euvp_dir, data_dir, n, seed):
    names = nested_subset(list_images(euvp_dir), n, seed)
    link_all(names, euvp_dir, os.path.join(data_dir, "clean"))
    print("{:<16s} {} images".format("clean", len(names)))


# ----------------------------------------------------------------------------
# Running scripts
# ----------------------------------------------------------------------------
def run(args):
    """Run a repo script with the current Python; print the tail; raise on failure."""
    print("$", " ".join(args))
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise RuntimeError("{} failed (exit {})".format(args[0], r.returncode))


def run_logged(args, log_path, show=("Complete", "Checkpoint", "Iter ", "Loading", "Resum")):
    """Run a long training script: full output to log_path, only summary lines on screen."""
    t0 = time.time()
    tail = []
    with open(log_path, "a") as log:
        p = subprocess.Popen([sys.executable, "-u"] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            tail = (tail + [line])[-40:]
            if any(s in line for s in show):
                print("[{:6.1f} min] {}".format((time.time() - t0) / 60, line.rstrip()), flush=True)
        p.wait()
    if p.returncode:
        print("".join(tail))
        raise RuntimeError("{} failed (exit {}) -- full log: {}".format(args[0], p.returncode, log_path))


def summarize(work):
    """Mean of every metrics CSV under work/ -- one row per method/run."""
    rows = [("raw input (no enhancement)", os.path.join(work, "metrics_raw.csv")),
            ("UDnet author weights", os.path.join(work, "metrics_author.csv"))]
    runs = os.path.join(work, "runs")
    if os.path.isdir(runs):
        rows += [(r, os.path.join(runs, r, "metrics_test.csv")) for r in sorted(os.listdir(runs))]
    keys = ["PSNR", "SSIM", "GMSD", "UIQM", "UCIQE"]
    print("{:<64s}".format("method / run") + "".join("{:>9s}".format(k) for k in keys) + "    n")
    for name, path in rows:
        if not os.path.exists(path):
            continue
        data = list(csv.DictReader(open(path)))
        vals = ["{:>9.4f}".format(np.mean([float(d[k]) for d in data])) if k in data[0] else "{:>9s}".format("-")
                for k in keys]
        print("{:<64s}".format(name[:64]) + "".join(vals) + "  {:>3d}".format(len(data)))

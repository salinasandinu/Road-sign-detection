"""
YOLOv8 TEST-ONLY SCRIPT (3-Class Mapping)

Classes:
0 -> DWS-01
1 -> DWS-02
2 -> other

This script:
1. Converts TEST XML annotations to YOLO format (3 classes)
2. Creates test data.yaml
3. Loads best.pt
4. Evaluates on TEST dataset (mAP, Precision, Recall)
"""

import os
import yaml
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CLASS MAPPING =====================

ORIGINAL_CLASS_DICT = {
    "DWS-01": 0, "DWS-02": 1, "DWS-03": 2, "DWS-04": 3, "DWS-09": 4, "DWS-10": 5,
    "DWS-11": 6, "DWS-12": 7, "DWS-13": 8, "DWS-14": 9, "DWS-15": 10,
    "DWS-16": 11, "DWS-17": 12, "DWS-18": 13, "DWS-19": 14, "DWS-20": 15,
    "DWS-21": 16, "DWS-25": 17, "DWS-26": 18, "DWS-27": 19, "DWS-28": 20,
    "DWS-29": 21, "DWS-32": 22, "DWS-33": 23, "DWS-35": 24, "DWS-36": 25,
    "DWS-40": 26, "DWS-41": 27, "DWS-42": 28, "DWS-44": 29, "DWS-46": 30,
    "MNS-01": 31, "MNS-02": 32, "MNS-03": 33, "MNS-04": 34, "MNS-05": 35,
    "MNS-06": 36, "MNS-07": 37, "MNS-09": 38, "OSD-01": 39, "OSD-02": 40,
    "OSD-03": 41, "OSD-04": 42, "OSD-06": 43, "OSD-07": 44, "OSD-16": 45,
    "OSD-17": 46, "OSD-26": 47, "PHS-01": 48, "PHS-02": 49, "PHS-03": 50,
    "PHS-04": 51, "PHS-09": 52, "PHS-23": 53, "PHS-24": 54, "PRS-01": 55,
    "PRS-02": 56, "RSS-02": 57, "SLS-100": 58, "SLS-15": 59, "SLS-40": 60,
    "SLS-50": 61, "SLS-60": 62, "SLS-70": 63, "SLS-80": 64,
    "APR-09": 65, "APR-10": 66, "APR-11": 67, "APR-12": 68, "APR-14": 69,
    "TLS-C": 70, "TLS-E": 71, "TLS-G": 72, "TLS-R": 73, "TLS-Y": 74
}

TARGET_CLASS_IDS = {0, 1}  # DWS-01, DWS-02

def remap_class_id(original_id):
    return original_id if original_id in TARGET_CLASS_IDS else 2

# ===================== XML → YOLO =====================

def get_image_size(xml_path):
    tree = ET.parse(xml_path)
    size = tree.getroot().find("size")
    return int(size.find("width").text), int(size.find("height").text)

def convert_xml_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in ORIGINAL_CLASS_DICT:
            continue

        class_id = remap_class_id(ORIGINAL_CLASS_DICT[name])
        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return yolo_labels

# ===================== PREPARE TEST DATA =====================

def prepare_test_dataset(test_dir, out_dir):
    img_out = Path(out_dir, "images/test")
    lbl_out = Path(out_dir, "labels/test")
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
    logger.info(f"Found {len(images)} test images")

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        xml_path = img_path.replace(".jpg", ".xml")

        if not os.path.exists(xml_path):
            continue

        w, h = get_image_size(xml_path)
        labels = convert_xml_to_yolo(xml_path, w, h)

        shutil.copy(img_path, img_out / img_name)
        with open(lbl_out / img_name.replace(".jpg", ".txt"), "w") as f:
            f.write("\n".join(labels))

    data_yaml = {
        "path": str(Path(out_dir).absolute()),
        "train": "images/train",   # not used
        "val": "images/test",
        "nc": 3,
        "names": {0: "DWS-01", 1: "DWS-02", 2: "other"}
    }

    yaml_path = Path(out_dir, "data_test.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    return str(yaml_path)

# ===================== EVALUATION =====================

def main():
    test_dir = "/kaggle/input/road-signs/test/test"   # CHANGE IF NEEDED
    test_out = "test_dataset"
    best_model = "/kaggle/input/best-model/best.pt"

    test_yaml = prepare_test_dataset(test_dir, test_out)

    model = YOLO(best_model)
    metrics = model.val(
        data=test_yaml,
        imgsz=640,
        conf=0.001,
        iou=0.5,
        device=0
    )

    print("\n===== TEST METRICS =====")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()

    
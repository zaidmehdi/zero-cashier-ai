"""Converting annotations from XML format to YOLO format and splitting train/val"""

import os
import random
import shutil
from pylabel import importer


def convert_format(path, output_path):

    dataset = importer.ImportVOC(path=path)
    dataset.export.ExportToYoloV5(output_path=output_path)


def split_train_val(img_dir, annotations_dir, train_dir, val_dir, val_split=0.2):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    num_val_images = int(len(image_files) * val_split)

    val_images = random.sample(image_files, num_val_images)

    for image in val_images:
        image_path = os.path.join(img_dir, image)
        annotation_path = os.path.join(annotations_dir, image.replace('.JPG', '.txt').replace('.jpg', '.txt'))
        shutil.copy(image_path, os.path.join(val_dir, image))
        shutil.copy(annotation_path, os.path.join(val_dir, os.path.basename(annotation_path)))

    for image in os.listdir(img_dir):
        if image not in val_images:
            image_path = os.path.join(img_dir, image)
            annotation_path = os.path.join(annotations_dir, image.replace('.JPG', '.txt').replace('.jpg', '.txt'))
            shutil.copy(image_path, os.path.join(train_dir, image))
            shutil.copy(annotation_path, os.path.join(train_dir, os.path.basename(annotation_path)))


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def main():

    convert_format(path="datasets/data/annotated", 
                   output_path="datasets/data/yolo")

    img_dir = "datasets/data/raw"
    annotations_dir = "datasets/data/yolo"
    train_dir = "datasets/data/train"
    val_dir = "datasets/data/val"

    clear_directory(train_dir)
    clear_directory(val_dir)

    split_train_val(img_dir, annotations_dir, train_dir, val_dir)
    

if __name__ == "__main__":
    main()

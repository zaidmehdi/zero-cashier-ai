"""Converting annotations from XML format to YOLO format"""

from pylabel import importer


def main():

    dataset = importer.ImportVOC(path="data/annotated")
    dataset.export.ExportToYoloV5(output_path="data/yolo")


if __name__ == "__main__":
    main()

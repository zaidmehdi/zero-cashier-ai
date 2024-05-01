import warnings

from object_detection import ObjectDetection


warnings.filterwarnings("ignore")


def main():

    detector = ObjectDetection(capture_index=0)
    detector()


if __name__ == "__main__":
    main()
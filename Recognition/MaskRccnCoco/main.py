import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation

def object_detection_on_an_image():
    segment_image = instance_segmentation() #модель генерирующая углы и рамки сегментации
    segment_image.load_model("mask_rcnn_coco.h5")

    segment_image.segmentImage(
        image_path="object1.jpg",
        output_image_name="new1_object.jpg"
    )

def main():
    object_detection_on_an_image()

if __name__ == '__main__':
    main()

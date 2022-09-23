import cv2
from yolov4 import Yolov4BatchTRT
from utils import plot_boxes_cv2

if __name__ == '__main__':
    # detector = Yolov4BatchTRT(model_path='weights/yolov4_-1_3_608_608_dynamic.trt',
    #                             input_shape = (608, 608),
    #                             batch_size = 4,
    #                             engine = 'TRT')
    # ONNX
    detector = Yolov4BatchTRT(model_path='weights/yolov4_-1_3_608_608_dynamic.onnx',
                                input_shape = (608, 608),
                                batch_size = 4,
                                engine = 'ONNX')

    # TRITON
    # detector = Yolov4BatchTRT(input_shape = (608, 608),
    #                     batch_size = 4,
    #                     engine = "TRITON",
    #                     triton_model_name = 'infer_object_detection_yolov4_coco',
    #                     triton_protocol = 'HTTP',
    #                     triton_host = "0.0.0.0:8000",
    #                     triton_verbose = False)

    # Prepare input
    bgr_image = cv2.imread('test_images/test.png')


    # Get prediction
    bboxes = detector.detect([bgr_image], conf_thresh=0.5, nms_thresh = 0.5, pad = True)

    plot_boxes_cv2(bgr_image, bboxes[0], savename='predictions_onnx.jpg', class_names=detector.class_names)
import cv2
import numpy as np
import onnxruntime
import sys
sys.path.append('..')
from object_detection_yolov4.utils import post_processing, load_class_names
from object_detection_yolov4.exec_backends.triton_backend import Yolov4TritonGRPC, Yolov4TritonHTTP
# from object_detection_yolov4.exec_backends.trt_backend import TrtModel

class Yolov4BatchTRT:
    def __init__(self,
                model_path='weights/yolov4_-1_3_608_608_dynamic.onnx',
                input_shape = (608, 608),
                batch_size = 1,
                engine = 'TRT',
                triton_model_name = 'infer_object_detection_yolov4_coco',
                triton_protocol = 'GRPC',
                triton_host = "0.0.0.0:8001",
                triton_verbose = False,
                namesfile = 'weights/coco.names',
                ):

        self.engine = engine
        self.input_size = input_shape
        if self.engine == 'TRT':
            self.model = TrtModel(model_path, max_size = max(self.input_size))
        elif self.engine == 'ONNX':
            self.model = onnxruntime.InferenceSession(model_path)
        elif self.engine == 'TRITON':
            if triton_protocol == 'GRPC':
                self.model = Yolov4TritonGRPC(triton_host = triton_host, # default GRPC port
                                triton_model_name = triton_model_name,
                                verbose = triton_verbose)
            elif triton_protocol == 'HTTP':
                self.model = Yolov4TritonHTTP(triton_host = triton_host, # default HTTP port
                                triton_model_name = triton_model_name,
                                verbose = triton_verbose)
        else:
            raise NotImplementedError("Current support only TRT, ONNX and TRITON engine")
        
        self.batch_size = batch_size
        self.class_names = load_class_names(namesfile)
        
    def preprocess_batch(self, list_bgr_img, input_size, pad = False):
        '''
            Preprocess batch of image
        '''
        batch_size = len(list_bgr_img)
        batch_image = np.zeros((batch_size, input_size[1], input_size[0], 3), dtype=np.uint8)
        batch_scale = []
        for ix, img in enumerate(list_bgr_img):
            if pad:
                im_ratio = float(img.shape[0]) / img.shape[1]
                model_ratio = float(input_size[1]) / input_size[0]
                if im_ratio>model_ratio:
                    new_height = input_size[1]
                    new_width = int(new_height / im_ratio)
                else:
                    new_width = input_size[0]
                    new_height = int(new_width * im_ratio)
                scale = float(new_height) / img.shape[0]
                resized_img = cv2.resize(img, (new_width, new_height))
                batch_image[ix, :new_height, :new_width, :] = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('resized_img.jpg', batch_image[ix])
                batch_scale.append((new_width, new_height))
            else:
                resized_img = cv2.resize(img, (input_size[1], input_size[0]))
                batch_image[ix] = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                batch_scale.append((input_size[1], input_size[0]))
        batch_image = batch_image.astype('float32') / 255.0 # Normalization
        batch_image = np.transpose(batch_image, (0, 3, 1, 2))
        return batch_image, batch_scale

    def detect_one_batch(self, list_bgr_img, conf_thresh=0.5, nms_thresh = 0.5, pad = False):
        # Pre-process
        batch_image, batch_scale = self.preprocess_batch(list_bgr_img, self.input_size, pad = pad)
        if self.engine == 'ONNX':
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: batch_image})
        elif self.engine == 'TRT':
            output = self.model.run(batch_image)

        else:
            output = self.model.run([batch_image])

        # Infer
        # Post-process
        boxes = post_processing(output = output,
                                conf_thresh = conf_thresh, 
                                nms_thresh = nms_thresh, 
                                class_names = self.class_names, 
                                pad = pad, 
                                batch_scale = batch_scale,
                                input_size = self.input_size)
        return boxes

    def detect(self, list_bgr_img, conf_thresh=0.5, nms_thresh = 0.5, pad = False):
        '''
            Predict batch of image
            Input:
                - img: pillow Image
        '''
        # Preprocess
        total_img = len(list_bgr_img)
        total_batch = int(total_img/self.batch_size) if total_img % self.batch_size == 0 else int(total_img/self.batch_size) + 1    
        # Feed to CNN + transformer
        result = []
        for i in range(total_batch):
            low = i*self.batch_size
            high = min(total_img, (i+1)*self.batch_size) 
            result.extend(self.detect_one_batch(list_bgr_img[low:high], conf_thresh=conf_thresh, nms_thresh = nms_thresh, pad = pad))
        
        return result
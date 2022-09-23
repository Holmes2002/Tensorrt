# Yolov4 COCO

Convert ONNX to TensorRT
```
/usr/src/tensorrt/bin/trtexec \
        --onnx=weights/yolov4_-1_3_608_608_dynamic.onnx \
        --saveEngine=weights/yolov4_-1_3_608_608_dynamic.trt \
        --maxBatch=4 \
        --verbose \
        --workspace=128
```
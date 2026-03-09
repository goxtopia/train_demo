import onnx
import sys

output_path = "app/src/main/assets/yolov8n_feature_extractor.onnx"
try:
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("YOLOv8n ONNX model verified.")
except Exception as e:
    print(f"Error: {e}")

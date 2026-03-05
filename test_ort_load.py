import onnxruntime as ort

model_path = "app/src/main/assets/yolov8n_feature_extractor.onnx"

print("Trying to load directly from file path...")
try:
    session = ort.InferenceSession(model_path)
    print("Success loading from file path.")
except Exception as e:
    print(f"Error loading from file path: {e}")

print("Trying to load from byte array...")
try:
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    session = ort.InferenceSession(model_bytes)
    print("Success loading from byte array.")
except Exception as e:
    print(f"Error loading from byte array: {e}")

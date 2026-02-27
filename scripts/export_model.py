import torch
import timm
import onnx
import os

def export_model():
    # Load convnextv2_atto model
    # num_classes=0 removes the classification head
    # global_pool='' removes the global pooling layer, keeping spatial features
    print("Loading model...")
    model = timm.create_model('convnextv2_atto', pretrained=True, num_classes=0, global_pool='')
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Define output path
    output_path = "app/src/main/assets/feature_extractor.onnx"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Model exported successfully.")

    # Downgrade IR version for compatibility with ONNX Runtime 1.17.0
    # The error message says "Unsupported model IR version: 10, max supported IR version: 9"
    print("Downgrading IR version to 9...")
    onnx_model = onnx.load(output_path)
    onnx_model.ir_version = 9
    onnx.save(onnx_model, output_path)
    print("Model IR version updated.")

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified.")

if __name__ == "__main__":
    export_model()

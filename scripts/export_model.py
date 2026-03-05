import torch
import torch.nn as nn
import timm
import onnx
import os
import struct

def export_convnextv2():
    print("Loading ConvNeXtV2 model...")
    model = timm.create_model('convnextv2_atto', pretrained=True, num_classes=0, global_pool='')
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = "app/src/main/assets/feature_extractor.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Downgrading IR version to 9 and bundling data...")
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model.ir_version = 9
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=False
    )
    if os.path.exists(output_path + ".data"):
        os.remove(output_path + ".data")

def export_yolov8n():
    from ultralytics import YOLO
    print("Loading YOLOv8n-cls model...")
    yolo_model = YOLO('yolov8n-cls.pt')

    # YOLOv8n-cls model structure: yolo_model.model.model has 10 layers (0 to 9)
    # Layer 9 is the Classify head: Conv2d(256->1280), BN, SiLU, GAP, Linear(1280->1000)
    # We want to extract features from layer 8

    class FeatureExtractorModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            # Extract layers 0 through 8
            self.features = nn.Sequential(*list(model.model.model.children())[:9])

        def forward(self, x):
            return self.features(x)

    feature_model = FeatureExtractorModel(yolo_model)
    feature_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = "app/src/main/assets/yolov8n_feature_extractor.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting YOLOv8n feature extractor to {output_path}...")
    torch.onnx.export(
        feature_model, dummy_input, output_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Downgrading IR version to 9 and bundling data for YOLOv8n...")
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model.ir_version = 9
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=False
    )
    if os.path.exists(output_path + ".data"):
        os.remove(output_path + ".data")

    # Now extract the fused weights from Layer 9 Conv + BN
    classify_layer = yolo_model.model.model[9]
    conv_module = classify_layer.conv

    conv_w = conv_module.conv.weight.detach() # shape [1280, 256, 1, 1]
    # Fuse BN into Conv
    if conv_module.bn is not None:
        bn = conv_module.bn
        bn_rm = bn.running_mean.detach()
        bn_rv = bn.running_var.detach()
        bn_w = bn.weight.detach()
        bn_b = bn.bias.detach()
        bn_eps = bn.eps

        # BN scale and bias
        scale = bn_w / torch.sqrt(bn_rv + bn_eps)
        bias = bn_b - bn_rm * scale

        # Fused weights
        fused_w = conv_w * scale.view(-1, 1, 1, 1)
        fused_b = bias
    else:
        fused_w = conv_w
        if conv_module.conv.bias is not None:
            fused_b = conv_module.conv.bias.detach()
        else:
            fused_b = torch.zeros(conv_w.size(0))

    # Flatten weights
    fused_w = fused_w.view(-1).numpy()
    fused_b = fused_b.view(-1).numpy()

    weights_path = "app/src/main/assets/yolov8n_conv_weights.bin"
    print(f"Exporting fused conv weights to {weights_path}...")

    with open(weights_path, 'wb') as f:
        # Write format version 1
        f.write(struct.pack('>B', 1))

        # Write lengths
        f.write(struct.pack('>I', len(fused_w)))
        f.write(struct.pack('>I', len(fused_b)))

        # Write floats
        f.write(struct.pack(f'>{len(fused_w)}f', *fused_w))
        f.write(struct.pack(f'>{len(fused_b)}f', *fused_b))

    print("YOLOv8n export complete.")

def export_model():
    export_convnextv2()
    export_yolov8n()

if __name__ == "__main__":
    export_model()

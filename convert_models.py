import argparse
import openvino as ov
import torch
import torchvision

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNet external video inference")
    parser.add_argument("--ef_model", type=str, required=False, help="Path to EF OpenVINO model (.xml)")
    parser.add_argument("--seg_model", type=str, required=True, help="Path to segmentation OpenVINO model (.xml)")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as exc:
        # PyTorch 2.6+ defaults to weights_only=True; old checkpoints may need full unpickling.
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Handle DataParallel/non-DataParallel key mismatch.
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[7:]] = value
        else:
            stripped[key] = value
    try:
        model.load_state_dict(stripped)
        return
    except RuntimeError:
        pass

    prefixed = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            prefixed[key] = value
        else:
            prefixed[f"module.{key}"] = value
    model.load_state_dict(prefixed)

def build_seg_model(weights_path: str) -> torch.nn.Module:
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )
    load_checkpoint(model, weights_path)
    model.eval()
    
    # TODO later...
    # Static export - would have fastest inference but... 
    # requires more logic later to account for static max size 
    # dummy_input = torch.randn(64, 3, 112, 112)
    # torch.onnx.export(
    #     model, 
    #     dummy_input, 
    #     "deeplabv3.onnx", 
    #     export_params=True, 
    #     opset_version=11,
    #     do_constant_folding=True,
    #     input_names=['input'],
    #     output_names=['output']
    # )

    # Dynamic export
    onnx_model_name = "deeplabv3.onnx"
    dummy_input = torch.randn(64, 3, 112, 112)
    torch.onnx.export(
        model,         
        dummy_input,
        "deeplabv3.onnx", 
        export_params=True, 
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            }
        }
    )
    ov_seg_model = ov.convert_model(onnx_model_name)  
    return ov_seg_model    


def build_ef_model(weights_path: str) -> torch.nn.Module:
    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    load_checkpoint(model, weights_path)
    ov_model = ov.convert_model(model)
    return ov_model



args = parse_args()

ov_model = build_ef_model(args.ef_model)
ov.save_model(ov_model, "ef.xml")

ov_model = build_seg_model(args.seg_model)
ov.save_model(ov_model, "seg.xml")

"""
Export utility to convert YOLO weights to Intel OpenVINO and ONNX formats
for high-speed CPU / iGPU inference on Intel Core i3 / i5 / i7 processors.
"""
import os
import shutil
import sys


def export_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("Find My Avenue: YOLO Model Acceleration Exporter")
    print("Optimized for: Intel CPU + Intel UHD Graphics")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' is not installed in the current Python environment.")
        print("Please install requirements first: pip install -r requirements.txt")
        sys.exit(1)

    model_name = os.getenv("ROAD_OBJECT_MODEL_NAME", "yolov8n.pt")
    print(f"[*] Loading base model: {model_name}")
    model = YOLO(model_name)

    # 1. Export to Intel OpenVINO (Highest speed on Intel CPUs & iGPUs)
    openvino_target = os.path.join(models_dir, "yolov8n_openvino_model")
    print("\n[*] Exporting to Intel OpenVINO (FP16, imgsz=320)...")
    try:
        exported_path = model.export(format="openvino", half=True, imgsz=320)
        print(f"[+] OpenVINO export successful: {exported_path}")
        if os.path.abspath(exported_path) != os.path.abspath(openvino_target):
            if os.path.exists(openvino_target):
                shutil.rmtree(openvino_target)
            shutil.move(exported_path, openvino_target)
            print(f"[+] Model placed at: {openvino_target}")
    except Exception as exc:
        print(f"[!] OpenVINO export failed or 'openvino' package not installed: {exc}")
        print("    Tip: run 'pip install openvino' for Intel acceleration.")

    # 2. Export to ONNX (Universal fallback)
    onnx_target = os.path.join(models_dir, "yolov8n.onnx")
    print("\n[*] Exporting to ONNX (imgsz=320)...")
    try:
        onnx_exported = model.export(format="onnx", imgsz=320, dynamic=True)
        print(f"[+] ONNX export successful: {onnx_exported}")
        if os.path.abspath(onnx_exported) != os.path.abspath(onnx_target):
            if os.path.exists(onnx_target):
                os.remove(onnx_target)
            shutil.move(onnx_exported, onnx_target)
            print(f"[+] Model placed at: {onnx_target}")
    except Exception as exc:
        print(f"[!] ONNX export skipped or failed: {exc}")

    print("\n" + "=" * 60)
    print("Export complete! Find My Avenue will automatically detect and")
    print("use the accelerated model on subsequent video analyses.")
    print("=" * 60)


if __name__ == "__main__":
    export_models()

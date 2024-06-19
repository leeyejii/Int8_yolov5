"""
Quantize a trained YOLOv5 detection model on a detection dataset with PTQ method.

Usage:
    $ python quant.py --weights yolov5s.pt --data coco128.yaml --img 640

"""

import argparse
from pathlib import Path
import sys
import os
import io
import torch
import copy
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from copy import deepcopy
from datetime import datetime
#from datasets import LoadImagesAndLabels  # Utility for loading data
from utils.general import LOGGER, check_yaml, check_dataset, check_img_size, non_max_suppression, scale_segments
from utils.torch_utils import select_device, de_parallel
import torch.quantization
from torch.quantization import prepare, convert

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] #YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) #add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) #relative

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize YOLOv5 model')
    parser.add_argument('--weights', type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument('--data', type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device selection, "0" for GPU, "" or "cpu" for CPU')
    args = parser.parse_args()
    args.data = check_yaml(args.data)
    return args

def main(
        single_cls=False,
        task = "val"
):
    args = parse_args()

    # Model and dataset configuration
    weights = args.weights
    data = args.data
    imgsz = args.img
    batch_size = args.batch
    device = select_device(args.device)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

    model_fp32 = model.model
    
    # Prepare for quantization
    model_fp32.fuse()  # Fuse Convolution, BatchNorm, ReLU layers
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # qnnpack is optimized for mobile devices.
    torch.quantization.prepare(model_fp32, inplace=True)

    # Data
    data = check_dataset(data)  # check

    nc = 1 if single_cls else int(data["nc"])

    # Dataloader
    if pt and not single_cls:  # check --weights are trained on --data
        ncm = model_fp32.nc
        assert ncm == nc, (
            f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
            f"classes). Pass correct combination of --weights and --data that are trained together."
        )
    
    pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
    task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
    )[0]
    
    default_qconfig = torch.quantization.get_default_qconfig('qnnpack')
    m = model_fp32.eval()

    prepare_orig = prepare(m, {'' : default_qconfig})

    with torch.no_grad():
        for images, _, _, _ in dataloader:
            images = images.to(device).float()
            prepare_orig(images)  # Run calibration for quantization

    # Apply quantization
    quantized_model = torch.quantization.convert(prepare_orig, inplace=True)


    torch.save(quantized_model.state_dict(), 'quantized_best61.pt')
    
    
    


    # Save the quantized model
    #torch.save(model_quantized.state_dict(), 'yolov5s_quantized.pt')
    # Save model
    ckpt = {
        'epoch' : -1,
        'best_fitness' : None,
        'model' : quantized_model,
        'ema' : None,
        'optimizer' : None,
        'wandb_id' : None,
        'date' : datetime.now().isoformat(),
    }
    
    torch.save(ckpt, 'quantized_model.pt')
    
    print(ckpt)

if __name__ == '__main__':
    main()

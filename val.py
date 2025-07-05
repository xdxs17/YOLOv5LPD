import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.yolo import Model  # [替换 DetectMultiBackend]
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.torch_utils import select_device, time_sync

def save_one_txt(predn, save_conf, shape, file):
    """
    Saves one detection result to a txt file in normalized xywh format, optionally including confidence.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")

def save_one_json(predn, jdict, path, class_map):
    """
    Saves a single JSON detection result, including image ID, category ID, bounding box, and confidence score.
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )

def process_batch(detections, labels, iouv):
    """
    Return a correct prediction matrix given detections and labels at various IoU thresholds.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def run(
    data,
    weights=None,
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    task="val",
    device="",
    workers=8,
    single_cls=False,
    augment=False,
    verbose=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=False,
    save_json=False,
    project=ROOT / "runs/val",
    name="exp",
    exist_ok=False,
    half=True,
    model=None,
    dataloader=None,
    save_dir=Path(""),
):
    """
    Evaluates a YOLOv5 model on a dataset and logs performance metrics.
    """
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device
        half &= device.type != "cpu"
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        # [使用 Model 类加载]
        model = Model(weights, nc=check_dataset(data)['nc']).to(device)
        stride = model.stride.max()
        imgsz = check_img_size(imgsz, s=stride)
        half &= device.type != "cpu"
        model.half() if half else model.float()
        data = check_dataset(data)

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    nc = 1 if single_cls else int(data["nc"])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Dataloader
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=0.5,
            rect=False,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else {i: f'class{i}' for i in range(nc)}
    class_map = list(range(1000))  # [适配 CCPD]
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    dt = Profile(device=device), Profile(device=device), Profile(device=device)
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()
            im /= 255
            nb, _, height, width = im.shape

        # Inference
        with dt[1]:
            preds, train_out = model(im, augment=augment), None  # [移除 compute_loss]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
                confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Save
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if (verbose or nc <= 1) and len(stats):  # [简化输出，适配单类]
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(batch_size, 3, imgsz, imgsz)}" % t)

    # Save JSON
    if save_json and len(jdict):
        pred_json = str(save_dir / "predictions.json")
        LOGGER.info(f"\nSaving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

    # Return results
    model.float()
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

def parse_opt():
    """
    Parse command-line options for configuring YOLOv5 model inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/ccpd.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt

def main(opt):
    """
    Executes YOLOv5 validation tasks.
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
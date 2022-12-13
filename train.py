import math
import os
import os.path as osp
import time
from argparse import ArgumentParser
from datetime import timedelta

import torch
import wandb
from dataset import SceneTextDataset
from east_dataset import EASTDataset
from model import EAST
from seed_everything import _init_fn, seedEverything  # seed를 주는 부분
from torch import cuda
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from itertools import permutations
from shapely.geometry import Polygon
import json
import warnings

warnings.filterwarnings("ignore")

seedEverything(2022)  # seed를 주는 부분
log_step = 10  # wandb logging을 할 step입니다. 예를 들어 10 step에 1번 로깅합니다.


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "../input/data/ICDAR17_Korean"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "trained_models"),
    )

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


from inference_local import do_inference


def evaluate():
    model = EAST(pretrained=False).to("cuda")
    ckpt_fpath = osp.join("trained_models/latest.pth")
    if not osp.exists("predictions"):
        os.makedirs("predictions")
    print("Inference in progress")
    ufo_result = dict(images=dict())
    split_result = do_inference(
        model, ckpt_fpath, "../input/data/ICDAR17_Korean/val", 1024, 20
    )
    ufo_result["images"].update(split_result["images"])
    output_fname = "output.json"
    with open(osp.join("predictions", output_fname), "w") as f:
        json.dump(ufo_result, f, indent=4)
    with open("predictions/output.json") as f:
        pred = json.load(f)
    with open("../input/data/ICDAR17_Korean/ufo/val.json") as f:
        gt = json.load(f)
    total_gt_mat = []
    total_pred_mat = []
    files = gt["images"].keys()
    for f in files:
        pred_points = [
            pred["images"][f]["words"][index]["points"]
            for index in pred["images"][f]["words"].keys()
        ]
        gt_points = [
            gt["images"][f]["words"][index]["points"]
            for index in gt["images"][f]["words"].keys()
        ]

        area_recall = [
            [0 for _ in range(len(pred_points))] for _ in range(len(gt_points))
        ]
        area_precision = [
            [0 for _ in range(len(pred_points))] for _ in range(len(gt_points))
        ]

        for i in range(len(gt_points)):
            for j in range(len(pred_points)):
                polygon_gt = Polygon(gt_points[i])
                pred_points_all = list(permutations(pred_points[j], 4))
                for po in pred_points_all:
                    if Polygon(po).is_valid:
                        polygon_pred = Polygon(po)
                # polygon_pred = Polygon(pred_points[j])
                if polygon_gt.intersects(polygon_pred):
                    area_recall[i][j] = (
                        polygon_gt.intersection(polygon_pred).area / polygon_gt.area
                    )
                    area_precision[i][j] = (
                        polygon_gt.intersection(polygon_pred).area / polygon_pred.area
                    )

        pred_tmp = [0 for _ in range(len(pred_points))]
        gt_tmp = [0 for _ in range(len(gt_points))]
        pred_mat = [0 for _ in range(len(pred_points))]
        gt_mat = [0 for _ in range(len(gt_points))]
        # many to one
        for i in range(len(pred_points)):
            cnt = 0
            p_recall = 0
            p_precision = 0
            gt_visit = []
            for j in range(len(gt_points)):
                if area_recall[j][i] != 0:
                    cnt += 1
                    gt_visit.append(j)
                    p_recall += area_recall[j][i]
                    p_precision += area_precision[j][i]
            if cnt >= 2:
                pred_tmp[i] = 1
                for x in gt_visit:
                    gt_tmp[x] = 1
                p_recall /= cnt
            if p_precision >= 0.4:
                pred_mat[i] = 0.8
            if p_recall >= 0.4:
                for x in gt_visit:
                    gt_mat[x] = 0.8

        # one to many
        for i in range(len(gt_points)):
            cnt = 0
            p_recall = 0
            p_precision = 0
            pred_visit = []
            for j in range(len(pred_points)):
                if area_recall[i][j] != 0:
                    cnt += 1
                    pred_visit.append(j)
                    p_recall += area_recall[i][j]
                    p_precision += area_precision[i][j]
            if cnt >= 2:
                gt_tmp[i] = 1
                for x in pred_visit:
                    pred_tmp[x] = 1
                p_precision /= cnt
            if p_precision >= 0.4:
                for x in pred_visit:
                    pred_mat[x] = 1
            if p_recall >= 0.4:
                gt_mat[i] = 1

        # one to one
        for i in range(len(gt_points)):
            for j in range(len(pred_points)):
                if gt_tmp[i] == 0 and pred_tmp[j] == 0 and area_recall[i][j] != 0:
                    gt_mat[i] = 1
                    pred_mat[j] = 1

        total_pred_mat.extend(pred_mat)
        total_gt_mat.extend(gt_mat)
    recall = sum(total_gt_mat) / len(total_gt_mat)
    precision = sum(total_pred_mat) / len(total_pred_mat)
    f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1


def do_training(
    data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
    resume_from,
):
    wandb.init(
        # Set the team where this run will be logged
        entity="level2_object-detection-cv14",
        # Set the project where this run will be logged
        project="data-competition",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"baseline",
        # Track hyperparameters and run metadata
        config={
            "num_workers": 4,
            "image_size": 1024,
            "input_size": 512,
            "batch_size": 12,
            "learning_rate": 1e-3,
            "max_epoch": 200,
            "save_interval": 5,
        },
    )
    dataset = SceneTextDataset(
        data_dir, split="train", image_size=image_size, crop_size=input_size
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_init_fn,  # seed를 주는 부분
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    if resume_from:
        model.load_state_dict(torch.load(resume_from))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for step, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(
                train_loader
            ):
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "train/Cls loss": extra_info["cls_loss"],
                    "train/Angle loss": extra_info["angle_loss"],
                    "train/IoU loss": extra_info["iou_loss"],
                    "train/total loss": loss,
                }
                # Log metrics from your script to W&B
                if step % log_step == 0:
                    lr_dict = {"optimize/learning_rate": scheduler.get_last_lr()[0]}
                    wandb.log(lr_dict)
                    wandb.log(val_dict)
                pbar.set_postfix(val_dict)

        scheduler.step()

        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)
            )
        )

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f"{wandb.run.name}_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_fpath)
        recall, precision, f1 = evaluate()
        print(f"recall: {recall} | precision: {precision} | fl-score: {f1}")
    wandb.finish()


def main(args):
    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)

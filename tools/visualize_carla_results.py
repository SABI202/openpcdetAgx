"""
Visualizes TransFusion detection results on CARLA nuScenes-format val samples.
Renders bird's-eye-view (BEV) plots with ground-truth boxes (green) and
predicted boxes (red), saved as PNGs — no GUI/Open3D display required.

Usage (run from OpenPCDet/tools):
    python visualize_carla_results.py \
        --cfg_file cfgs/nuscenes_models/transfusion_lidar_carla.yaml \
        --ckpt /workspace/OpenPCDet/output/nuscenes_models/transfusion_lidar_carla/pretrained_run/ckpt/checkpoint_epoch_6.pth \
        --num_samples 10 \
        --out_dir /workspace/OpenPCDet/output/viz_carla \
        --score_thresh 0.3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless backend, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Visualize CARLA TransFusion results (BEV, saved as PNG)')
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset/model config yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--num_samples', type=int, default=10, help='how many val samples to visualize')
    parser.add_argument('--out_dir', type=str, default='./viz_carla', help='where to save PNGs')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='min score to draw a predicted box')
    parser.add_argument('--point_size', type=float, default=0.5, help='matplotlib scatter point size')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def box_to_corners_2d(box):
    """
    box: [x, y, z, dx, dy, dz, heading, ...]
    Returns 4x2 array of BEV (x, y) corners, ordered for a closed polygon.
    """
    x, y, dx, dy, heading = box[0], box[1], box[3], box[4], box[6]
    corners = np.array([
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
    ])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_bev(points, gt_boxes, pred_boxes, pred_scores, pred_labels, class_names,
             score_thresh, point_size, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Points: BEV scatter, colored by height for a bit of depth cue
    ax.scatter(points[:, 0], points[:, 1], s=point_size, c=points[:, 2],
               cmap='viridis', alpha=0.5, linewidths=0)

    # Ground truth boxes: green
    if gt_boxes is not None:
        for box in gt_boxes:
            if np.all(box == 0):
                continue
            corners = box_to_corners_2d(box)
            ax.add_patch(Polygon(corners, closed=True, fill=False,
                                  edgecolor='lime', linewidth=2, label='_nolegend_'))

    # Predicted boxes: red, filtered by score threshold, labeled with class + score
    if pred_boxes is not None:
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score < score_thresh:
                continue
            corners = box_to_corners_2d(box)
            ax.add_patch(Polygon(corners, closed=True, fill=False,
                                  edgecolor='red', linewidth=2, label='_nolegend_'))
            class_name = class_names[int(label) - 1] if 0 < int(label) <= len(class_names) else f'cls{label}'
            ax.text(box[0], box[1], f'{class_name}\n{score:.2f}',
                    color='red', fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(handles=[
        plt.Line2D([0], [0], color='lime', lw=2, label='Ground truth'),
        plt.Line2D([0], [0], color='red', lw=2, label='Prediction'),
    ], loc='upper right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------CARLA Result Visualization-------------------------')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the REAL val dataset (nuScenes-format, correct sweeps/GT handling),
    # not a naive raw-file reader — this ensures GT boxes and point format
    # match exactly what the model was trained/evaluated on.
    val_set, val_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=2,
        training=False,
        logger=logger,
    )
    logger.info(f'Total val samples available: {len(val_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    num_samples = min(args.num_samples, len(val_set))
    with torch.no_grad():
        for idx in range(num_samples):
            data_dict = val_set.collate_batch([val_set[idx]])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            points = data_dict['points'][:, 1:4].cpu().numpy()  # drop batch idx col, keep x,y,z
            gt_boxes = data_dict['gt_boxes'][0].cpu().numpy() if 'gt_boxes' in data_dict else None

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            save_path = out_dir / f'sample_{idx:03d}.png'
            draw_bev(
                points=points,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                class_names=cfg.CLASS_NAMES,
                score_thresh=args.score_thresh,
                point_size=args.point_size,
                title=f'Sample {idx} — green=GT, red=pred (score>{args.score_thresh})',
                save_path=save_path,
            )
            logger.info(f'Saved: {save_path}')

    logger.info(f'Done. {num_samples} visualizations saved to {out_dir}')


if __name__ == '__main__':
    main()

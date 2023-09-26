import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from camera_utils import get_ir_intrinsics, triangulate_joints

from topdown_demo_with_mmdet import process_one_image


def process_batch_image(args, imgs, detector, pose_estimator, Ks, Ts):
    pred_instances_list = []
    proj_mats = Ks @ Ts
    joints_2d_list = []
    for img in imgs:
        pred_instances = process_one_image(args, img, detector, pose_estimator, None)
        joints_2d_list.append(pred_instances.keypoints)
    joints_2d = np.concatenate(joints_2d_list, axis=0)
    print(joints_2d[0])
    print(joints_2d[1])
    joints_3d = triangulate_joints(joints_2d, Ks, Ts, np.ones((2,)))
    print(joints_3d)
    # print(pred_instances_list)
    


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection',
                        default='/home/ljh/disk_4T/cj/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_ir_right_left.py')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection',
                        default='/home/ljh/disk_4T/cj/mmdetection/work_dirs/rtmdet_tiny_ir_right_left/epoch_50.pth')
    parser.add_argument('pose_config', help='Config file for pose',
                        default='configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose', 
                        default='work_dirs/rtmpose_m_coco_hand/epoch_210.pth')
    parser.add_argument('--intrinsics-path', type=str, 
                        default='/home/ljh/disk_4T/cj/qiyuan_data/intrinsics/215122253496/multiple_cameras-camchain.yaml')
    parser.add_argument('--input-ir1', type=str, 
                        default='/home/ljh/disk_4T/cj/qiyuan_data_anno/ir_images/0905_2/wyg/drone004_stop/1693900504039.231_215122253496_0.jpg')
    parser.add_argument('--input-ir2', type=str, 
                        default='/home/ljh/disk_4T/cj/qiyuan_data_anno/ir_images/0905_2/wyg/drone004_stop/1693900504039.231_215122253496_1.jpg')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    
    args = parser.parse_args()
    
     # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))
    
    intrinsics = get_ir_intrinsics(args.intrinsics_path)
    Ts, Ds, Ks = intrinsics
    
    ir1 = cv2.imread(args.input_ir1, cv2.IMREAD_GRAYSCALE)
    ir2 = cv2.imread(args.input_ir2, cv2.IMREAD_GRAYSCALE)
    ir1 = cv2.undistort(ir1, Ks[0], Ds[0])
    ir2 = cv2.undistort(ir2, Ks[1], Ds[1])
    process_batch_image(args, [ir1, ir2], detector, pose_estimator, Ks, Ts)
    # for i in range(10):
    #     pred_instances1 = process_one_image(args, ir1, detector, pose_estimator, None)
    #     pred_instances2 = process_one_image(args, ir2, detector, pose_estimator, None)
    
    # start_time = time.time()
    # for i in range(100):
    #     process_batch_image(args, [ir1, ir2], detector, pose_estimator)
    # end_time = time.time()
    # print('FPS: {}'.format(100 / (end_time - start_time)))
    

if __name__ == '__main__':
    main()
import os
import argparse

import time
import torch

import cv2
import torch
from PIL import Image
import numpy as np

import hydra

from vot.region import RegionType
from vot.dataset import load_dataset

from tracking_wrapper_mot import DAM4SAMMOT
from visualization_utils import Visualizer


def vot2bbox(vot_region):
    if vot_region.type != RegionType.SPECIAL:
        bb = vot_region.convert(RegionType.RECTANGLE)
        if not bb.is_empty():
            return [bb.x, bb.y, bb.width, bb.height]
        return None

@torch.inference_mode()
@torch.cuda.amp.autocast()
def run_sequence(dataset_path, sequence_name, checkpoint_path, visualize):
    dataset = load_dataset(dataset_path)

    seq_names = dataset.list()
    if sequence_name is not None:
        seq_names = [sequence_name]

    tracker = DAM4SAMMOT(model_size='large', checkpoint_dir=checkpoint_path)

    for seq_name in seq_names:
        sequence = dataset[seq_name]
        objs = sequence.objects()
        objs_list = sorted(list(objs))
        sequence_len = len(sequence)

        if visualize:
            visualizer = Visualizer(seq_len=sequence_len)

        per_frame_time = []
        pred_masks = []
        for ti in range(sequence_len):
            img_vis = sequence.frame(ti).image()
            image = Image.fromarray(img_vis)
            
            if ti == 0:
                init_regions = []
                for obj_id in objs_list:
                    # obj_id is a string, e.g., 'obj1', 'obj2', 'object', ...
                    init_region = sequence.object(obj_id)[ti]
                    if init_region.type == RegionType.MASK:
                        init_mask = init_region.rasterize((0, 0, image.width-1, image.height-1))
                        init_mask = (init_mask > 0.5).astype(np.uint8)
                        init_regions.append({'obj_id': obj_id, 'mask': init_mask})
                    elif init_region.type == RegionType.RECTANGLE:
                        bb_ = [init_region.x, init_region.y, init_region.width, init_region.height]
                        init_regions.append({'obj_id': obj_id, 'bbox': bb_})
                    else:
                        print('Error: Unknown init region type:', init_region.type)
                        exit(-1)

                outputs = tracker.initialize(image, init_regions)
                pred_masks = None
            else:
                torch.cuda.synchronize()
                t_ = time.time()
                outputs = tracker.track(image)
                torch.cuda.synchronize()
                t_i = time.time() - t_
                per_frame_time.append(t_i)
                # print('%d: %.4f' % (ti, t_i))
                pred_masks = outputs['masks']
                # pred_masks: list with n_obj elements, 
                # where n_obj in the number of objects being tracked
                # each element of the list: 
                # numpy array (uint8) with zeros/ones

            if visualize:
                if pred_masks is not None:
                    visualizer.visualize(img_vis.copy(), mask=pred_masks, frame_index=ti)
                else:
                    if 'mask' in init_regions[0]:
                        msks_ = [reg['mask'] for reg in init_regions]
                        visualizer.visualize(img_vis.copy(), mask=msks_, frame_index=ti)
                    elif 'bbox' in init_regions[0]:
                        bbxs_ = [reg['bbox'] for reg in init_regions]
                        visualizer.visualize(img_vis.copy(), bbox=bbxs_, frame_index=ti)
                    else:
                        print('Warning: cannot visualize intialization.')

        hydra.core.global_hydra.GlobalHydra.instance().clear()

        print('-----------------------')
        print('    %s: %d targets' % (seq_name, len(objs_list)))
        avg_time = sum(per_frame_time) / len(per_frame_time)
        avg_speed = 1 / avg_time
        print('    Average time: %.3f' % (avg_time))
        print('    Average speed: %.1f' % (avg_speed))


def main():
    parser = argparse.ArgumentParser(description='Visualize sequence.')
    parser.add_argument('--dataset', type=str, required=True, help='VOTS23 dataset path.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory.')
    parser.add_argument('--visualize', action='store_true', help='Visualize.')
    
    args = parser.parse_args()

    run_sequence(args.dataset, args.sequence, args.checkpoint_dir, args.visualize)

if __name__ == "__main__":
    main()
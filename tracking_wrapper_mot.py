import os
import numpy as np
import cv2
import torch
import time

from collections import OrderedDict

import logging
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Rectangle

from sam2.utils.misc import fill_holes_in_mask_scores


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

def build_sam(config_file, ckpt_path=None, device="cuda", mode="eval", hydra_overrides_extra=[], apply_postprocessing=True):
    # hydra_overrides = [
    #     "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    # ]
    hydra_overrides = []
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            # "++model.fill_hole_area=8",  # [AL] commented out due to different SAM class
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def keep_largest_component(mask):
    """
    Keeps only the largest connected component from a binary mask.
    
    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.
    
    Returns:
    - filtered_mask (numpy array): Binary mask with only the largest connected component.
    """
    # Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Find the index of the largest component (excluding background)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)
    # Create a mask that contains only the largest component
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_component] = 1
    return filtered_mask

def load_confs(chkpt_path, model_size):
    if model_size == 'large':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_large.pt')
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif model_size == 'base':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_base.pt')
        model_cfg = "configs/sam2.1/sam2.1_hiera_b.yaml"
    elif model_size == 'small':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_small.pt')
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif model_size == 'tiny':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_tiny.pt')
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        print('Error: Unknown model size:', model_size)
        exit(-1)
    
    return checkpoint, model_cfg


class DAM4SAMMOT():
    def __init__(self, model_size='large', checkpoint_dir=None, offload_state_to_cpu=False):
        
        if not checkpoint_dir:
            checkpoint_dir = './checkpoints'
        checkpoint, model_cfg = load_confs(checkpoint_dir, model_size)
        
        self.sam = build_sam(model_cfg, checkpoint)

        self.input_image_size = 1024
        self.fill_hole_area = 8
        
        self._img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        self.img_width = None
        self.img_height = None

        self.frame_index = 0
        self.n_frames = None

        self.maskmem_pos_enc = None
        
        self.output_dict = {'cond_frame_outputs': {}, 
                            'non_cond_frame_outputs': {}, 
                            'maskmem_pos_enc': None, 
                            'per_obj_dict': {}}

        self.mask_inputs_per_obj = {}
        self.output_dict_per_obj = {}
        self.temp_output_dict_per_obj = {}
        self.consolidated_frame_inds = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }

        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        self.device = torch.device("cuda")
        if offload_state_to_cpu:
            self.storage_device = torch.device("cpu")
        else:
            self.storage_device = torch.device("cuda")
        
        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = True

        # MOT-specific fields
        self.per_object_outputs_all = {}
        self.per_object_obj_ptr = {}  # separate object pointers since they are updated differently
        self.next_obj_id = 1
        self.all_obj_ids = []
        self.max_batch_sz = 200  # how many objects will be processed together (should not impact tracking)
        self.update_delta = 5  # update every delta frames
        self.max_ram = 3
        self.max_drm = 3
        self.use_last = True  # always use last frame in RAM
        self.add_to_drm_next = {}  # needed for DRM update (to prevent adding twice the same frame to the memory)
    
    def _prepare_image(self, image):
        # image is RGB PIL image: values on range [0, 255]
        # normalize values, resize/pad output to (3x1024x1024)
        img = np.array(image.convert("RGB").resize((self.input_image_size, self.input_image_size)))
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # normalize
        img -= self._img_mean
        img /= self._img_std
        return img.cuda()
    
    def _get_features(self, image, num_obj=1):
        # compute backbone features
        backbone_out = self.sam.forward_image(image)
        # vision_features = backbone_out['vision_features']  # (1, 256, 64, 64)
        vision_pos_enc = backbone_out['vision_pos_enc']  # list: [(1, 256, 256, 256), (1, 256, 128, 128), (1, 256, 64, 64)]
        backbone_fpn = backbone_out['backbone_fpn']  # list: [(1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64)]
        # Note: vision_features is the same as backbone_fpn[-1]

        batch_size = num_obj
        for i, feat in enumerate(backbone_fpn):
            backbone_fpn[i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(vision_pos_enc):
            vision_pos_enc[i] = pos.expand(batch_size, -1, -1, -1)
        
        expanded_backbone_out = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc}
        features = self.sam._prepare_backbone_features(expanded_backbone_out)
        _, vision_feats, vision_pos_embeds, feat_sizes = features

        # vision_feats: [(65536, 1, 32), (16384, 1, 64), (4096, 1, 256)]
        # vision_pos_embeds: [(65536, 1, 256), (16384, 1, 256), (4096, 1, 256)]
        # feat_sizes: actual values: [(256, 256), (128, 128), (64, 64)]
        return vision_feats, vision_pos_embeds, feat_sizes

    def _get_maskmem_pos_enc(self, batch_size=1):
        expanded_maskmem_pos_enc = [
            x.expand(batch_size, -1, -1, -1) for x in self.maskmem_pos_enc
        ]
        return expanded_maskmem_pos_enc
    
    def _npmask2box(self, mask):
        # mask is a 2D numpy array in a np.uint8 format
        x_ = np.where(mask.sum(0) > 0)[0]
        y_ = np.where(mask.sum(1) > 0)[0]
        x0, x1 = x_.min(), x_.max()
        y0, y1 = y_.min(), y_.max()
        # convert to (x, y0, width, height) bbox format
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


    # *****************************************************************
    # **                        VOT Tracker                          **
    # *****************************************************************
    def initialize(self, image, init_regions):
        self.frame_index = 0
        
        if self.img_width is None or self.img_height is None:
            self.img_width = image.width
            self.img_height = image.height

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)  # (1, 3, 1024, 1024)
        
        # compute features
        feats, pos, feat_sizes = self._get_features(img)  # Note: removed number of objects

        self.object_sizes = []
        self.last_added = []

        # take all unmatched detections and put them in memory for future tracking
        for reg in init_regions:
            # support both - bbox and mask initialization
            if 'mask' in reg:
                mask = reg['mask']
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.bool)
                mask_H, mask_W = mask.shape
                mask_inputs_orig = mask[None, None]  # add batch and channel dimension
                mask_inputs_orig = mask_inputs_orig.float().to(feats[0].device)

                # resize the mask if it doesn't match the model's image size
                if mask_H != self.sam.image_size or mask_W != self.sam.image_size:
                    mask_inputs = torch.nn.functional.interpolate(
                        mask_inputs_orig,
                        size=(self.sam.image_size, self.sam.image_size),
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                    mask_inputs_ = (mask_inputs >= 0.5).float()
                else:
                    mask_inputs_ = mask_inputs_orig
                
                point_inputs_ = None

                self.object_sizes.append([])
                self.last_added.append(-1)
            elif 'bbox' in reg:
                bbox = reg['bbox']
                box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                points = torch.zeros(0, 2, dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int32)
                if points.dim() == 2:
                    points = points.unsqueeze(0)  # add batch dimension
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)  # add batch dimension
                    
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
                box_coords = box.reshape(1, 2, 2)
                box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
                box_labels = box_labels.reshape(1, 2)
                points = torch.cat([box_coords, points], dim=1)
                labels = torch.cat([box_labels, labels], dim=1)
                points = points / torch.tensor([image.width, image.height]).to(points.device)
                
                points = points * self.sam.image_size
                points = points.to(feats[0].device)
                labels = labels.to(feats[0].device)
                
                point_inputs_ = {"point_coords": points, "point_labels": labels}
                mask_inputs_ = None

                self.object_sizes.append([])
                self.last_added.append(-1)
            else:
                print('Error: Input region should be mask or rectangle.')
                exit(-1)

            output_dict_ = {'per_obj_dict': {}, 'maskmem_pos_enc': None}
            current_out = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=True,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs_,
                mask_inputs=mask_inputs_,
                output_dict=output_dict_,
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=False,  # We might need to put this on True since it is not run separately
                prev_sam_mask_logits=None,
            )          
            pred_masks_gpu = current_out["pred_masks"]

            # potentially fill holes in the predicted masks
            if self.fill_hole_area > 0:
                pred_masks_gpu = fill_holes_in_mask_scores(
                    pred_masks_gpu, self.fill_hole_area
                )
                
            pred_masks = pred_masks_gpu.to(img.device, non_blocking=True)

            high_res_masks = torch.nn.functional.interpolate(
                pred_masks,
                size=(self.sam.image_size, self.sam.image_size),
                mode="bilinear",
                align_corners=False,
            )

            maskmem_features, maskmem_pos_enc = self.sam._encode_new_memory(
                current_vision_feats=feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=current_out['object_score_logits'],
                is_mask_from_pts=True
            )

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(img.device, non_blocking=True)
            
            if self.maskmem_pos_enc is None:
                self.maskmem_pos_enc = [x[0:1].clone() for x in maskmem_pos_enc]
                maskmem_pos_enc_ = self.maskmem_pos_enc[0].to(img.device)
                self.output_dict['maskmem_pos_enc'] = maskmem_pos_enc_

            per_obj_dict = {
                "maskmem_features": maskmem_features,  # (1, 64, 64, 64)
                "pred_masks": pred_masks,  # (1, 1, 256, 256)
                "is_init": True, "frame_idx": self.frame_index, "is_drm": False
            }

            # obj_ptr dimmension: (1, 256)
            per_obj_obj_ptr_dict = {"obj_ptr": current_out["obj_ptr"], "frame_idx": self.frame_index, "is_init": True}
            
            self.per_object_outputs_all[self.next_obj_id] = [per_obj_dict]
            self.per_object_obj_ptr[self.next_obj_id] = [per_obj_obj_ptr_dict]
            self.add_to_drm_next[self.next_obj_id] = None
            self.all_obj_ids.append(self.next_obj_id)
            self.next_obj_id += 1
        
        return None
    
    def track(self, image):
        self.frame_index += 1

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)  # (1, 3, 1024, 1024)
        
        # compute features
        feats, pos, feat_sizes = self._get_features(img)  # Note: removed number of objects
        
        output_dict_ = {
            'per_obj_dict': self.per_object_outputs_all,
            'per_obj_obj_ptr_dict': self.per_object_obj_ptr,
            'maskmem_pos_enc': self.output_dict['maskmem_pos_enc'], 
            'obj_ids_list': self.all_obj_ids
            }

        # n_runs tells how many times we need to call the track function
        # this is useful especially in MOT setup, where few hundreds of objects 
        # is tracked at the same time
        # in VOT the number of objects is much lower (up to 10)
        # which means that n_runs is always 1
        n_runs = ((len(output_dict_['obj_ids_list']) - 1) // self.max_batch_sz) + 1

        current_out = None  # output structure to collect (concatenate) outputs from multiple runs
        for i in range(n_runs):
            start_obj_idx = i * self.max_batch_sz
            end_obj_idx = min(len(output_dict_['obj_ids_list']), 
                                i * self.max_batch_sz + self.max_batch_sz)

            obj_ids_list_ = output_dict_['obj_ids_list'][start_obj_idx:end_obj_idx]
            per_obj_dict_ = {}
            per_obj_obj_ptr_dict_ = {}
            for id_ in obj_ids_list_:
                per_obj_dict_[id_] = output_dict_['per_obj_dict'][id_]
                per_obj_obj_ptr_dict_[id_] = output_dict_['per_obj_obj_ptr_dict'][id_]
            output_dict_tmp = {'per_obj_dict': per_obj_dict_, 
                               'per_obj_obj_ptr_dict': per_obj_obj_ptr_dict_,
                               'maskmem_pos_enc': output_dict_['maskmem_pos_enc'], 
                               'obj_ids_list': obj_ids_list_}
            
            current_out_tmp = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=False,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict_tmp,
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            current_out_tmp['maskmem_pos_enc'] = None

            # this if is here only to support multi-run setup (when huge number of objects is tracked)
            if current_out is None:
                current_out = current_out_tmp
            else:
                current_out['pred_masks'] = torch.cat([current_out['pred_masks'], current_out_tmp['pred_masks']], 0)
                current_out['obj_ptr'] = torch.cat([current_out['obj_ptr'], current_out_tmp['obj_ptr']], 0)
                current_out['object_score_logits'] = torch.cat([current_out['object_score_logits'], current_out_tmp['object_score_logits']], 0)
                current_out['maskmem_features'] = torch.cat([current_out['maskmem_features'], current_out_tmp['maskmem_features']], 0)
            
        pred_masks_gpu = current_out["pred_masks"]  # [N_obj, 1, 256, 256]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        
        sz_ = (self.img_height, self.img_width)
        masks_out = torch.nn.functional.interpolate(pred_masks_gpu, size=sz_, mode="bilinear", align_corners=False)
        m = [(m_[0] > 0).float().cpu().numpy().astype(np.uint8) for m_ in masks_out]
        n_pixels_pos = [m_single.sum() for m_single in m]
        
        maskmem_features = current_out["maskmem_features"].to(torch.bfloat16)
        
        alternative_masks_all = torch.nn.functional.interpolate(current_out["multimasks_logits"], size=sz_, mode="bilinear", align_corners=False)
        alternative_masks_all = (alternative_masks_all > 0).detach().cpu().numpy().astype(np.uint8)
        all_ious = current_out["ious"].detach().cpu().numpy()
        
        for obj_idx, obj_id in enumerate(self.all_obj_ids):

            # check if DRM has to be updated from the element from previous frame
            if self.add_to_drm_next[obj_id]:
                obj_mem = self.per_object_outputs_all[obj_id]
                obj_mem[-1] = self.add_to_drm_next[obj_id]
                self.add_to_drm_next[obj_id] = None
                drm_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and mem_el['is_drm'])]
                if len(drm_idxs) > self.max_drm:
                    # remove from DRM if more than max DRM elements
                    obj_mem.pop(drm_idxs[0])

            # update only if object is visible
            if n_pixels_pos[obj_idx] > 0:
                # list with all memory elements for this object
                obj_mem = self.per_object_outputs_all[obj_id]

                # Update object pointers firs
                per_obj_obj_ptr_dict = {"obj_ptr": current_out["obj_ptr"][obj_idx].unsqueeze(0), 
                                        "frame_idx": self.frame_index, "is_init": False}
                obj_mem_obj_ptr = self.per_object_obj_ptr[obj_id]
                obj_mem_obj_ptr.append(per_obj_obj_ptr_dict)
                if len(obj_mem_obj_ptr) > self.sam.max_obj_ptrs_in_encoder:
                    # get first non-init frame and remove it from the list
                    rem_idx = None
                    for i, ptr_el in enumerate(obj_mem_obj_ptr):
                        if not ptr_el["is_init"]:
                            rem_idx = i
                            break
                    if rem_idx:
                        obj_mem_obj_ptr.pop(rem_idx)

                # Here the per-object update is performed
                # create object dict and append it to list
                per_obj_dict = {
                    "maskmem_features": maskmem_features[obj_idx].unsqueeze(0),  # (1, 64, 64, 64)
                    "pred_masks": pred_masks_gpu[obj_idx].unsqueeze(0).detach().cpu().numpy(),  # (1, 1, 256, 256)
                    "is_init": False, "frame_idx": self.frame_index, "is_drm": False
                }

                if self.use_last:
                    ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                    
                    if len(ram_idxs) == 0:
                        obj_mem.append(per_obj_dict)
                    elif (self.frame_index % self.update_delta) == 0:
                        if (obj_mem[ram_idxs[-1]]['frame_idx'] % self.update_delta) == 0:
                            obj_mem.append(per_obj_dict)
                        else:
                            obj_mem[ram_idxs[-1]] = per_obj_dict
                    else:
                        if (obj_mem[ram_idxs[-1]]['frame_idx'] % self.update_delta) == 0:
                            obj_mem.append(per_obj_dict)
                        else:
                            obj_mem[ram_idxs[-1]] = per_obj_dict
                else:
                    if (self.frame_index % self.update_delta) == 0:
                        obj_mem.append(per_obj_dict)
                
                # check if memory is full for this object
                # remove the oldest non-init RAM element
                ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                if len(ram_idxs) > self.max_ram and len(obj_mem) > self.sam.num_maskmem:
                    obj_mem.pop(ram_idxs[0])
                
                # update the DRM memory - but first, check if DRM is even in use
                if self.max_drm > 0:
                    # check for update the DRM part of the memory
                    m_idx = np.argmax(all_ious[obj_idx]) # Index of the chosen predicted mask
                    m_iou = all_ious[obj_idx][m_idx] # Predicted IoU of the chosen predicted mask
                    # Delete the chosen predicted mask from the list of all predicted masks, leading to only alternative masks
                    alternative_masks = [mask for i, mask in enumerate(alternative_masks_all[obj_idx]) if i != m_idx]

                    # Determine if the object ratio between the current frame and the previous frames is within a certain range
                    self.object_sizes[obj_idx].append(n_pixels_pos[obj_idx])
                    if len(self.object_sizes[obj_idx]) > 1:
                        obj_sizes_ratio = n_pixels_pos[obj_idx] / np.median([
                            size for size in self.object_sizes[obj_idx][-300:] if size >= 1
                        ][-10:])
                    else:
                        obj_sizes_ratio = -1

                    # The first condition checks if:
                    #  - the chosen predicted mask has a high predicted IoU, 
                    #  - the object size ratio is within a +- 20% range compared to the previous frames, 
                    #  - the target is present in the current frame,
                    #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
                    if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and \
                        (self.frame_index - self.last_added[obj_idx] > self.update_delta or self.last_added[obj_idx] == -1):
                        
                        # Numpy array of the chosen mask and corresponding bounding box
                        chosen_mask_np = m[obj_idx]
                        chosen_bbox = self._npmask2box(m[obj_idx])

                        # Delete the parts of the alternative masks that overlap with the chosen mask
                        alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                        # Keep only the largest connected component of the processed alternative masks
                        alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                        if len(alternative_masks) > 0:
                            # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                            alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                            # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                            alternative_bboxes = [self._npmask2box(m_) for m_ in alternative_masks]
                            # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                            ious = [calculate_overlaps([Rectangle(*chosen_bbox)], [Rectangle(*bbox)])[0] for bbox in alternative_bboxes]
                            # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                            # That would mean that there are significant differences between the chosen mask and the processed alternative masks, 
                            # leading to possible detections of distractors within alternative masks.
                            if np.min(np.array(ious)) <= 0.7:
                                self.last_added[obj_idx] = self.frame_index # Update the last added frame index
                                
                                # add element to DRM
                                per_obj_dict = {
                                    "maskmem_features": maskmem_features[obj_idx].unsqueeze(0),  # (1, 64, 64, 64)
                                    "pred_masks": pred_masks_gpu[obj_idx].unsqueeze(0).detach().cpu().numpy(),  # (1, 1, 256, 256)
                                    "is_init": False, "frame_idx": self.frame_index, "is_drm": True
                                }
                                
                                if self.frame_index == obj_mem[-1]['frame_idx']:
                                    # this frame is already in RAM; 
                                    # put into the temporary mem structure and add to DRM in the next frame
                                    self.add_to_drm_next[obj_id] = per_obj_dict
                                else:
                                    # this frame is not in RAM yet - add directly to DRM
                                    obj_mem.append(per_obj_dict)
                                    
                                    # check if memory is full for this object
                                    # remove the oldest non-init DRM element
                                    if len(obj_mem) > self.sam.num_maskmem:
                                        drm_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and mem_el['is_drm'])]
                                        if len(drm_idxs) > self.max_drm:
                                            # remove from DRM if more than max DRM elements
                                            obj_mem.pop(drm_idxs[0])
                                        else:
                                            # remove from RAM elsewhere
                                            ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                                            obj_mem.pop(ram_idxs[0])
                                    
        outputs = {'masks': m}
        return outputs

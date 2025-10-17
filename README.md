# DAM4SAM: Distractor-aware memory for SAM2

DAM4SAM (CVPR25) implementation that supports tracking multiple objects simultaneously. The original implementation of the method is available in [this repository](https://github.com/jovanavidenovic/DAM4SAM).

If you find this work interesting, please cite the original publication:
```bibtex
@InProceedings{dam4sam,
  author = {Videnovic, Jovana and Lukezic, Alan and Kristan, Matej},
  title = {A Distractor-Aware Memory for Visual Object Tracking with {SAM2}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
  pages = {24255-24264}
}
```

## Instructions

Run the following command to test it on a single VOTS2025/2024/2023 sequence:
```bash
CUDA_VISIBLE_DEVICES=0 python run_on_vot_multi.py --dataset <dataset_dir> --visualize --sequence <sequence_name>
```
Optionally, you can set also the `--checkpoint_dir` input argument passing the path to the directory containing checkpoints. 

## VOTS 2025 integration

Use the `vot_wrapper.py` script to integrate it with the vot toolkit and running experiments. More information about the VOTS challenges is available on the [official VOTS webpage](https://www.votchallenge.net/).

## Results

TODO: Add results on the VOTS2025 main and realtime challenge. 

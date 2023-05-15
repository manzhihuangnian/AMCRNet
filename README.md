# Actor-Multi-Scale Context Bidirectional Higher Order Interactive Relation Network for Spatio-Temporal Action Localization  

## notification
1. This is the pytorch version implementation of paper 'AMCRNet:Actor-Multi-scale Context Bidirectional higher order Interactive Relation Network for Spatio-Temporal Action Localization',rebuilt from mmaction
2. Trained models are provided in [R50_8x8_k400.pth](https://pan.baidu.com/s/1IZpKiOfE34mU_bJauETN3A?pwd=kesu)

## test
1. preparing data

```python
    python tools\extract_frames.py --video_dir ava_path --frame_dir saving_dir --num_processes nuber_process
```
2. run test script  
```bash
    cd workspace && sh inference_AMCRNet.sh
```

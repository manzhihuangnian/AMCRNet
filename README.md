# Actor-Multi-Scale Context Bidirectional Higher Order Interactive Relation Network for Spatio-Temporal Action Localization  

## notification
1. This is the pytorch version implementation of paper 'AMCRNet:Actor-Multi-scale Context Bidirectional higher order Interactive Relation Network for Spatio-Temporal Action Localization',rebuilt from mmaction
2. Trained models are provided in [R50_8x8_k400.pth](https://pan.baidu.com/s/1IZpKiOfE34mU_bJauETN3A?pwd=kesu)

## Update
1.We correct the typos in Eq.(5) and Eq.(6) as follows.　　

$Q_i,K_i,V_i=conv2d(I_i)$,\
$Attn_{i,j}=softmax(\frac{<Q_i,K_i>}{\sqrt{C}})$,　　　　(5)\
$H_i=\displaystyle\sum_{j}^{N+M}Attn_{i,j}*V_j$
　　
$Q_i,K_i,V_i=conv2d(\{\{H,F\}+temporal\_pos\}_i)$,\
$Attn_{i,j}=softmax(\frac{<Q_i,K_i>}{\sqrt{C}}+HMask(i,j))$,　　　　(6)\
$Ｏ_i=\displaystyle\sum_{j}^{N+M}Attn_{i,j}*V_j$



## test
1. preparing data

```python
    python tools\extract_frames.py --video_dir ava_path --frame_dir saving_dir --num_processes nuber_process
```
2. run test script  
```bash
    cd workspace && sh inference_AMCRNet.sh
```

## TODO
1. Add pre-trained weights of two-stage: R50_4x16_k400.pth、R101_8x8_k400.pth
2. Add code of one-stage

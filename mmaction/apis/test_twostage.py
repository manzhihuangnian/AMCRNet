# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
import torch
import torch.distributed as dist
import mmcv
from mmcv.runner import get_dist_info
from mmaction.models.LFB.AMCRNet_Dynamic_LFB import Dynamic_Feature_Bank
import os
from mmaction.models.transformer.transformer import bbox2roi
import shutil

def single_gpu_test_twostage(model, data_loader, LFB_cfg):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    prog_bar_twostage = mmcv.ProgressBar(len(dataset))
    img_metas=[]
    rois=[]
    LFB_cfg.pop("type")
    Dynamic_LFB = Dynamic_Feature_Bank(**LFB_cfg)
    LFB_path="./results/Dynamic_lfb_re.pkl"
    if os.path.exists(LFB_path):
        LFB_dict=torch.load(LFB_path, map_location="cpu")
        Dynamic_LFB.update(LFB_dict)

    for data in data_loader:
        if not os.path.exists(LFB_path):
            with torch.no_grad():
                features_dict, img_metas_batch, rois_batch = model(stage=1,return_loss=False, **data)
                Dynamic_LFB.update(features_dict)
        img_metas_batch=data["img_metas"][0]
        img_metas.append(img_metas_batch)
        proposals=data['proposals'][0].data[0]
        rois_batch=bbox2roi(proposals)
        rois.append(rois_batch)
        batch_size = len(img_metas_batch)
        for _ in range(batch_size):
            prog_bar.update()

    torch.save(Dynamic_LFB.cache,LFB_path)

    print("video num:", len(list(Dynamic_LFB.cache.keys())))
    print("begin two stage inference！！！")

    for img_meta, roi in zip(img_metas, rois):
        with torch.no_grad():
            result=model(stage=2, return_loss=False,img_metas=img_meta, LFB=Dynamic_LFB, roi=roi)
        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar_twostage.update()
    return results

def multi_gpu_test_twostage(model, data_loader, tmpdir=None, gpu_collect=False,LFB_cfg=None,):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        prog_bar_twostage = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    img_metas = []
    rois = []
    LFB_cfg.pop("type")
    Dynamic_LFB = Dynamic_Feature_Bank(**LFB_cfg)

    LFB_path = "./results/Dynamic_slowfast/Dynamic_lfb_MGPU.pkl"

    print(os.path.exists(LFB_path))
    print(LFB_path)

    os.makedirs(os.path.dirname(LFB_path),exist_ok=True)

    if os.path.exists(LFB_path):
        LFB_dict = torch.load(LFB_path, map_location="cpu")
        Dynamic_LFB.update(LFB_dict)

    for data in data_loader:
        if not os.path.exists(LFB_path):
            with torch.no_grad():
                features_dict, img_metas_batch, rois_batch = model(stage=1, return_loss=False, **data)
                Dynamic_LFB.update(features_dict)
        else:
            img_metas_batch = data["img_metas"][0]
            proposals = data['proposals'][0].data[0]
            rois_batch = bbox2roi(proposals)
        img_metas.append(img_metas_batch)
        rois.append(rois_batch)
        if rank == 0:
            batch_size = len(img_metas_batch)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    number=0
    if not os.path.exists(LFB_path) :
        mem_feature_dict = Dynamic_LFB.cache
        for key in mem_feature_dict.keys():
            number+=len(list(mem_feature_dict[key].keys()))
        print(f"rank {rank} befor number of video", number)
        torch.save(Dynamic_LFB.cache, os.path.join(os.path.dirname(LFB_path),f"Dynamic_lfb_MGPU_{rank}.pkl"))
        dist.barrier()
        if rank==0:
            LFB_list=os.listdir(os.path.dirname(LFB_path))
            for lfb_path_part in LFB_list:
                lfb_part = torch.load(os.path.join(os.path.dirname(LFB_path),lfb_path_part), map_location="cpu")
                Dynamic_LFB.update(lfb_part)
            torch.save(Dynamic_LFB.cache, LFB_path)
    dist.barrier()
    Dynamic_LFB.update(torch.load(LFB_path, map_location="cpu"))
    number=0

    for key in Dynamic_LFB.cache:
        number+=len(list(Dynamic_LFB[key].keys()))
    print(f"rank {rank} gather number of video", number)

    print("begin two stage inference！！！")
    for img_meta, roi in zip(img_metas, rois):
        with torch.no_grad():
            result = model(stage=2, return_loss=False, img_metas=img_meta, LFB=Dynamic_LFB, roi=roi)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar_twostage.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar_twostage.completed
            for _ in range(batch_size_all):
                prog_bar_twostage.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir, test=True)
    dist.barrier()
    del Dynamic_LFB
    if rank==0:
        print(f"rank {rank} gather ",len(results))
        shutil.rmtree(os.path.dirname(LFB_path))

    return results

def collect_results_cpu(result_part, size, tmpdir=None, test=False):  # noqa: F811
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()),
                dtype=torch.uint8,
                device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        tmpdir = osp.join(tmpdir, '.dist_test')
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # synchronizes all processes for loading pickle file
    dist.barrier()
    # collect all parts
    if test:
        if rank != 0:
            return None
    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    if not test:
        dist.barrier()
    if rank==0:
        shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):  # noqa: F811
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)),
        dtype=torch.uint8,
        device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    return None

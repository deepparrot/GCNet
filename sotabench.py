import argparse
import json
import os.path as osp
import shutil
import tempfile
import urllib.request

from sotabencheval.object_detection import COCOEvaluator

import copy
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector

# Extract val2017 zip
from torchbench.utils import extract_archive
image_dir_zip = osp.join('./.data/vision/coco', 'val2017.zip')
extract_archive(from_path=image_dir_zip, to_path='./.data/vision/coco')


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        try:
            result = results[idx]
        except IndexError:
            break
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        try:
            det, seg = results[idx]
        except IndexError:
            break
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def cached_results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files

def single_gpu_test(model, data_loader, show=False, evaluator=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))                    
        
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if i == 0:
            temp_result_files = cached_results2json(copy.deepcopy(dataset), copy.deepcopy(results), 'temp_results.pkl')
            anns = json.load(open(temp_result_files['bbox']))
            evaluator.add(anns)
            from sotabencheval.object_detection.utils import get_coco_metrics
            print(evaluator.batch_hash)
            print(evaluator.cache_exists)
            if evaluator.cache_exists:
                return results, True
        
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
            
    return results, False


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def evaluate_model(model_name, paper_arxiv_id, weights_url, weights_name, paper_results, config):
    
    evaluator = COCOEvaluator(
    root='./.data/vision/coco',
    model_name=model_name,
    paper_arxiv_id=paper_arxiv_id,
    paper_results=paper_results)

    out = 'results.pkl'
    launcher = 'none'

    if out is not None and not out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(config)
    cfg.data.test['ann_file'] = './.data/vision/coco/annotations/instances_val2017.json'
    cfg.data.test['img_prefix'] = './.data/vision/coco/val2017/'

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    local_checkpoint, _ = urllib.request.urlretrieve(
        weights_url,
        weights_name)

    # '/home/ubuntu/GCNet/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'
    checkpoint = load_checkpoint(model, local_checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, cache_exists = single_gpu_test(model, data_loader, False, evaluator)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, '')

    if cache_exists:
        print('Cache exists: %s' % (evaluator.batch_hash))
        evaluator.save()
    
    else:
        from mmdet.core import results2json

        rank, _ = get_dist_info()
        if out and rank == 0:
            print('\nwriting results to {}'.format(out))
            mmcv.dump(outputs, out)
            eval_types = ['bbox']
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = out
                else:
                    if not isinstance(outputs[0], dict):
                        result_files = results2json(dataset, outputs, out)
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = out + '.{}'.format(name)
                            result_files = results2json(dataset, outputs_,
                                                        result_file)


        anns = json.load(open(result_files['bbox']))
        evaluator.detections = []
        evaluator.add(anns)
        evaluator.save()

model_configs = []

# Results on R50-FPN with backbone (fixBN)
model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-50-FPN, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.1/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth',
     'weights_name': 'mask_rcnn_r50_fpn_1x_20181010-069fa190.pth',
     'config': './configs/gcnet/r50/mask_rcnn_r50_fpn_1x.py',
     'paper_results': {'box AP': 0.372, 'AP50': 0.590, 'AP75': 0.401}}
)
model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-50-FPN, 2x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.2/mask_rcnn_r50_fpn_2x-4615e866.pth',
     'weights_name': 'mask_rcnn_r50_fpn_2x-4615e866.pth',
     'config': './configs/gcnet/r50/mask_rcnn_r50_fpn_2x.py',
     'paper_results': None}
)
model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-50-FPN, GC r16, 2x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.3/mask_rcnn_r16_gcb_c3-c5_r50_fpn_2x-bf3a5059.pth',
     'weights_name': 'mask_rcnn_r16_gcb_c3-c5_r50_fpn_2x-bf3a5059.pth',
     'config': './configs/gcnet/r50/mask_rcnn_r16_gcb_c3-c5_r50_fpn_2x.py',
     'paper_results': {'box AP': 0.394, 'AP50': 0.616, 'AP75': 0.424}}
)
model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-50-FPN, GC r4, 2x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.3/mask_rcnn_r4_gcb_c3-c5_r50_fpn_2x-360c29f3.pth',
     'weights_name': 'mask_rcnn_r4_gcb_c3-c5_r50_fpn_2x-360c29f3.pth',
     'config': './configs/gcnet/r50/mask_rcnn_r4_gcb_c3-c5_r50_fpn_2x.py',
     'paper_results': {'box AP': 0.399, 'AP50': 0.622, 'AP75': 0.429}}
)

# Results on R101-FPN with backbone (fixBN, syncBN)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-101-FPN, fixBN, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.2/mask_rcnn_r101_fpn_1x.pth',
     'weights_name': 'mask_rcnn_r101_fpn_1x.pth',
     'config': './configs/gcnet/r101/mask_rcnn_r101_fpn_1x.py',
     'paper_results': None}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-101-FPN, syncBN, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.4/mask_rcnn_r101_fpn_syncbn_1x_20190602-b2a0e2b7.pth',
     'weights_name': 'mask_rcnn_r101_fpn_syncbn_1x_20190602-b2a0e2b7.pth',
     'config': './configs/gcnet/r101/backbone_syncbn/mask_rcnn_r101_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.398, 'AP50': 0.613, 'AP75': 0.429}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-101-FPN, syncBN, GC r16, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.4/mask_rcnn_r16_gcb_c3-c5_r101_fpn_syncbn_1x_20190602-717e6dbd.pth',
     'weights_name': 'mask_rcnn_r16_gcb_c3-c5_r101_fpn_syncbn_1x_20190602-717e6dbd.pth',
     'config': './configs/gcnet/r101/backbone_syncbn/mask_rcnn_r16_gcb_c3-c5_r101_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.411, 'AP50': 0.636, 'AP75': 0.45}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-101-FPN, syncBN, GC r4, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.4/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x_20190602-a893c718.pth',
     'weights_name': 'mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x_20190602-a893c718.pth',
     'config': './configs/gcnet/r101/backbone_syncbn/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.417, 'AP50': 0.637, 'AP75': 0.455}}
)	

# Results on X101-FPN with backbone (syncBN) - START HERE

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101-FPN, syncBN, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.5/mask_rcnn_x101_32x4d_fpn_syncbn_1x_20190602-bb8ae7e5.pth',
     'weights_name': 'mask_rcnn_x101_32x4d_fpn_syncbn_1x_20190602-bb8ae7e5.pth',
     'config': './configs/gcnet/x101/mask_rcnn_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.412, 'AP50': 0.63, 'AP75': 0.451}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101-FPN, syncBN, GC r16, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.5/mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-c28edb53.pth',
     'weights_name': 'mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-c28edb53.pth',
     'config': './configs/gcnet/x101/mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.424, 'AP50': 0.646, 'AP75': 0.465}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101-FPN, syncBN, GC r4, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.5/mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-930b3d51.pth',
     'weights_name': 'mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-930b3d51.pth',
     'config': './configs/gcnet/x101/mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.429, 'AP50': 0.652, 'AP75': 0.47}}
) 

# Results on X101-FPN with backbone + cascade (syncBN)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101-FPN, syncBN, cascade, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.5/cascade_mask_rcnn_x101_32x4d_fpn_syncbn_1x_20190602-63a800fb.pth',
     'weights_name': 'cascade_mask_rcnn_x101_32x4d_fpn_syncbn_1x_20190602-63a800fb.pth',
     'config': './configs/gcnet/x101/cascade/cascade_mask_rcnn_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.447, 'AP50': 0.63, 'AP75': 0.485}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101-FPN, syncBN, cascade, GC r16, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.5/cascade_mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-3e168d88.pth',
     'weights_name': 'cascade_mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-3e168d88.pth',
     'config': './configs/gcnet/x101/cascade/cascade_mask_rcnn_r16_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.459, 'AP50': 0.648, 'AP75': 0.50}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101-FPN, syncBN, cascade, GC r4, 1x LR)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.5/cascade_mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b579157f.pth',
     'weights_name': 'cascade_mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b579157f.pth',
     'config': './configs/gcnet/x101/cascade/cascade_mask_rcnn_r4_gcb_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.465, 'AP50': 0.654, 'AP75': 0.507}}
)

# Cascade + DCN

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101 + DCN + cascade)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.6/cascade_mask_rcnn_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-9aa8c394.pth',
     'weights_name': 'cascade_mask_rcnn_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-9aa8c394.pth',
     'config': './configs/gcnet/x101/cascade/dcn/cascade_mask_rcnn_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.471, 'AP50': 0.661, 'AP75': 0.513}}
)

model_configs.append(
    {'model_name': 'GCNet (ResNeXt-101 + DCN + cascade + GC r4)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.6/cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b4164f6b.1.pth',
     'weights_name': 'cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b4164f6b.1.pth',
     'config': './configs/gcnet/x101/cascade/dcn/cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': {'box AP': 0.479, 'AP50': 0.669, 'AP75': 0.522}}
)

model_configs.append(
    {'model_name': 'GCNet (ResNeXt-101 + DCN + cascade + GC r16)', 
     'paper_arxiv_id': '1904.11492',
     'weights_url': 'https://github.com/deepparrot/GCNet/releases/download/0.7/cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b86027a6.pth',
     'weights_name': 'cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_20190602-b86027a6.pth',
     'config': './configs/gcnet/x101/cascade/dcn/cascade_mask_rcnn_r16_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x.py',
     'paper_results': None}
)
            
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
    
for model_config in model_configs:
    evaluate_model(model_name=model_config['model_name'], 
                   paper_arxiv_id=model_config['paper_arxiv_id'],
                   weights_url=model_config['weights_url'],
                   weights_name=model_config['weights_name'],
                   paper_results=model_config['paper_results'],
                   config=model_config['config'])

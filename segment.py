# System libs
import csv
import os

# Numerical libs
import numpy as np
import torch
from scipy.io import loadmat

from material_segmentation.mit_semseg.dataset import TestDataset
from material_segmentation.mit_semseg.lib.nn import user_scattered_collate

# Our libs
from .mit_semseg.lib.nn import async_copy_to
from .mit_semseg.lib.utils import as_numpy
from .mit_semseg.utils import colorEncode

colors = loadmat('/usr/src/app/material_segmentation/data/color150.mat')['colors']
names = {}
with open('/usr/src/app/material_segmentation/data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    segment_image = np.array(pred_color) 
    # Convert RGB to BGR 
    segment_image = segment_image[:, :, ::-1].copy() 

    return segment_image


def test(segmentation_module, loader, gpu, cfg):

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            #scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                pred_tmp = pred_tmp.cpu()
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        segment_image = visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred
        )

    torch.cuda.empty_cache()
    return segment_image


def main(segmentation_module, loader_test, cfg, gpu):

    # Main loop
    segment_image = test(segmentation_module, loader_test, gpu, cfg)
    return segment_image


def generate_segment_image(img_path, cfg, segmentation_module):
    imgs = [img_path]
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    dataset_test = TestDataset(
    cfg.list_test,
    cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=1,
        drop_last=True)

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    segment_image = main(segmentation_module, loader_test, cfg, gpu=0)

    return segment_image

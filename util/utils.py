import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from statistics import mean
from numpy import ndarray
from skimage import measure
import pandas as pd
from scipy.ndimage import gaussian_filter
from clip.model import CLIP
from torchvision import transforms
import torch
import numpy as np
from dataset.mvtec import MVTecDataset
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from torchvision import transforms
from clip.model import CLIP
import copy
from torchvision import models
import math
import matplotlib.pyplot as plt
import seaborn as sns


def transform_invert(img_, transform_train):
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
    return img_


def show_cam_on_image(img, anomaly_map, alpha=0.5):
    img = np.float32(img)
    anomaly_map = np.float32(anomaly_map)
    cam = alpha * img + (1 - alpha) * anomaly_map
    return np.uint8(cam)


def cvt2heatmap(gray):
    gray = np.float32(gray)
    gray = normalize(gray)
    gray = gray * 255
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = normalize(scoremap)
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def plot_attention(attention_weights, filename, vmax=None):
    """
    """
    nrows, ncols = attention_weights.shape[0], attention_weights.shape[1]

    for row in range(nrows):
        for col in range(ncols):
            fig, ax = plt.subplots(figsize=(10, 5))

            im = ax.imshow(attention_weights[row, col],
                           cmap='viridis',
                           interpolation='nearest',
                           vmax=vmax
                           )
            ax.axis('off')
            file_path = f"{filename}_{row}_{col}.png"
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0, transparent=True, )
            plt.close()


def visualize(clip_model: CLIP, test_dataset, args, transform, device):
    cnt = 0
    with torch.no_grad():
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        cnt = 0
        for data in test_dataloader:
            img_paths = data[-1]
            labels = data[1]
            if torch.sum(labels) >= 1:
                imgs = data[0].to(device)
                _, anomaly_maps = clip_model.detect_forward(imgs, args)
                anomaly_maps = F.interpolate(anomaly_maps, size=(imgs.size(-2), imgs.size(-1)),
                                             mode='bilinear').cpu().numpy()
                anomaly_maps = np.stack([gaussian_filter(mask, sigma=2) for mask in anomaly_maps])
                anomaly_maps = anomaly_maps.reshape(anomaly_maps.shape[0], anomaly_maps.shape[2], anomaly_maps.shape[3])
                imgs = transform_invert(imgs, transform)
                gts = data[2].squeeze()
                if len(gts.shape) == 3:
                    pack = zip(imgs, anomaly_maps, gts, labels, img_paths)
                else:
                    pack = zip(imgs, anomaly_maps, labels, img_paths)
                for p in pack:
                    if p[-2] != 0:
                        print(p[-1])
                        save_file_name = '_'.join(p[-1].split('/')[-2:])
                        ano_map = cvt2heatmap(p[1])
                        img = cv2.cvtColor((p[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                                           cv2.COLOR_RGB2BGR)
                        cam_map = show_cam_on_image(img, ano_map)
                        result_path = os.path.join(args.vis_dir, '{}-shot'.format(args.fewshot),
                                                   test_dataset.dataset_name, test_dataset.category)
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)
                        if len(p) == 5:
                            gt = cvt2heatmap(p[2])
                            cam_gt = show_cam_on_image(img, gt)
                            res = np.concatenate((img, cam_gt, cam_map), axis=1)
                        else:
                            res = np.vstack((img, ano_map, cam_map))
                        img_path = os.path.join(result_path, save_file_name)
                        cv2.imwrite(img_path, res)
                        cnt += 1


def calculate_metrics(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
    max_f1 = np.max(f1_scores)
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return {"AUROC": roc, "AP": ap, 'max-F1': max_f1}


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()


        df.loc[len(df)] = {
            "pro": mean(pros),
            "fpr": fpr,
            "threshold": th
        }

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_res_str(metrics):
    score_res_str = ""
    for key, value in metrics.items():
        # score_res_str += '\n'
        for item, v in value.items():
            score_res_str += "{}_{}: {:.6f} ".format(key, item, v)
    return score_res_str


def cal_average_res(total_res):
    avergae = {}
    category_num = len(total_res)
    for res in total_res:  # every category res
        for key, ip in res.items():  # sample or pixel
            if key not in avergae:
                avergae[key] = {}
            for m, v in ip.items():
                if m not in avergae[key]:
                    avergae[key][m] = 0
                avergae[key][m] += v

    for key, ip in avergae.items():
        for m, v in ip.items():
            avergae[key][m] = v / category_num

    return avergae

from skimage.measure import label as cc_label

def structural_consistency_metrics(amaps, masks, threshold_mode="quantile", quantile=0.95, fixed_thr=0.5):
    """
    Compute structural consistency metrics for anomaly maps.

    Args:
        amaps: np.ndarray, shape [N, H, W]
            Predicted anomaly maps.
        masks: np.ndarray, shape [N, H, W]
            Binary ground-truth masks.
        threshold_mode: str
            "quantile": use per-image quantile threshold.
            "fixed": use fixed threshold.
        quantile: float
            Used when threshold_mode="quantile".
            0.95 means top 5% response regions are regarded as predicted anomaly regions.
        fixed_thr: float
            Used when threshold_mode="fixed".

    Returns:
        dict:
            CC-Num: average number of connected components. Lower is better.
            Largest-CC-Ratio: area ratio of the largest connected component. Higher is better.
            FG-CC-Num: connected component number inside GT foreground. Lower is better.
    """
    assert amaps.ndim == 3, f"amaps should be [N,H,W], but got {amaps.shape}"
    assert masks.ndim == 3, f"masks should be [N,H,W], but got {masks.shape}"
    assert amaps.shape == masks.shape, "amaps and masks should have the same shape."

    cc_nums = []
    largest_cc_ratios = []
    fg_cc_nums = []

    for amap, mask in zip(amaps, masks):
        mask = (mask > 0).astype(np.uint8)

        # 只在有异常区域的样本上统计结构连续性
        if mask.sum() == 0:
            continue

        # normalize each anomaly map to [0,1]
        amap = amap.astype(np.float32)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

        if threshold_mode == "quantile":
            thr = np.quantile(amap, quantile)
        elif threshold_mode == "fixed":
            thr = fixed_thr
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        pred_bin = (amap > thr).astype(np.uint8)

        # 如果没有预测区域，记为最差情况
        if pred_bin.sum() == 0:
            cc_nums.append(0.0)
            largest_cc_ratios.append(0.0)
            fg_cc_nums.append(0.0)
            continue

        # 1) 全图预测连通块数量
        comp = measure.label(pred_bin, connectivity=2)
        num_comp = comp.max()
        cc_nums.append(float(num_comp))

        # 2) 最大连通块面积占所有预测区域面积比例
        if num_comp > 0:
            areas = [(comp == i).sum() for i in range(1, num_comp + 1)]
            largest_cc_ratio = max(areas) / (sum(areas) + 1e-8)
        else:
            largest_cc_ratio = 0.0

        largest_cc_ratios.append(float(largest_cc_ratio))

        # 3) GT 前景内部的预测连通块数量
        fg_pred = pred_bin * mask
        fg_comp = measure.label(fg_pred, connectivity=2)
        fg_cc_nums.append(float(fg_comp.max()))

    if len(cc_nums) == 0:
        return {
            "CC-Num": 0.0,
            "Largest-CC-Ratio": 0.0,
            "FG-CC-Num": 0.0,
        }

    return {
        "CC-Num": float(np.mean(cc_nums)),
        "Largest-CC-Ratio": float(np.mean(largest_cc_ratios)),
        "FG-CC-Num": float(np.mean(fg_cc_nums)),
    }

#
# def evaluation_pixel(clip_model: CLIP, dataset_name, dataloader, args, device):
#     pixel_gt_list = []
#     pixel_score_list = []
#     sample_gt_list = []
#     sample_score_list = []
#     res = {}
#     pro = 0
#
#     # 这些数据集不计算 sample-level 分类
#     no_sample_cls_datasets = [
#         'isic',
#         'clinic',
#         'colon',
#         'kvasir',
#         'endo',
#     ]
#
#     # 这些数据集没有 pixel-level mask，不计算 Pixel / PRO
#     no_pixel_datasets = [
#         'br35h',
#         'brainmri',
#         'headct',
#         'oct2017',
#         'chest',  # <-- add
#     ]
#
#     def detect_with_flip_tta(imgs):
#         """
#         所有数据集统一使用 horizontal flip TTA。
#
#         对 mask：
#             flip 输入 -> 得到 flip_mask -> 再 flip 回来 -> 和原图 mask 平均
#
#         对 label：
#             原图 label score 和 flip 图 label score 直接平均
#         """
#         pred_label, pred_mask = clip_model.detect_forward(imgs, args)
#
#         flip_imgs = torch.flip(imgs, dims=[-1])
#         flip_label, flip_mask = clip_model.detect_forward(flip_imgs, args)
#
#         # mask 需要翻转回来，保证空间位置和原图对齐
#         flip_mask = torch.flip(flip_mask, dims=[-1])
#
#         final_label = (pred_label + flip_label) / 2.0
#         final_mask = (pred_mask + flip_mask) / 2.0
#
#         return final_label, final_mask
#
#     with torch.no_grad():
#         use_tta = getattr(args, "tta", 0)
#
#         for items in tqdm(dataloader):
#             imgs, labels, gt = items[:3]
#             imgs = imgs.to(device)
#
#             if use_tta:
#                 predict_labels, predict_masks = detect_with_flip_tta(imgs)
#             else:
#                 predict_labels, predict_masks = clip_model.detect_forward(imgs, args)
#
#             predict_masks = F.interpolate(
#                 predict_masks,
#                 size=(imgs.size(-2), imgs.size(-1)),
#                 mode='bilinear',
#                 align_corners=False
#             ).cpu().numpy()
#
#             predict_masks = np.stack([
#                 gaussian_filter(mask, sigma=4) for mask in predict_masks
#             ])
#
#             sample_gt_list.append(labels)
#             sample_score_list.append(predict_labels.cpu().numpy())
#
#             if dataset_name not in no_pixel_datasets:
#                 gt[gt > 0.5] = 1
#                 gt[gt <= 0.5] = 0
#                 gt = gt.cpu().numpy().astype(int)
#
#                 labels = np.max(gt.reshape(gt.shape[0], -1), axis=-1)
#                 pixel_gt_list.append(gt)
#                 pixel_score_list.append(predict_masks)
#
#         if dataset_name not in no_sample_cls_datasets:
#             sample_gt_list = np.concatenate(sample_gt_list)
#             sample_score_list = np.concatenate(sample_score_list)
#
#             res['Sample_CLS'] = calculate_metrics(
#                 sample_score_list,
#                 sample_gt_list
#             )
#
#         if dataset_name not in no_pixel_datasets:
#             pixel_gt_list = np.concatenate(pixel_gt_list)
#             pixel_score_list = np.concatenate(pixel_score_list)
#
#             if len(pixel_gt_list.shape) == 4:
#                 pixel_gt_list = pixel_gt_list.squeeze(1)
#
#             if len(pixel_score_list.shape) == 4:
#                 pixel_score_list = pixel_score_list.squeeze(1)
#
#             pro = compute_pro(pixel_gt_list, pixel_score_list)
#
#             res['Pixel'] = calculate_metrics(
#                 pixel_score_list.reshape(-1),
#                 pixel_gt_list.reshape(-1)
#             )
#             res['Pixel']['PRO'] = pro
#
#     return res

def evaluation_pixel(clip_model: CLIP, dataset_name, dataloader, args, device):
    pixel_gt_list = []
    pixel_score_list = []
    sample_gt_list = []
    sample_score_list = []
    res = {}
    pro = 0

    # 这些数据集不计算 sample-level 分类
    no_sample_cls_datasets = [
        'isic',
        'clinic',
        'colon',
        'kvasir',
        'endo',
    ]

    # 这些数据集没有 pixel-level mask，不计算 Pixel / PRO
    no_pixel_datasets = [
        'br35h',
        'brainmri',
        'headct',
        'oct2017',
        'chest',
    ]

    def detect_with_flip_tta(imgs):
        pred_label, pred_mask = clip_model.detect_forward(imgs, args)

        flip_imgs = torch.flip(imgs, dims=[-1])
        flip_label, flip_mask = clip_model.detect_forward(flip_imgs, args)

        flip_mask = torch.flip(flip_mask, dims=[-1])

        final_label = (pred_label + flip_label) / 2.0
        final_mask = (pred_mask + flip_mask) / 2.0

        return final_label, final_mask

    with torch.no_grad():
        use_tta = getattr(args, "tta", 0)

        for items in tqdm(dataloader):
            imgs, labels, gt = items[:3]
            imgs = imgs.to(device)

            if use_tta:
                predict_labels, predict_masks = detect_with_flip_tta(imgs)
            else:
                predict_labels, predict_masks = clip_model.detect_forward(imgs, args)

            predict_masks = F.interpolate(
                predict_masks,
                size=(imgs.size(-2), imgs.size(-1)),
                mode='bilinear',
                align_corners=False
            ).cpu().numpy()

            # 建议后续把 sigma 参数化；
            # 当前保持你原来的 sigma=4
            predict_masks = np.stack([
                gaussian_filter(mask, sigma=4) for mask in predict_masks
            ])

            # 建议转成 numpy，避免 np.concatenate 处理 torch tensor 不稳定
            sample_gt_list.append(labels.cpu().numpy())
            sample_score_list.append(predict_labels.cpu().numpy())

            if dataset_name not in no_pixel_datasets:
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                gt = gt.cpu().numpy().astype(int)

                pixel_gt_list.append(gt)
                pixel_score_list.append(predict_masks)

        if dataset_name not in no_sample_cls_datasets:
            sample_gt_list = np.concatenate(sample_gt_list)
            sample_score_list = np.concatenate(sample_score_list)

            res['Sample_CLS'] = calculate_metrics(
                sample_score_list,
                sample_gt_list
            )

        if dataset_name not in no_pixel_datasets:
            pixel_gt_list = np.concatenate(pixel_gt_list)
            pixel_score_list = np.concatenate(pixel_score_list)

            if len(pixel_gt_list.shape) == 4:
                pixel_gt_list = pixel_gt_list.squeeze(1)

            if len(pixel_score_list.shape) == 4:
                pixel_score_list = pixel_score_list.squeeze(1)

            pro = compute_pro(pixel_gt_list, pixel_score_list)

            res['Pixel'] = calculate_metrics(
                pixel_score_list.reshape(-1),
                pixel_gt_list.reshape(-1)
            )
            res['Pixel']['PRO'] = pro

            # 新增：结构连续性指标
            struct_res = structural_consistency_metrics(
                pixel_score_list,
                pixel_gt_list,
                threshold_mode="quantile",
                quantile=0.95
            )

            res['Pixel'].update(struct_res)

    return res

def eval_all_class(clip_model: CLIP, dataset_name, test_dataset, args, logger, device):
    total_res = []
    if args.fewshot > 0:
        fewshot_dataset = copy.deepcopy(test_dataset)
        fewshot_dataset.train = True
        fewshot_dataset.fewshot = args.fewshot

    for category in test_dataset.categories:
        print(category)
        test_dataset.update(category)

        ## store memory
        if args.fewshot > 0:
            print("use few shot")
            fewshot_dataset.update(category)
            logger.info("{}, {}".format(fewshot_dataset.category, fewshot_dataset.cur_img_paths))
            few_shot_dataloader = torch.utils.data.DataLoader(fewshot_dataset,
                                                              batch_size=max(args.fewshot, args.batch_size),
                                                              shuffle=False)
            with torch.no_grad():
                for items in tqdm(few_shot_dataloader):
                    imgs = items[0].to(device)
                    # imgs = torch.cat([imgs] + [transforms.functional.rotate(imgs, 90) for degrees in [90, 180, 270]])
                    clip_model.store_memory(imgs, args)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        if args.vis != 0:
            print("visualize")
            # visualize_attention_map(clip_model, test_dataset, args, test_dataset.transform, device)
            visualize(clip_model, test_dataset, args, test_dataset.transform, device)
        else:
            category_res = evaluation_pixel(clip_model, dataset_name, test_dataloader, args, device)
            total_res.append(category_res)
            res_str = get_res_str(category_res)
            logger.info("Category {}: {}".format(category, res_str))
    if args.vis == 0:
        average_res = cal_average_res(total_res)
        average_res_str = get_res_str(average_res)
        logger.info("Average: {}".format(average_res_str))

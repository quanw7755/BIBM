import os
import torch
from torch.utils.data import Dataset
from PIL import Image


MED_CLASS_NAMES = [
    'Brain',
    'Liver',
    'Retina_RESC',
    'Retina_OCT2017',
    'Chest',
    'Histopathology',
    'CVC-300',
]

MED_CLASS_INDEX = {
    'Brain': 3,
    'Liver': 2,
    'Retina_RESC': 1,
    'Retina_OCT2017': -1,
    'Chest': -2,
    'Histopathology': -3,
    'CVC-300': 4,   # >0 means segmentation mask is available
}


# 根据你服务器上的真实目录结构调整
MED_DATASET_DIRS = {
    'Brain': os.path.join('Med', 'Brain', 'Brain_AD'),
    'Liver': os.path.join('Med', 'Liver', 'Liver_AD'),
    'Retina_RESC': os.path.join('Med', 'Retina_RESC_AD', 'Retina_RESC_AD'),
    'Retina_OCT2017': os.path.join('Med', 'Retina_OCT2017', 'Retina_OCT2017_AD'),
    'Chest': os.path.join('Med', 'Chest', 'Chest_AD'),
    'Histopathology': os.path.join('Med', 'Histopathology', 'Histopathology_AD'),
    'CVC-300': os.path.join('Med', 'CVC-300'),
}


class MedicalADDataset(Dataset):
    def __init__(
        self,
        root,
        train=False,
        category='Brain',
        transform=None,
        gt_target_transform=None,
        dataset_name='medical_ad',
    ):
        assert category in MED_CLASS_NAMES, \
            f'category: {category}, should be in {MED_CLASS_NAMES}'

        self.root = root
        self.train = train
        self.category = category
        self.categories = [category]
        self.dataset_name = dataset_name

        self.transform = transform
        self.gt_target_transform = gt_target_transform

        self.seg_flag = MED_CLASS_INDEX[category]
        self.dataset_path = self._get_dataset_path(category)

        self.x, self.y, self.mask = self.load_dataset_folder()

    def _get_dataset_path(self, category):
        dataset_path = os.path.join(self.root, MED_DATASET_DIRS[category])

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f'\n[MedicalADDataset] Dataset path not found:\n'
                f'  {dataset_path}\n\n'
                f'category = {category}\n'
                f'root = {self.root}\n'
                f'Please check MED_DATASET_DIRS in dataset/medical_ad.py.\n'
            )

        return dataset_path

    def update(self, category):
        assert category in self.categories, \
            f'category {category} not in {self.categories}'

        self.category = category
        self.seg_flag = MED_CLASS_INDEX[category]
        self.dataset_path = self._get_dataset_path(category)
        self.x, self.y, self.mask = self.load_dataset_folder()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        label = self.y[idx]
        mask_path = self.mask[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # 无像素标注数据集：直接返回全零mask
        if self.seg_flag < 0:
            h, w = img.shape[-2], img.shape[-1]
            mask = torch.zeros([1, h, w], dtype=torch.float32)
            return img, label, mask, img_path

        # 有像素标注的数据集
        if mask_path is None:
            h, w = img.shape[-2], img.shape[-1]
            mask = torch.zeros([1, h, w], dtype=torch.float32)
            label = 0
        else:
            mask_img = Image.open(mask_path).convert('L')
            if self.gt_target_transform is not None:
                mask = self.gt_target_transform(mask_img)
            else:
                import torchvision.transforms as transforms
                mask = transforms.ToTensor()(mask_img)

            # 二值化，保证后续PRO计算稳定
            mask = (mask > 0.5).float()
            label = 1 if torch.max(mask) > 0 else 0

        return img, label, mask, img_path

    def _check_dir(self, dir_path, name):
        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                f'\n[MedicalADDataset] Required directory not found: {name}\n'
                f'  {dir_path}\n\n'
                f'Current category: {self.category}\n'
                f'Current dataset_path:\n'
                f'  {self.dataset_path}\n\n'
                f'Expected structure for default mode:\n'
                f'  {self.dataset_path}/test/good/img\n'
                f'  {self.dataset_path}/test/Ungood/img\n'
                f'  {self.dataset_path}/test/Ungood/anomaly_mask  # required when seg_flag > 0\n'
                f'Expected structure for CVC-300:\n'
                f'  {self.dataset_path}/images\n'
                f'  {self.dataset_path}/masks\n'
            )

    def _list_images(self, img_dir):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        return sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(exts)
        ])

    def _build_name_to_path(self, file_list):
        """
        将文件列表构造成 {basename_without_ext: full_path}
        """
        d = {}
        for p in file_list:
            name = os.path.splitext(os.path.basename(p))[0]
            d[name] = p
        return d

    def _load_cvc300(self):
        """
        CVC-300 结构:
          dataset_path/
            images/
            masks/
        使用文件名（不含后缀）进行 image-mask 对齐
        """
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, 'images')
        mask_dir = os.path.join(self.dataset_path, 'masks')

        self._check_dir(img_dir, 'cvc300_img_dir')
        self._check_dir(mask_dir, 'cvc300_mask_dir')

        img_list = self._list_images(img_dir)
        mask_list = self._list_images(mask_dir)

        img_map = self._build_name_to_path(img_list)
        mask_map = self._build_name_to_path(mask_list)

        common_keys = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
        if len(common_keys) == 0:
            raise RuntimeError(
                f'\n[MedicalADDataset:CVC-300] No matched image-mask pairs found.\n'
                f'img_dir={img_dir}\nmask_dir={mask_dir}\n'
                f'Please make sure file names are aligned (same stem).\n'
            )

        for k in common_keys:
            x.append(img_map[k])
            y.append(1)  # CVC-300 通常为异常样本
            mask.append(mask_map[k])

        print(
            f'[MedicalADDataset] Loaded {self.category}: '
            f'normal=0, abnormal={len(x)}, total={len(x)}, seg_flag={self.seg_flag}, '
            f'path={self.dataset_path}, matched_pairs={len(common_keys)}'
        )
        return x, y, mask

    def load_dataset_folder(self):
        # ---------- CVC-300 special branch ----------
        if self.category == 'CVC-300':
            return self._load_cvc300()

        # ---------- default branch ----------
        x, y, mask = [], [], []

        normal_img_dir = os.path.join(self.dataset_path, 'test', 'good', 'img')
        abnormal_img_dir = os.path.join(self.dataset_path, 'test', 'Ungood', 'img')
        abnormal_mask_dir = os.path.join(self.dataset_path, 'test', 'Ungood', 'anomaly_mask')

        self._check_dir(normal_img_dir, 'normal_img_dir')
        self._check_dir(abnormal_img_dir, 'abnormal_img_dir')

        normal_imgs = self._list_images(normal_img_dir)
        abnormal_imgs = self._list_images(abnormal_img_dir)

        x.extend(normal_imgs)
        y.extend([0] * len(normal_imgs))
        mask.extend([None] * len(normal_imgs))

        x.extend(abnormal_imgs)
        y.extend([1] * len(abnormal_imgs))

        if self.seg_flag > 0:
            self._check_dir(abnormal_mask_dir, 'abnormal_mask_dir')
            abnormal_masks = self._list_images(abnormal_mask_dir)

            # 默认按文件名对齐（更稳）
            img_map = self._build_name_to_path(abnormal_imgs)
            mask_map = self._build_name_to_path(abnormal_masks)
            common_keys = sorted(list(set(img_map.keys()) & set(mask_map.keys())))

            if len(common_keys) != len(abnormal_imgs):
                missing = sorted(list(set(img_map.keys()) - set(mask_map.keys())))
                raise AssertionError(
                    f'\n[MedicalADDataset] abnormal image-mask mismatch.\n'
                    f'category = {self.category}\n'
                    f'abnormal images = {len(abnormal_imgs)}\n'
                    f'abnormal masks = {len(abnormal_masks)}\n'
                    f'matched pairs = {len(common_keys)}\n'
                    f'missing mask for images (first 20): {missing[:20]}\n'
                    f'abnormal_img_dir = {abnormal_img_dir}\n'
                    f'abnormal_mask_dir = {abnormal_mask_dir}\n'
                )

            matched_masks = [mask_map[k] for k in self._build_name_to_path(abnormal_imgs).keys()]
            mask.extend(matched_masks)
        else:
            mask.extend([None] * len(abnormal_imgs))

        assert len(x) == len(y) == len(mask), \
            f'\n[MedicalADDataset] number of x/y/mask should be same.\n' \
            f'len(x) = {len(x)}, len(y) = {len(y)}, len(mask) = {len(mask)}\n'

        print(
            f'[MedicalADDataset] Loaded {self.category}: '
            f'normal={len(normal_imgs)}, abnormal={len(abnormal_imgs)}, '
            f'total={len(x)}, seg_flag={self.seg_flag}, '
            f'path={self.dataset_path}'
        )

        return x, y, mask

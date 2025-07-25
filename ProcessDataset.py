import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

class ProcessDataset(Dataset):
    def __init__(self, images_path, labels_path, fixsize=224, num_classes=1, do_train = True):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_names = os.listdir(images_path)
        self.fixsize = fixsize
        self.num_classes = num_classes
        self.do_train = do_train

        self.augmentation_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.PadIfNeeded(min_height=fixsize, min_width=fixsize, border_mode=0),
            A.RandomCrop(height=fixsize, width=fixsize),
            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),

            A.OneOf([
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1),
            ], p=0.5),

            A.OneOf([
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.4),

            A.OneOf([
                A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0, p=1),  # Instead of RandomContrast
                A.HueSaturationValue(p=1),
            ], p=0.3),

            A.Lambda(mask=round_clip_0_1),
            ToTensorV2()
        ])

        self.augmentation_val = A.Compose([
            ToTensorV2()
        ])


    def __len__(self):
        return len(self.image_names)

    def read_label(self, image_name):
        gtline_txt = os.path.join(self.labels_path, image_name[:-3] + 'txt')
        with open(gtline_txt, "r") as file:
            lines_str = file.readlines()

        lines = []
        for line_str in lines_str:
            values = [int(float(val)) for val in line_str.strip().split(',')]
            x1, y1, x2, y2, h, w = values
            h_ratio = self.fixsize / h if h != 0 else 1
            w_ratio = self.fixsize / w if w != 0 else 1
          
            xp1 = int(x1 * w_ratio)
            yp1 = int(y1 * h_ratio)
            xp2 = int(x2 * w_ratio)
            yp2 = int(y2 * h_ratio)

            if yp2 > yp1:
                x1, y1, x2, y2 = xp2, yp2, xp1, yp1
            else:
                x1, y1, x2, y2 = xp1, yp1, xp2, yp2

            lines.append([(x1, y1), (x2, y2)])
        return lines

    def create_segmentation_mask(self, lines):
        mask = np.zeros((self.fixsize, self.fixsize), dtype=np.uint8)
        # mask_ = np.zeros((self.fixsize, self.fixsize, self.num_classes), dtype=np.uint8)

        line1 = lines[0]
        line2 = lines[1]
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom != 0:
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            intersection_point = (int(px), int(py))
            triangle = np.array([intersection_point, line1[0], line2[0]], np.int32)
            cv2.fillPoly(mask, [triangle], 1)

            # mask_[..., 1] = mask
            # mask_[..., 0] = 1 - mask
        return mask

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.images_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.fixsize, self.fixsize))

        lines = self.read_label(image_name)
        mask = np.expand_dims(self.create_segmentation_mask(lines), -1)
       
        if self.do_train:
            augmented = self.augmentation_train(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].permute(2, 0, 1)
        else:
            augmented = self.augmentation_val(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].permute(2, 0, 1)
           
    
        #Mask is already divided by 255, so only the image will be divide
        return image.float()/255, mask.float()
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # Número de amostras a exibir
    num_samples = 5
    train_dataset = ProcessDataset("../data/Fruticultura/train_folder/images", "../data/Fruticultura/train_folder/labels", fixsize=640, do_train=True)


    for i in range(num_samples):
        image, mask = train_dataset[i]  # image: [3, 224, 224], mask: [2, 224, 224]
        image = image.permute(1, 2, 0).float()  # [C, H, W]
        mask = mask.permute(1, 2, 0).float()

        print(image.shape, mask.shape)
        print(np.unique(image))
        image_np = (image.numpy()*255).astype(np.uint8) # Convertendo para PIL para plotar
        mask_np = (mask.numpy()*255).astype(np.uint8)  # Classe 1 da máscara


        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(image_np)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(image_np)
        ax[1].imshow(mask_np, alpha=0.2)
        ax[1].set_title("Image + Mask")
        ax[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"albumentation_examples/example_{i}.jpg")




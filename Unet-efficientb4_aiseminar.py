# %%
import os
import time

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvtransforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms as T
from tqdm.notebook import tqdm

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_id=[i for i in range(0, 4)]
# device = torch.device(f"cuda:{gpu_id}")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
device

# %%
train_dir = "./train.txt"
val_dir = "./val.txt"
test_dir = "./test.txt"

f = open(train_dir, newline="")
train = f.readlines()
train = [i.rstrip("\n") for i in train]

f = open(val_dir, newline="")
val = f.readlines()
val = [i.rstrip("\n") for i in val]

f = open(test_dir, newline="")
test = f.readlines()
test = [i.rstrip("\n") for i in test]

# Train用、Val用、Test用の画像数の確認
print(len(train), len(val), len(test))

# %%
# x_train = os.path.join(DATA_DIR, 'train')
# y_train = os.path.join(DATA_DIR, 'train_anno')

# x_valid = os.path.join(DATA_DIR, 'val')
# y_valid = os.path.join(DATA_DIR, 'val_anno')

# x_test = os.path.join(DATA_DIR, 'test')
# y_test = os.path.join(DATA_DIR, 'test_anno')

# %%
x_train = train
y_train = [s.replace("_img", "").replace("image", "mask") for s in train]

x_valid = val
y_valid = [s.replace("_img", "").replace("image", "mask") for s in val]

x_test = test
y_test = [s.replace("_img", "").replace("image", "mask") for s in test]

# %%
print(y_train)

# %%
class Dataset(BaseDataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    CLASSES = [
        "__background__",
        "pavement",
        "braille_blocks",
        "pedestrian",
        "unlabelled",
    ]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # self.mapping = {(0, 0, 0): 0,  # 0 = background
        #                 (55, 125, 34): 1,  # 1 = class 1
        #                 (0, 9, 123): 2,  # 2 = class 2
        #                 (128, 127, 38): 3, # 3 = class 3
        #                 (117, 20, 12): 4}  # 4 = class 4

        self.mapping = [
            [0, 0, 0],
            [0, 128, 0],  # 歩道
            [0, 0, 128],  # 点字ブロック
            [128, 128, 0],  # 人
            [128, 0, 0],  # unlabel
        ]

    def mask_to_class_rgb(self, mask):
        print("----mask->rgb----")
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        print("unique values rgb    ", torch.unique(mask))
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(
                1
            ).unsqueeze(2)
            validx = idx.sum(0) == 4
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

        # check the present values after mapping, in my case 0, 1, 2, 3
        print("unique values mapped ", torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])
        return mask_out

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ファイル名がマスク画像と元画像で違うため整合性を取る処理
        self.masks_fps[i] = self.masks_fps[i].replace("_img", "")
        # mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.imread(self.masks_fps[i])

        image_pad = np.zeros(
            (image.shape[0] + 8, image.shape[1], image.shape[2]), dtype=np.uint8
        )
        image_pad[0:-8] = image
        mask_pad = np.zeros(
            (mask.shape[0] + 8, mask.shape[1], mask.shape[2]), dtype=np.uint8
        )
        mask_pad[0:-8] = mask

        image = image_pad
        mask = mask_pad
        # print(mask)

        image_mask = np.zeros(mask.shape, dtype=np.int8)

        for i, c in enumerate(self.mapping):
            image_mask[np.where((mask == c).all(axis=2))] = (i, 0, 0)

        mask = image_mask[:, :, 0]
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        image = t(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.ids)


# %%
import shutil

make_dir_list = ["train", "train_anno", "val", "val_anno", "test", "test_anno"]
copy_img_list = [x_train, y_train, x_valid, y_valid, x_test, y_test]

for i in make_dir_list:
    os.makedirs(f"./{i}", exist_ok=True)
    for j in copy_img_list:
        for k in j:
            shutil.copy(f"{k}", f"./{i}/")


# %%
x_train_dir = os.path.join("./", "train")
y_train_dir = os.path.join("./", "train_anno")

x_valid_dir = os.path.join("./", "val")
y_valid_dir = os.path.join("./", "val_anno")

x_test_dir = os.path.join("./", "test")
y_test_dir = os.path.join("./", "test_anno")

# %%
# Trainデータの確認

dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=["pavement", "braille_blocks", "pedestrian", "unlabelled"],
)

image, mask = dataset[0]  # get some sample

# %%
# 画像およびマスクの表示用関数


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


# %%
# Datasetの画像とマスクの確認。

dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=["pavement", "braille_blocks", "pedestrian", "unlabelled"],
)

image, mask = dataset[0]  # get some sample
visualize(
    image=image.permute(1, 2, 0), mask=mask  # To tensorでchannel, h, wが画像表示用と異なるため、配列変換
)


# %%
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480),
    ]
    return albu.Compose(test_transform)


def test_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    # test_transform = [
    #   albu.PadIfNeeded(384, 480)
    # ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# %%
# Augumentation処理後の画像の確認

dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=["pavement", "braille_blocks", "pedestrian", "unlabelled"],
    augmentation=get_training_augmentation(),
)

image, mask = dataset[12]  # get some sample
visualize(image=image.permute(1, 2, 0), mask=mask)
print(len(dataset))
print(len(mask))

# %%
# 画像サイズの確認

image.shape

# %% [markdown]
# ## Create model and train

# %%
# 画像のクラスを設定

CLASSES = ["__background__", "pavement", "braille_blocks", "pedestrian", "unlabelled"]
len(CLASSES)

# %%
# Semantic Segmentationのモデルを設定

model = smp.Unet(
    "efficientnet-b4", encoder_weights="imagenet", classes=len(CLASSES), activation=None
)
model = torch.nn.DataParallel(model, device_ids=[2])

# %%
# Train, Validationのデータセットを作成

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# %%
print(len(train_dataset))
print(len(valid_dataset))

# %%
model

# %% [markdown]
# ## Training

# %%
# # Create dummy target image
# nb_classes = 5 - 1 # 18 classes + background
# idx = np.linspace(0., 1., nb_classes)
# cmap = plt.cm.get_cmap('viridis')
# rgb = cmap(idx, bytes=True)[:, :3]  # Remove alpha value

# h, w = 190, 100
# rgb = rgb.repeat(1000, 0)
# target = np.zeros((h*w, 3), dtype=np.uint8)
# target[:rgb.shape[0]] = rgb
# target = target.reshape(h, w, 3)

# plt.imshow(target) # Each class in 10 rows

# # Create mapping
# # Get color codes for dataset (maybe you would have to use more than a single
# # image, if it doesn't contain all classes)
# target = torch.from_numpy(target)
# colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
# target = target.permute(2, 0, 1).contiguous()

# mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

# mask = torch.empty(h, w, dtype=torch.long)
# for k in mapping:
#     # Get all indices for current class
#     idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
#     validx = (idx.sum(0) == 3)  # Check that all channels match
#     mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
# print(mask, mask.shape)

# %%
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


# %%
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=len(CLASSES)):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = (
                    torch.logical_and(true_class, true_label).sum().float().item()
                )
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


# %%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    patch=False,
):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            # print(output.shape)
            # print(mask.shape)
            # print(mask.min(), mask.max())
            loss = criterion(output, mask)
            # output_probs = F.sigmoid(output)
            # output_flat = output_probs.view(output_probs.size(0),-1)
            # mask_flat = mask.view(mask.size(0),-1)
            # print(output_flat.shape)
            # print(mask_flat.shape)
            # print(output_flat.size(0))
            # print(mask_flat.size(0))
            # loss = criterion(output_flat, mask_flat)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print(
                    "Loss Decreasing.. {:.3f} >> {:.3f} ".format(
                        min_loss, (test_loss / len(val_loader))
                    )
                )
                min_loss = test_loss / len(val_loader)
                decrease += 1
                if decrease % 5 == 0:
                    print("saving model...")
                    # torch.save(model, 'Unet-_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader))) #Train途中もモデルを保存するときは実行する

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = test_loss / len(val_loader)
                print(f"Loss Not Decrease for {not_improve} time")
                if not_improve == 20:
                    print("Loss not decrease for 20 times, Stop Training")
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print(
                "Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                "Time: {:.2f}m".format((time.time() - since) / 60),
            )

    history = {
        "train_loss": train_losses,
        "val_loss": test_losses,
        "train_miou": train_iou,
        "val_miou": val_iou,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lrs": lrs,
    }
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history


# %%
max_lr = 1e-3
epoch = 2
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader)
)

history = fit(epoch, model, train_loader, valid_loader, criterion, optimizer, sched)

# %%
torch.save(model, "Unet-efficientb4_aiseminar.pt")

# %%
def plot_loss(history):
    plt.plot(history["val_loss"], label="val", marker="o")
    plt.plot(history["train_loss"], label="train", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def plot_score(history):
    plt.plot(history["train_miou"], label="train_mIoU", marker="*")
    plt.plot(history["val_miou"], label="val_mIoU", marker="*")
    plt.title("Score per epoch")
    plt.ylabel("mean IoU")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.plot(history["train_acc"], label="train_accuracy", marker="*")
    plt.plot(history["val_acc"], label="val_accuracy", marker="*")
    plt.title("Accuracy per epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


# %%
plot_loss(history)
plot_score(history)
plot_acc(history)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ## Test best saved model

# %%
# load best saved checkpoint
model = torch.load("Unet-efficientb4_qiita.pt")

# %%
class testDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = [
        "__background__",
        "pavement",
        "braille_blocks",
        "pedestrian",
        "unlabelled",
    ]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # t = T.Compose([T.ToTensor()])
        # image = t(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.ids)


# %%
# create test dataset
test_dataset = testDataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    # preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# %%
def predict_image_mask_miou(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


# %%
def predict_image_mask_pixel(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


# %%
image, mask = test_dataset[3]
pred_mask, score = predict_image_mask_miou(model, image, mask)

# %%
image, mask = test_dataset[3]
image.shape

# %%
def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


# %%
mob_miou = miou_score(model, test_dataset)
print("Test Set mIoU", np.mean(mob_miou))

# %%
def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy


# %%
mob_acc = pixel_acc(model, test_dataset)
print("Test Set Pixel Accuracy", np.mean(mob_acc))

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(image)
ax1.set_title("Picture")

ax2.imshow(mask)
ax2.set_title("Ground truth")
ax2.set_axis_off()

ax3.imshow(pred_mask)
ax3.set_title("UNet-MobileNet | mIoU {:.3f}".format(score))
ax3.set_axis_off()

# %%
image2, mask2 = test_dataset[15]
pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(image2)
ax1.set_title("Picture")

ax2.imshow(mask2)
ax2.set_title("Ground truth")
ax2.set_axis_off()

ax3.imshow(pred_mask2)
ax3.set_title("UNet-MobileNet | mIoU {:.3f}".format(score2))
ax3.set_axis_off()

# %%
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


# %%
for i in range(4):
    n = np.random.choice(len(test_dataset))

    image2, mask2 = test_dataset[n]

    pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)

    print("UNet-EfficientNet-B4 | mIoU {:.3f}".format(score2))

    visualize(
        image=image2,
        ground_truth=mask2,
        predict_mask=pred_mask2,
    )

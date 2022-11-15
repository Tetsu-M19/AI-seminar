# %%
import glob

import torch

# %%
img_path = "./image/*.[PpJj][NnPp][GgGg]"
# 指定パスの画像ファイルのリストを取得
img_data = sorted(glob.glob(img_path))
print(len(img_data))
train_ratio = int(len(img_data) * 0.7)
val_ratio = int(len(img_data) * 0.1)
test_ratio = int(len(img_data)) - train_ratio - val_ratio

train, val, test = torch.utils.data.random_split(
    dataset=img_data,
    lengths=[train_ratio, val_ratio, test_ratio],
    generator=torch.Generator().manual_seed(42),
)

print(len(train), len(val), len(test))

# %%
f = open("./val.txt", "w")
for i in val:
    f.write(f"{i}\n")

f.close()

f = open("./train.txt", "w")
for i in train:
    f.write(f"{i}\n")

f.close()

f = open("./test.txt", "w")
for i in test:
    f.write(f"{i}\n")

f.close()

# %%

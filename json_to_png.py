# %%
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

# %%
TEMP_DIR = "./braille_blocks_only_json/"

parser = argparse.ArgumentParser()
parser.add_argument("json_dir", help="JSON files directory.")
parser.add_argument("-o", required=True, help="Output directory.")
parser.add_argument("-label_file", required=True, help="Label name file.")

args = parser.parse_args()

json_dir_path = Path(args.json_dir)
out_dir_path = Path(args.o)
label_file_path = Path(args.label_file)

assert json_dir_path.exists(), "JSON directory is not exist."
assert out_dir_path.exists(), "Output directory is not exist."
assert label_file_path.exists(), "Label file is not exist."

# Preprocessing.
correct_label_dict = {}
with open(str(label_file_path), "r") as file:
    for i, line in enumerate(file):
        correct_label_dict[line.replace("\n", "")] = i + 1
temp_dir_path = out_dir_path.joinpath(TEMP_DIR)
if not temp_dir_path.exists():
    temp_dir_path.mkdir()

# Batch process Converting json to png.
for json_file_path in tqdm.tqdm(list(json_dir_path.glob("*.json"))):
    # Execute "labelme_json_to_dataset".
    subprocess.run(
        ["labelme_json_to_dataset", str(json_file_path), "-o", str(temp_dir_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    image = Image.open(str(temp_dir_path.joinpath("label.png"))).convert("P")
    origin_color_palette = image.getpalette()
    image_array = np.array(image)

    # Fix label.
    label_indexes = {}
    with open(str(temp_dir_path.joinpath("label_names.txt")), "r") as label_file:
        for label_num, label_name in enumerate(label_file):
            label_name = label_name.replace("\n", "")
            if label_name == "_background_":
                continue
            label_indexes[label_name] = image_array == label_num

        for label_name, label_index in label_indexes.items():
            correct_label_num = correct_label_dict[label_name]
            image_array[label_index] = correct_label_num

    new_image = Image.fromarray(image_array, mode="P")
    new_image.putpalette(origin_color_palette)
    new_image.save(str(out_dir_path.joinpath(json_file_path.name).with_suffix(".png")))

# Post processing.
if temp_dir_path.exists():
    shutil.rmtree(str(temp_dir_path))

print("Conversion is finished.")

# %%

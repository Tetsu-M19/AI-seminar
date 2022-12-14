# -*- coding:utf-8 -*-
import glob
import json
import os

json_path = sorted(glob.glob("./json/*.json"))
json_filename = [os.path.basename(path) for path in json_path]
print(json_filename)

for count, i in enumerate(json_path):
    json_code = {}
    old_json = open(i, "r", encoding="utf-8_sig")
    json_data = json.load(old_json)
    for key in json_data:
        # 点字ブロックのshapeデータのみを抽出
        if key == "shapes":
            shape_data = [
                shape_data
                for shape_data in json_data["shapes"][:]
                if shape_data["label"] == "braille_blocks"
            ]
            json_code.update([(key, shape_data)])
        # ../ が追加されているのでそれを削除する
        elif key == "imagePath":
            json_code.update([(key, json_data[f"{key}"].replace("../", ""))])
        else:
            json_code.update([(key, json_data[f"{key}"])])
    old_json.close()
    new_json = open(f"./braille_blocks_only_json/{json_filename[count]}", mode="w")
    json.dump(json_code, new_json, indent=4)
    new_json.close()

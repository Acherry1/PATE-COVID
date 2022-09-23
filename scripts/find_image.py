# @Author:殷梦晗
# @Time:2022/7/5 19:51
import os
import shutil


def main():
    import numpy as np
    pairs_list = [[1, 2], [3, 4], [5, 6]]
    pairs_list = np.array(pairs_list)
    a = pairs_list[:, 0]
    b = pairs_list[:, 1]
    print(a)
    print(b)


def get_image():
    path = r"E:\archive (1)\metadata.csv"
    import pandas as pd
    a = pd.read_csv(path)
    a = a[(a["finding"] == "COVID-19")]
    # print(len(a))
    for i in a["filename"]:
        print(i)
        if i.split(".")[-1] != "gz":
            base_path = r"E:\archive (1)\images"
            new_path = r"E:\mh-dataset-0707\new_pre_dataset_covid"
            image_path = os.path.join(base_path, i)
            new_path = os.path.join(new_path, i)
            shutil.move(image_path, new_path)


def get_exist():
    result_file_path = ""
    if os.path.exists(result_file_path):
        print("不存在")


if __name__ == '__main__':
    get_image()

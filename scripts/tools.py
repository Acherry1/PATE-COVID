# @Author:殷梦晗
# @Time:2022/7/30 18:14
import os

import torch
from keras_preprocessing.image import ImageDataGenerator


def load_images(basedir, path, input_size):
    batches = ImageDataGenerator(rescale=1 / 255.).flow_from_directory(os.path.join(basedir, path),
                                                                       target_size=(input_size),
                                                                       batch_size=10000,
                                                                       class_mode='binary',
                                                                       shuffle=False,
                                                                       seed=42)

    # print(batches.class_indices)
    return batches[0][0].tolist(), batches[0][1], batches.filenames


import pandas as pd


def main():
    """
    生成带有filename的表
    """
    test_path = r"..\scripts\dataset\pretrain_2c_0727"
    images, labels, filenames = load_images(test_path, 'test', (100, 100))
    df = pd.DataFrame({"image": images, "labels": labels, "filenames": filenames})
    df.to_excel("c.xlsx", index=False)


def con(all_teachers, n_shot):
    """
    将训练结果与带有filename的表合并
    """
    c = pd.read_excel("c.xlsx")
    a = pd.read_excel(
        r"F:\chest project dataset\dataset\teacher_results_xlsx\noise_{}_{}\noise_{}_{}_1.0.xlsx".format(
            all_teachers, n_shot, all_teachers, n_shot))
    d = pd.merge(c, a, on="image")
    d.to_excel("{}_{}_epsilon_1.xlsx".format(all_teachers, n_shot), index=False)


import numpy as np


def teacher_gap():
    """
    教师之间的投票差距
    """
    a_list = []
    # for i in [7, 8, 9, 10]:
    df_source = pd.read_excel(r"..\scripts\agg_result\result_all_5_9.xlsx")
    df_source.drop_duplicates(subset="image", inplace=True)
    # print(len(df_source))
    # print(df_source.columns)
    # a_list = []
    df_source = df_source[["one_count", "zeo_count"]]
    print(df_source.columns)
    print(len(df_source))

    df_source["gap"] = np.where(df_source['one_count'] > df_source['zeo_count'],
                                df_source['one_count'] - df_source['zeo_count'],
                                df_source['zeo_count'] - df_source['one_count'])
    a = df_source["gap"].sum() - 20
    a_list.append(a)
    print(a_list)
    # print(a)
    # df_source['max'] = np.where(df_source['one_count'] > df_source['zeo_count'], df_source['one_count'],
    #                             df_source['zeo_count'])
    # df_source["gap"] = np.where(df_source["max"] > 10, 20 - df_source["max"], 10 - df_source["max"])
    # df_source.to_excel("aaa.xlsx", index=False)


from torchvision import transforms


def gap_norm():
    """
    gap的归一化表示
    """
    a_list = [10.0, 20.0, 50.0, 90.0]


def normalization(data):
    """
    归一化函数
    把所有数据归一化到[0，1]区间内，数据列表中的最大值和最小值分别映射到1和0，所以该方法一定会出现端点值0和1。
    此映射是线性映射，实质上是数据在数轴上等比缩放。

    :param data: 数据列表，数据取值范围：全体实数
    :return:
    """
    min_value = min(data)
    max_value = max(data)
    new_list = []
    for i in data:
        new_list.append((i - min_value) / (max_value - min_value))
    return new_list


import math


def softmax(data):
    """
    非线性映射归一化函数。归一化到[0, 1]区间，且和为1。归一化后的数据列依然保持原数据列中的大小顺序。
    非线性函数使用以e为底的指数函数:math.exp()。
    使用它可以把输入数据的范围区间（-∞, +∞）映射到（0, +∞），这样就可以使得该函数有能力处理负数。

    :param data: 数据列，数据的取值范围是全体实数
    :return:
    """
    exp_list = [math.exp(i) for i in data]
    sum_exp = sum(exp_list)
    new_list = []
    for i in exp_list:
        new_list.append(i / sum_exp)
    return new_list


def generator_input_mat():
    """
    生成隐私计算需要的input_mat的输入
    5个教师1200条标签
    [[1,0,1,1,1,1,1...,0],[],[],[],[]]
    将标签不统一的给统一起来
    """
    # df = pd.read_excel("result_all_5_9.xlsx")
    # # print(df)
    # a = df[["175_1_9", "175_2_9", "175_3_9", "175_4_9", "175_5_9"]].T.values.tolist()
    # print(a)
    # a = [[2, 3], [0, 5], [1, 4], [3, 2]]
    b_list = []
    for i in range(len(a)):

        if i==0:
            j = i + 1
            result = [i + j for i, j in zip(a[i], a[j])]
            # print(c)
            b_list.append(result)
        if (i % 2) == 0:
            continue
        elif i == len(a)-1:
            break
        else:
            j = i + 1
            result = [i + j for i, j in zip(a[i], a[j])]
            # print(c)
            b_list.append(result)
    print(b_list)


# df=df.drop_duplicates(["image","175_1_9","175_2_9","175_3_9","175_4_9","175_5_9"], inplace=True)
# print(df)
# for i in df["image"]:
#     a = df[df["image"] == i]
#     print(a["175_1_9"])
#     break
#     print(a)


if __name__ == '__main__':
    generator_input_mat()
    # all_teachers = 3
    # n_shot = 10
    # con(all_teachers, n_shot)
    # teacher_gap()
    # d = [1.516, 3.015, 3.196, 3.188,4.715,4.988,8.19]
    # # print(normalization(d))
    # print(softmax(d))
    # gap_norm()

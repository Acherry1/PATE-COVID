# @Author:殷梦晗
# @Time:2022/7/26 9:24
import os
import random
import shutil


def make_dir(dir_sub_path):
    """
    检查文件夹是否存在：
    存在：检查图像数量
    不存在：创建相应文件夹
    return: 返回创建好的文件夹名称,是否需要添加图像
    """
    covid_dir_path = os.path.join(dir_sub_path, "train", "covid")
    normal_dir_path = os.path.join(dir_sub_path, "train", "normal")
    if os.path.exists(dir_sub_path):
        is_add = False
    else:
        os.makedirs(covid_dir_path)
        os.makedirs(normal_dir_path)
        is_add = True
    return covid_dir_path, normal_dir_path, is_add


def check_repeat(n, root_path, shot, covid_name_list, normal_name_list,
                 covid_exits_path, normal_exits_path):
    """
    检查全局文件是否有重复
    重复则不放入
    不重复放入后，加入已存在列表
    """
    all_covid_list = os.listdir(covid_exits_path)
    all_normal_list = os.listdir(normal_exits_path)
    for i in range(1, n + 1):
        dir_sub_path = os.path.join(root_path, "t{}-{}".format(i, shot))
        covid_dir_path, normal_dir_path, is_add = make_dir(dir_sub_path)
        if not is_add:
            continue
        k = shot
        while k != 0:
            current_file_name = random.choice(all_covid_list)
            if current_file_name not in covid_name_list:
                shutil.copyfile(os.path.join(covid_exits_path, current_file_name)
                                , os.path.join(covid_dir_path, current_file_name))
                covid_name_list.append(current_file_name)
                k = k - 1
        k = shot
        while k != 0:
            current_file_name = random.choice(all_normal_list)
            if current_file_name not in normal_name_list:
                shutil.copyfile(os.path.join(normal_exits_path, current_file_name),
                                os.path.join(normal_dir_path, current_file_name))

                normal_name_list.append(current_file_name)
                k = k - 1


def main(n, shot):
    """
    划分数据集：train:160,test:66 all:226
    10-shot；20-shot
    n个教师：1,2,4,8,16;1,2,4,8
    例：1个教师一个文件夹，2个教师两个文件夹；
       10-shot每个文件夹中有20张图片；20-shot每个文件夹中有20张图像
       二分类：covid：10；normal：10；covid：20；normal：20
    计划：1.创建相应的文件夹
         2.将文件不重复的放入文件夹
    """
    covid_name_list = []
    normal_name_list = []
    root_path = r"F:\chest project dataset\auto_datasets\auto_t{}_{}".format(n, shot)
    covid_exits_path = r"F:\chest project dataset\dataset\pretrain_2c_0727\train\covid"
    normal_exits_path = r"F:\chest project dataset\dataset\pretrain_2c_0727\train\normal"
    for main_dir, sub_dir, file_name_list in os.walk(root_path):
        if str.split(main_dir, "\\")[-1] == "covid":
            print(main_dir)
            covid_name_list += file_name_list
        elif str.split(main_dir, "\\")[-1] == "normal":
            normal_name_list += file_name_list
    check_repeat(n, root_path, shot, covid_name_list, normal_name_list, covid_exits_path, normal_exits_path)


if __name__ == '__main__':
    shot = 20
    n = 2
    main(n, shot)

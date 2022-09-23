# @Author:殷梦晗
# @Time:2022/7/6 15:24
import os
import pandas as pd
import numpy as np

BASE_DIR = r"F:\chest project dataset\dataset\teacher_results_xlsx"


def aggregate_xlsx(n, shot):
    """
    聚合表
    """
    # 1.读取所有教师的xlsx
    path = os.listdir(os.path.join(BASE_DIR, r"result_{}_{}".format(n, shot)))
    # print(path)
    df_result = pd.DataFrame()
    for k, i in enumerate(path):
        df_1 = pd.read_excel(os.path.join(BASE_DIR, r"result_{}_{}".format(n, shot), i))
        if len(df_result) == 0:
            df_result["image"] = df_1["image"]
        df_result["{}".format(i.split(".")[0])] = df_1["label_{}".format(k + 1)]
    agg_path = os.path.join(BASE_DIR, r"agg_{}_{}".format(n, shot))
    if not os.path.exists(agg_path):
        os.makedirs(agg_path)
    df_result_path = os.path.join(agg_path, "result_all_{}_{}.xlsx".format(n, shot))
    df_result.to_excel(df_result_path, index=False)
    return df_result_path


def count_xlsx(df_result_path):
    """
    统计1和0的数量
    """
    # 3.统计1的数量和0的数量，加一列聚合后的唯一标签
    df_result = pd.read_excel(df_result_path)
    # print(df_result.columns)
    one_count_list = []
    zeo_count_list = []
    for i in df_result["image"]:
        one_count = sum([sum(df_result[df_result["image"] == i][k]) for k in df_result.columns if k != "image"])
        zeo_count = (len(df_result.columns) - 1) * len(df_result[df_result["image"] == i]) - one_count
        one_count_list.append(one_count)
        zeo_count_list.append(zeo_count)
    df_result["one_count"] = one_count_list
    df_result["zeo_count"] = zeo_count_list
    df_result.to_excel(df_result_path, index=False)
    return df_result_path


def pred_labels(df_result_path, n, shot):
    """
    全部教师的预测标签
    """
    # df_result_path = os.path.join(BASE_DIR, "result_all.xlsx")
    df_result = pd.read_excel(df_result_path)
    df_result.drop_duplicates(subset=["image", "one_count", "zeo_count"], inplace=True)
    df_result = df_result[["image", "one_count", "zeo_count"]]

    df_one = df_result[df_result["one_count"] >= df_result["zeo_count"]]
    df_one["teacher_labels"] = [1 for _ in range(len(df_one))]
    df_zeo = df_result[df_result["one_count"] < df_result["zeo_count"]]
    df_zeo["teacher_labels"] = [0 for _ in range(len(df_zeo))]
    df_label = pd.concat([df_one, df_zeo])
    df_result = pd.merge(df_result, df_label)

    df_label_path = os.path.join(BASE_DIR, "result_label_{}_{}.xlsx".format(n, shot))
    df_result.to_excel(df_label_path, index=False)
    return df_label_path, df_result


# ----------------添加拉普拉斯噪声--------------------
def add_noise(predicted_labels, epsilon=0.1):
    noisy_labels = []
    # print(predicted_labels)
    for predicts in predicted_labels:
        # print(predicts)
        label_counts = np.bincount(predicts, minlength=2)
        # print(label_counts)
        # 在标签上添加拉普拉斯噪声
        epsilon = epsilon
        beta = 1 / epsilon
        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)
        new_label = np.argmax(label_counts)
        noisy_labels.append(new_label)
    # print(len(noisy_labels))
    # 返回添加噪声后的标签
    return np.array(noisy_labels)


def get_noise_labels(epsilon, result_true_xlsx):
    # df_result_path = os.path.join(BASE_DIR, "result_label.xlsx")
    # df_result = pd.read_excel(df_result_path)
    a_all = []
    for i in result_true_xlsx.itertuples():
        a = [1 for _ in range(getattr(i, "one_count"))]
        a += [0 for _ in range(getattr(i, "zeo_count"))]
        a_all.append(a)
    labels_with_noise = add_noise(a_all, epsilon=epsilon)
    # a_str = "labels_with_noise_{}".format(epsilon)
    result_true_xlsx["labels_with_noise_{}".format(epsilon)] = labels_with_noise
    # result_true_xlsx.to_excel(result_true_xlsx, index=False)
    return result_true_xlsx


def acc_rate(result_true_xlsx, title):
    """
    计算聚合后的准确率
    """
    # df_result = pd.read_excel(result_xlsx)
    # df_true = pd.read_excel(true_xlsx)
    count_num = result_true_xlsx[result_true_xlsx[title] == result_true_xlsx["true_label"]]
    # assert len(df_result) == len(df_true)
    return "{:.2%}".format(len(count_num) / len(result_true_xlsx))


# ----------------------把以上的标签保存下来就可以了，训练好的教师模型已经用不到了-------------------


def main(epsilon=0.9):
    true_xlsx = r"F:\chest project dataset\dataset\true_result_xlsx\true_labels.xlsx"
    print("开始教师结果聚合--" * 3)
    df_result_path = aggregate_xlsx(n, shot)
    print("开始统计标签数量--" * 3)
    df_result_path = count_xlsx(df_result_path)
    print("开始选出标签--" * 3)
    df_label_path, df_label = pred_labels(df_result_path, n, shot)
    df_true = pd.read_excel(true_xlsx)
    result_true_xlsx = pd.merge(df_true, df_label, on="image")
    a = acc_rate(result_true_xlsx, title='teacher_labels')
    print("无噪声准确率为：{}".format(a))

    print("开始加噪，epsilon:{}".format(epsilon) * 10)
    result_true_xlsx = get_noise_labels(epsilon, result_true_xlsx)
    b = acc_rate(result_true_xlsx, title='labels_with_noise_{}'.format(epsilon))
    noise_path = os.path.join(BASE_DIR, "noise_{}_{}".format(n, shot))
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)
    result_true_xlsx.to_excel(os.path.join(noise_path, "noise_{}_{}_{}.xlsx".format(n, shot,epsilon)), index=False)
    print("epsilon={}时，准确率为：{}".format(epsilon, b))


if __name__ == '__main__':
    n = 3
    shot = 10
    main(epsilon=1.0)

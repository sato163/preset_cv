import math
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def calc_cos_similarity_freq(model, class_num, batch_size, separate_onehot=1, step_num=100):
    all_input = tf.constant([[x] for x in range(class_num)])

    freq_np = np.zeros([step_num+1])
    for index in tqdm(range(int(class_num//batch_size))):
        input = all_input[index*batch_size:(index+1)*batch_size]
        cos_similarity = model.predict_on_batch(input)
        cos_similarity = tf.cast(cos_similarity, dtype=tf.float16)
        cos_similarity = tf.reshape(cos_similarity, [(class_num*batch_size)])
        cos_similarity = tf.clip_by_value(cos_similarity, clip_value_max=1, clip_value_min=-1)

        # ヒストグラムを書くために、cos_similarityからそれぞれの値の頻度を計算
        index_freq = (cos_similarity + 1) * (step_num / 2)
        index_freq = tf.cast(index_freq, dtype=tf.int32)
        # index_freq = tf.one_hot(indices=tf.squeeze(index_freq), depth=181, dtype=tf.uint8)
        if separate_onehot > 1:
            separate_dim = int(index_freq.shape[0] / separate_onehot)
            for x in range(separate_onehot):
                sep_index_freq = index_freq[x*separate_dim:(x+1)*separate_dim]
                sep_index_freq = tf.one_hot(indices=tf.squeeze(sep_index_freq), depth=(step_num+1))
                sep_index_freq = tf.reduce_sum(sep_index_freq, axis=0)
                # 頻度配列を更新
                freq_np = freq_np + sep_index_freq.numpy()

        else:
            index_freq = tf.one_hot(indices=tf.squeeze(index_freq), depth=(step_num+1))
            index_freq = tf.reduce_sum(index_freq, axis=0)
            # 頻度配列を更新
            freq_np = freq_np + index_freq.numpy()

    # -1から1の横軸を作成
    x_list = [(x/(step_num / 2) - 1) for x in range((step_num+1))]
    # ロスを計算する関係上、カスタムレイヤ内で
    # 自分自身への内積が入る要素は1でなく0になるように処理している。
    # そのため、0の要素がクラス数分多くなっているため、過剰分を引く
    freq_np[int((step_num / 2)+1)] = freq_np[int((step_num / 2)+1)] - class_num
    freq_list = list(freq_np)

    return (x_list, freq_list)


def calc_vector_norm_freq(vector_matrix, range_max=10, step_num=10):
    vector_matrix = tf.math.square(vector_matrix)
    l2_norm = tf.math.sqrt(tf.math.reduce_sum(vector_matrix, axis=1))

    range_max = tf.math.reduce_max(l2_norm)

    # normalized_l2_norm = l2_norm / tf.math.reduce_max(l2_norm)
    normalized_l2_norm = l2_norm / range_max

    # ヒストグラムを書くために、normalized_l2_normからそれぞれの値の頻度を計算
    index_freq = normalized_l2_norm * step_num
    index_freq = tf.cast(index_freq, dtype=tf.int32)
    index_freq = tf.one_hot(indices=tf.squeeze(index_freq), depth=step_num)

    # 頻度配列
    freq_np = tf.reduce_sum(index_freq, axis=0)

    lenght_per_step = range_max / step_num
    # -1から1の横軸を作成
    x_list = [(lenght_per_step*x) for x in range(step_num)]
    # ロスを計算する関係上、カスタムレイヤ内で
    # 自分自身への内積が入る要素は1でなく0になるように処理している。
    # そのため、0の要素がクラス数分多くなっているため、過剰分を引く
    freq_list = list(freq_np)

    return (x_list, freq_list)

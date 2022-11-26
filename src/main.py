import click
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm

import evaluate
import loss
import models


@click.command()
@click.option("--class_num", type=int, default=100)
@click.option("--vector_dim", type=int, default=10)
@click.option("--batch_size", type=int, default=10)
@click.option("--epoch", type=int, default=1)
@click.option("--loss_type", type=str, default="nearest_orthogonal_loss")
@click.option("--output_dir", type=str, default="./")
@click.option("--separate_onehot", type=int, default=1)
@click.option("--evaluate_batch_size", type=int, default=100)
@click.option("--cv_cos_hist_step_num", type=int, default=100)
@click.option("--flag_fp16", is_flag=True)
def main(class_num,
         vector_dim,
         batch_size,
         epoch,
         loss_type,
         output_dir,
         separate_onehot,
         evaluate_batch_size,
         cv_cos_hist_step_num,
         flag_fp16):
    output_dir = Path(output_dir)

    if flag_fp16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
    cv_layer = models.AngulerLayer(class_num, vector_dim)
    outputs_layer = cv_layer(input_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=outputs_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    # "nearest_orthogonal_loss"
    # "nearest_orthogonal_or_more_loss"
    # "all_orthogonal_or_more_loss"
    # "all_orthogonal_loss"
    # "all_inverse_loss"
    loss_function = loss.get_loss(loss_type)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=[])
    model.summary()

    # 学習
    inputs = tf.constant([[x] for x in range(class_num)])
    dummy_label = np.zeros([class_num, 1])
    if epoch > 0:
        model.fit(inputs, dummy_label, epochs=epoch, batch_size=batch_size)

    # センターベクトルの取得
    vector_matrix = model.get_layer("anguler_layer").get_weights()[0]
    np.save((output_dir / f"cv_class_num_{class_num}_vector_dim_{vector_dim}.npy"), vector_matrix)

    # 全センターベクトル間の内積のヒストグラム
    x_list, freq_list = evaluate.calc_cos_similarity_freq(model, class_num, batch_size=evaluate_batch_size, separate_onehot=separate_onehot, step_num=cv_cos_hist_step_num)
    evaluate.plot_hist(x_list, freq_list, file=(output_dir / "cv_cos_hist.png"))

    # l2ノルムのヒストグラム
    x_list, freq_list = evaluate.calc_vector_norm_freq(vector_matrix, range_max=1, step_num=50)
    evaluate.plot_hist(x_list, freq_list, file=(output_dir / "l2_norm_hist.png"))

    # ベクトルが3次元だったら描写する
    if vector_matrix.shape[-1] == 3:
        evaluate.plot_vector(list(vector_matrix))


if __name__ == "__main__":
    main()

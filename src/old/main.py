import sys
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, vector_num, vector_dim, split_num=1):
        super().__init__()
        self.vector_dim = vector_dim
        self.vector_num = vector_num
        self.split_num = split_num
        self.partial_vector_num = int(vector_num / self.split_num)

        label_matrix = np.zeros([self.partial_vector_num, self.partial_vector_num])
        for index in range(self.partial_vector_num):
            label_matrix[index][index] = 1
        self.label_matrix = tf.constant([label_matrix], dtype=tf.float32)

    def build(self, input_shape):
        self.cv = self.add_weight(name='center_vector', shape=[self.vector_num, self.vector_dim])
        self.built = True

    def call(self, inputs, **kwargs):
        # 正規化
        norm_cv = tf.math.l2_normalize(self.cv, 1)
        
        if self.split_num == 1:
            # 転置
            transposed_cv = tf.transpose(norm_cv)
            # 自分自身との内積
            output = tf.matmul(norm_cv, transposed_cv)
            return output

        loss = 0
        for x in range(self.split_num):
            base_norm_cv = norm_cv[x*self.partial_vector_num:(x+1)*self.partial_vector_num]
            for y in range(self.split_num):
                temp_norm_cv = norm_cv[y*self.partial_vector_num:(y+1)*self.partial_vector_num]
                partial_transposed_cv = tf.transpose(temp_norm_cv)
                partial_matmul = tf.matmul(base_norm_cv, partial_transposed_cv)
                if x == y:
                    loss_matrix = tf.math.square((partial_matmul - self.label_matrix[0]))
                    loss = loss + tf.reduce_sum(loss_matrix)
                else:
                    loss_matrix = tf.math.square((partial_matmul))
                    loss = loss + tf.reduce_sum(loss_matrix)

        loss = loss / (self.vector_dim * self.vector_dim)
        return loss


def main():
    vector_num = 100
    vector_dim = 10
    epoch = 1

    split_num = 10

    dummy_input = np.zeros([1])

    cv_layer = MyLayer(vector_num, vector_dim, split_num)
    dummy_input_layer = tf.keras.layers.Input(shape=(1,))
    cv_layer = cv_layer(dummy_input_layer)
    
    model = tf.keras.models.Model(inputs=dummy_input_layer, outputs=cv_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer,
                  loss="mean_squared_error",
                  metrics=[])

    model.summary()

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_yscale("log")
    ax2 = ax1.twiny()
    ax2.set_yscale("log")
    # ax3 = ax1.twinx()

    if split_num == 1:
        label_matrix = np.zeros([vector_num, vector_num])
        for index in range(vector_num):
            label_matrix[index][index] = 1
        label_matrix = tf.constant([label_matrix])
        model.fit(dummy_input, label_matrix, epochs=epoch)
        output_matrix = model.predict(dummy_input)
        flatten = []
        for x in range(vector_num):
            for y in range(x):
                flatten.extend([output_matrix[x, y]])
        # print(output_matrix)
        output = np.array(flatten, dtype=np.float32)
        np.save("./temp.npy", output)
        flatten = np.array(flatten)
        flatten_degree = [math.degrees(math.acos(x)) for x in flatten]
        ax1.hist(flatten_degree, bins=181, range=(0, 180))
        ax2.hist(flatten, bins=181, range=(-1, 1), alpha=0.0)
    else:
        dummy_label = np.zeros([1])
        model.fit(dummy_input, dummy_label, epochs=epoch)
        output_matrix = model.get_layer("my_layer").get_weights()[0]
        norm_cv = tf.math.l2_normalize(output_matrix, 1)
        partial_vector_num = int(vector_num / split_num)
        for x in range(split_num):
            print(f"{x+1}/{split_num}")
            base_norm_cv = norm_cv[x*partial_vector_num:(x+1)*partial_vector_num]
            for y in tqdm(range(split_num)):
                temp_norm_cv = norm_cv[y*partial_vector_num:(y+1)*partial_vector_num]
                partial_transposed_cv = tf.transpose(temp_norm_cv)
                partial_matmul = tf.matmul(base_norm_cv, partial_transposed_cv)
                # partial_matmul = np.array(partial_matmul)
                flatten = []
                for x in range(partial_vector_num):
                    for y in range(x):
                        if x == y:
                            continue
                        flatten.extend([partial_matmul[x, y]])
                # flatten = [partial_matmul[x, y] for y in range(partial_vector_num) for x in range(partial_vector_num)]
                # flatten = tf.clip_by_value(flatten, -1.0, 1.0)
                flatten_radian = tf.math.acos(flatten)
                # flatten_degree = tf.math.angle(flatten_radian)
                flatten_degree = [math.degrees(x) for x in flatten_radian]
                flatten = np.array(flatten)
                flatten_degree = np.array(flatten_degree)
                ax1.hist(flatten_degree, bins=181, range=(0, 180), color="blue")
                ax2.hist(flatten, bins=181, range=(-1, 1), alpha=0.0)

    print("x")
    # ax3.plot()

    # plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()
    # [ 0.20909989 -0.4189835   0.00460873 ... -0.54085976  0.1349384  0.0793024 ]

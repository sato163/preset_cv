import sys

import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, vector_num, vector_dim, split_num=1):
        super().__init__()
        self.vector_dim = vector_dim
        self.vector_num = vector_num
        self.split_num = split_num
        self.partial_vector_num = int(vector_num / self.split_num)

        label_matrix = tf.zeros([self.partial_vector_num, self.partial_vector_num])
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


class MiniBatchAngulerLayer(tf.keras.layers.Layer):
    def __init__(self, class_num, vector_dim, batch_size=100):
        super().__init__()
        self.class_num = class_num
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.all_itr_num = int(self.class_num / self.batch_size)
        print(self.all_itr_num)
        self.index_list = [x for x in range(self.all_itr_num)]
        # sys.exit()
        self.itr_index = 0

    def build(self, input_shape):
        self.cv = self.add_weight(name='center_vector', shape=[self.class_num, self.vector_dim])
        self.built = True
        print("-----------------------------")
        print("done building cv")
        print("-----------------------------")

    def call(self, inputs, **kwargs):
        # センターベクトルを正規化
        # norm_cv.shape = (class_num, vector_num)
        norm_cv = tf.math.l2_normalize(self.cv, 1)
        # print(norm_cv)
        # sys.exit()

        # センターベクトルからミニバッチを抽出
        # mini_batch_vector = norm_cv[inputs*self.batch_size:(inputs+1)*self.batch_size]
        mini_batch_vector = norm_cv[self.batch_size*self.itr_index:self.batch_size*(1+self.itr_index)]
        self.itr_index = self.itr_index + 1
        # mini_batch_vector = tf.reshape(norm_cv[0], (self.vector_dim, 1))
        # sys.exit()

        # センターベクトルを転置
        transposed_mini_batch_vector = tf.transpose(mini_batch_vector)

        # センターベクトルとミニバッチの内積 = 角度
        matmuled_mini_batch = tf.matmul(norm_cv, transposed_mini_batch_vector)

        # sys.exit()
        # return matmuled_mini_batch

        loss = tf.reduce_sum(matmuled_mini_batch)

        # for x in range(self.split_num):
        #     base_norm_cv = norm_cv[x*self.partial_vector_num:(x+1)*self.partial_vector_num]
        #     for y in range(self.split_num):
        #         temp_norm_cv = norm_cv[y*self.partial_vector_num:(y+1)*self.partial_vector_num]
        #         partial_transposed_cv = tf.transpose(temp_norm_cv)
        #         partial_matmul = tf.matmul(base_norm_cv, partial_transposed_cv)
        #         if x == y:
        #             loss_matrix = tf.math.square((partial_matmul - self.label_matrix[0]))
        #             loss = loss + tf.reduce_sum(loss_matrix)
        #         else:
        #             loss_matrix = tf.math.square((partial_matmul))
        #             loss = loss + tf.reduce_sum(loss_matrix)
        # loss = loss / (self.vector_dim * self.vector_dim)
        return loss


class AngulerLayer(tf.keras.layers.Layer):
    def __init__(self, class_num, vector_dim):
        super().__init__()
        self.class_num = class_num
        self.vector_dim = vector_dim

    def build(self, input_shape):

        self.cv = self.add_weight(name='center_vector',
                                  shape=[self.class_num, self.vector_dim],
                                  initializer='random_normal')
        self.built = True
        print("-----------------------------")
        print("done building cv")
        print("-----------------------------")

    def call(self, inputs, **kwargs):
        # センターベクトルを正規化
        norm_cv = tf.math.l2_normalize(self.cv, 1)
        # norm_cv.shape = [class_num, vector_dim]

        # センターベクトルからミニバッチを抽出
        mini_batch_vector = tf.gather_nd(norm_cv, inputs)
        # mini_batch_vector.shape = [batch_size, vector_dim]

        # ミニバッチを転置
        transposed_mini_batch_vector = tf.transpose(mini_batch_vector)
        # transposed_mini_batch_vector.shape = [vector_dim, batch_size]

        # センターベクトルとミニバッチの内積 = 全ベクトル間の角度
        cos_similarity_matrix = tf.matmul(norm_cv, transposed_mini_batch_vector)
        # angle_matrix.shape = [class_num, batch_size]

        # 自分自身への内積は常に１となるので無視。
        # バッチサイズ１の時だけ例外処理
        if inputs.shape[0] != 1:
            gt_matrix = tf.one_hot(indices=tf.squeeze(inputs), depth=self.class_num)
        else:
            gt_matrix = tf.one_hot(indices=inputs[0], depth=self.class_num)
        # gt_matrix.shape = [batch_size, class_num]
        gt_matrix = tf.transpose(gt_matrix)
        # gt_matrix.shape = [class_num, batch_size]
        cos_similarity_matrix = cos_similarity_matrix - gt_matrix

        return cos_similarity_matrix

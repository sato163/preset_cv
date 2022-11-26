
import tensorflow as tf

# 自分以外の全ベクトルとできるだけ離したい

# y_trueは全て0のダミーデータだが、計算に使わないとエラーになるので無理やり使っている
# No gradient ---

def nearest_orthogonal_loss(angle_matrix, y_true):
    # 最も近いベクトル同士を直行させる(逆行を許さない)
    loss = tf.square(tf.reduce_max(angle_matrix, axis=0) - tf.reduce_max(y_true))
    # バッチ平均
    batch_size = angle_matrix.shape[-1]
    loss = tf.reduce_sum(loss) / batch_size
    return loss


def nearest_orthogonal_or_more_loss(angle_matrix, y_true):
    # 最も近いベクトル同士を直行以上(90度以上)にさせる
    # (90度以上になっていたら無視)
    mask_matrix = tf.cast(angle_matrix > 0, dtype=tf.float32)
    angle_matrix = angle_matrix * mask_matrix
    loss = tf.square(tf.reduce_max(angle_matrix, axis=0) - tf.reduce_max(y_true))
    # バッチ平均
    batch_size = angle_matrix.shape[-1]
    loss = tf.reduce_sum(loss) / batch_size
    return loss


def all_orthogonal_or_more_loss(angle_matrix, y_true):
    # 全ベクトル同士を直行以上(90度以上)にさせる
    # (90度以上になっている部分は無視)
    mask_matrix = tf.cast(angle_matrix > 0, dtype=tf.float32)
    angle_matrix = angle_matrix * mask_matrix
    loss = tf.square(tf.reduce_sum(angle_matrix, axis=0) - tf.reduce_max(y_true))
    # バッチ平均
    batch_size = angle_matrix.shape[-1]
    loss = tf.reduce_sum(loss) / batch_size
    return loss


def all_orthogonal_loss(angle_matrix, y_true):
    # 全ベクトル同士を直行させる(逆行を許さない)
    loss = tf.square(tf.reduce_sum(angle_matrix, axis=0) - tf.reduce_max(y_true))
    # バッチ平均
    batch_size = angle_matrix.shape[-1]
    loss = tf.reduce_sum(loss) / batch_size
    return loss


def all_inverse_loss(angle_matrix, y_true):
    # 全ベクトル同士を逆行させる
    distance = -1 - tf.reduce_sum(angle_matrix, axis=0) - tf.reduce_max(y_true)
    # 自分自身への内積は0になるように処理しているので、余分に大きくなっている分を除外する
    distance = distance + 1
    loss = tf.square(distance)
    # バッチ平均
    batch_size = angle_matrix.shape[-1]
    loss = tf.reduce_sum(loss) / batch_size
    return loss

def get_loss(loss_type):
    loss_dict = {"nearest_orthogonal_loss":nearest_orthogonal_loss,
                 "nearest_orthogonal_or_more_loss":nearest_orthogonal_or_more_loss,
                 "all_orthogonal_or_more_loss":all_orthogonal_or_more_loss,
                 "all_orthogonal_loss":all_orthogonal_loss,
                 "all_inverse_loss":all_inverse_loss
                }
    
    assert (loss_type in loss_dict)

    return loss_dict[loss_type]



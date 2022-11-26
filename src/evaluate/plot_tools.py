
import matplotlib.pyplot as plt

def coordinate_3d(axes, range_x, range_y, range_z, grid = True):
    axes.set_xlabel("x", fontsize = 16)
    axes.set_ylabel("y", fontsize = 16)
    axes.set_zlabel("z", fontsize = 16)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    axes.set_zlim(range_z[0], range_z[1])
    if grid == True:
        axes.grid()

# 3Dベクトル描画関数
def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
              vector[0], vector[1], vector[2],
              color = color, length = 1,
              arrow_length_ratio = 0.2)

def plot_vector(vector_list):
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D座標を設定
    coordinate_3d(ax, [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], grid = True)

    # 始点を設定
    loc = [0, 0, 0]

    for vector in vector_list:
        # 3Dベクトルを配置
        visual_vector_3d(ax, loc, vector, "red")

    plt.show()


def plot_hist(x_list, freq_list, file="./hist.png", flag_show=False):
    plt.figure()
    plt.yscale("log")
    plt.plot(x_list, freq_list)
    # plt.bar(x_list, freq_list)
    plt.savefig(file)
    if flag_show:
        plt.show()

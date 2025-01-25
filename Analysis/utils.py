# import module from the parent directory
import sys
import os
import numpy as np
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import imageio
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def fig_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()  # 将 figure 渲染到 canvas
    buf = canvas.buffer_rgba()  # 获取缓冲区中的 RGBA 图像数据
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)  # 转换为 numpy 数组并重塑为图像大小
    return image[:, :, :3]  # 去掉 alpha 通道，返回 RGB 图像
    
def figureList2gif(figList, gifName, fps = 0.5):
    images = []
    for fig in figList:
        images.append(fig_to_image(fig))
    imageio.mimsave(gifName, images, fps = fps)
    
if __name__ == "__main__":
    pass
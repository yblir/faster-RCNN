'''
predict.py有几个注意点
1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
3、如果想要获得预测框的坐标，可以进入detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
4、如果想要利用预测框截取下目标，可以进入detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
在原图上利用矩阵的方式进行截取。
5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入detect_image函数，在绘图部分对predicted_class进行判断，
比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
'''
import os
import tensorflow as tf
from PIL import Image

from frcnn_inference import FRCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    # 若只预测一张，可以注释上一行，把下面这行和最后的break取消注释
    # img = 'img/street.jpg'
    try:
        image = Image.open(img)
        # -------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        # -------------------------------------#
        image = image.convert("RGB")
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
    # break
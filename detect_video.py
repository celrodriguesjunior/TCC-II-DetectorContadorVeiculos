import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import numpy as np


flags.DEFINE_string('classes', 'D:/Users/Biblioteca/Documentos/FURB/TCC/yolov3/yolov3-tf2-master/data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', 'D:/Users/Biblioteca/Documentos/FURB/TCC/yolov3/yolov3-tf2-master/checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def draw_outputs2(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    contador = 0
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        central_point_x = ((x2y2[0] + x1y1[0])/2).astype(np.int32)
        central_point_y = ((x2y2[1] + x1y1[1]) / 2).astype(np.int32)
        diferenca = 8
        if(central_point_y > (wh[1]/2 - diferenca) and central_point_y < (wh[1]/2 + diferenca)):
            contador += 1
        img = cv2.circle(img, (central_point_x,central_point_y), radius=5, color=(255, 0, 0), thickness=-1)
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img, contador

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None
    contador = 0

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    logging.info('COMECOU')
    contador_total = 0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            contador += 1
            if contador >= 5:
                cv2.destroyAllWindows()
                logging.info('TERMINOU')
                logging.info("Total de carros que passaram: "+ str(contador_total))
                break
            continue


        width2 = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img = cv2.line(img, (0, int(height2 / 2)), (width2, int(height2 / 2)), (255,0,0), 8)

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img, contador = draw_outputs2(img, (boxes, scores, classes, nums), class_names)
        #img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
        #                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        contador_total += contador
        img = cv2.putText(img, "Contagem de veiculo: "+str(contador_total), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

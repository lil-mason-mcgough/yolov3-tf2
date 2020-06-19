import time
import os
import hashlib
import glob

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import tqdm

flags.DEFINE_string('data_list', './train.txt', 'Path to dataset list.')
flags.DEFINE_string('output_file', './.tfrecord', 'Path to output tfrecord file.')
flags.DEFINE_string('classes', './classes.names', 'Path to classes file.')
SUPPORTED_EXTS = ('.jpg', '.jpeg')


def build_example(annotation, class_map):
    img_raw = open(annotation['filename'], 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes.append(obj['label_idx'])
            classes_text.append(class_map[obj['label_idx']].encode('utf8'))
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example

def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
        img_width: int.
        img_height: int
    '''

    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height


def main(_argv):
    class_map = {idx: name for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    image_list = open(FLAGS.data_list).read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))
    for image in tqdm.tqdm(image_list):
        # parse annotation data
        _, img_path, boxes, labels, img_width, img_height = parse_line(image)
        img_ext = os.path.splitext(img_path)[-1]
        assert img_ext in SUPPORTED_EXTS, '{} format not supported. '\
            'Supported formats: {}'.format(img_ext, SUPPORTED_EXTS)
        annotation = {
            'filename': img_path,
            'size': {
                'width': img_width,
                'height': img_height
            },
            'object': [
                {
                    'label_idx': label_idx,
                    'bndbox': {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    },
                    'difficult': 0,
                    'truncated': 0,
                    'pose': 'Unspecified'
                }
            for label_idx, (xmin, ymin, xmax, ymax) in zip(labels, boxes)]
        }

        # write to tfrecord
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)

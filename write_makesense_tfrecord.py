import os
import glob

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import tqdm
from absl.flags import FLAGS

import tfcereal

IMG_EXT = '.png'
LABEL_EXT = '.txt'

def _write_bbox_line(img_path, label_path, img_idx, detections):
    img_c, img_r = Image.open(img_path).size
    assert os.path.isfile(img_path), \
        '{} has no corresponding file {}.'.format(label_path, img_path)
    img_line = '{} {} {} {} '.format(img_idx, img_path, img_c, img_r)
    return img_line + ' '.join(detections)

def read_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = [name for name in f.read().split('\n') if len(name) > 0]
    assert len(class_names) == len(set(class_names)), \
        'Duplicate class names in {}.'.format(class_names_path)
    return class_names

def convert_makesense_data_to_yolo(imgs_dir, labels_dir, class_names, output_file):
    img_paths = glob.glob(os.path.join(imgs_dir, '*' + IMG_EXT))
    print('Imgs found: {}'.format(len(img_paths)))
    with open(output_file, 'w') as f_out:
        print('Writing file: {}'.format(output_file))
        line_ctr = 0
        for img_path in tqdm.tqdm(img_paths):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_c, img_r = Image.open(img_path).size
            label_path = os.path.join(labels_dir, img_name + LABEL_EXT)
            with open(label_path, 'r') as f_in:
                detections = []
                for l_in in f_in:
                    pl = l_in.strip().split(' ')
                    class_idx = pl[0]
                    bbox_xywh = [
                        img_c * float(pl[1]), img_r * float(pl[2]), 
                        img_c * float(pl[3]), img_r * float(pl[4])
                    ]
                    width = bbox_xywh[2]
                    height = bbox_xywh[3]
                    bbox = [
                        bbox_xywh[0] - width / 2.,
                        bbox_xywh[1] - height / 2.,
                        bbox_xywh[0] + width / 2.,
                        bbox_xywh[1] + height / 2.
                    ]
                    bbox = list(map(lambda x: str(int(round(x))), bbox))
                    detection_str = ' '.join([class_idx] + bbox)
                    detections.append(detection_str)

                line = _write_bbox_line(img_path, label_path, line_ctr, detections)
                if len(line.split(' ')) < 9:
                    continue
                f_out.write(line + '\n')
                line_ctr += 1

def combine_dataset_paths(data_dir, list_path):
    classes = read_class_names(os.path.join(data_dir, 'classes.names'))
    imgs_dir = os.path.join(data_dir, 'makesense_images')
    labels_dir = os.path.join(data_dir, 'makesense_labels')
    if os.path.isdir(imgs_dir) and os.path.isdir(labels_dir):
        convert_makesense_data_to_yolo(imgs_dir, labels_dir, classes, 
            list_path)
    else:
        raise FileNotFoundError('Folders {} and/or {} do not exist.'.format(
            imgs_dir, labels_dir))


if __name__ == '__main__':
    data_dir = '../data/dewalt_boxes_v3'
    list_path = '../data/dewalt_boxes_v3/test.txt'
    output_dir = '../data/dewalt_boxes_v3/testing/tfrecord'

    # combine_dataset_paths(data_dir, list_path)
    data_list = open(list_path, 'r').readlines()
    img_paths = []
    labels = []
    for line in data_list:
        line = line.split(' ')
        img_paths.append(line[1])
        labels.append(' '.join(line[4:]))

    data = {
        'img': img_paths,
        'label': labels,
    }
    classes = read_class_names(os.path.join(data_dir, 'classes.names'))

    def preprocess_fn(datum):
        labels = datum['label'].split(' ')
        assert len(labels) % 5 == 0, 'Label formatting is not correct.'
        n_labels = len(labels) // 5
        idx_labels = []
        text_labels = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for i in range(n_labels):
            idx_label = int(labels[i * 5])
            idx_labels.append(idx_label)
            text_labels.append(classes[idx_label].encode('utf-8'))
            xmin.append(float(labels[i * 5 + 1]))
            ymin.append(float(labels[i * 5 + 2]))
            xmax.append(float(labels[i * 5 + 3]))
            ymax.append(float(labels[i * 5 + 4]))
        img = tf.io.encode_jpeg(
            cv2.imread(datum['img'])[:, :, ::-1],
            'rgb').numpy()
        return {
            'image/encoded': img,
            'image/object/class/label': idx_labels,
            'image/object/class/text': text_labels,
            'image/object/bbox/xmin': xmin,
            'image/object/bbox/ymin': ymin,
            'image/object/bbox/xmax': xmax,
            'image/object/bbox/ymax': ymax
        }

    def parse_example(x, size):
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)

        y_train = tf.stack([
            tf.sparse.to_dense(x['image/object/bbox/xmin']),
            tf.sparse.to_dense(x['image/object/bbox/ymin']),
            tf.sparse.to_dense(x['image/object/bbox/xmax']),
            tf.sparse.to_dense(x['image/object/bbox/ymax']),
            tf.cast(tf.sparse.to_dense(x['image/object/class/label']), tf.float32)
        ], axis=1)

        paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
        y_train = tf.pad(y_train, paddings)

        return x_train, y_train

    filenames = tfcereal.serialize_dataset(data, output_dir, 
        preprocess_fn=preprocess_fn, max_points=1600, verbose=True)
    print('Wrote files:\n{}'.format(filenames))

    # parse and display to verify
    # dataset, count = tfcereal.create_tfrecord_dataset(output_dir)
    # dataset = dataset.map(lambda x: parse_example(x, 416))

    # x_train, y_train = next(dataset.__iter__())
    # img = x_train.numpy()
    # for box in y_train:
    #     box = tuple(box.numpy())
    #     cv2.rectangle(img, box[0:2], box[2:4], (255, 0, 0), 2)
    # cv2.imwrite('output.jpg', img[:, :, ::-1])
    
import os
from glob import glob
from warnings import warn

from PIL import Image
import tqdm

IMG_EXT = '.jpg'
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

def convert_kitti_data_to_yolo(imgs_dir, labels_dir, output_data_path, class_names):
    print('Writing file: {}'.format(output_data_path))

    # get class names
    class_idxs = {name: idx for idx, name in enumerate(class_names)}

    # parse bounding boxes from kitti format
    label_paths = glob(os.path.join(labels_dir, '*' + LABEL_EXT))
    if not len(label_paths) > 0:
        warn('No labels found in {}'.format(labels_dir))
    
    with open(output_data_path, 'w') as f_out:
        line_ctr = 0
        for label_path in tqdm.tqdm(label_paths):
            # get detections part of line
            with open(label_path, 'r') as f_in:
                detections = []
                for l_in in f_in:
                    pl = l_in.split(' ')
                    class_idx = str(class_idxs[pl[0]])
                    bbox = list(map(lambda x: str(int(round(float(x)))), 
                        [pl[4], pl[5], pl[6], pl[7]]))
                    detection_str = ' '.join([class_idx] + bbox)
                    detections.append(detection_str)

            # get image part of line
            img_name = os.path.splitext(os.path.basename(label_path))[0] + IMG_EXT
            img_path = os.path.join(imgs_dir, img_name)

            line = _write_bbox_line(img_path, label_path, line_ctr, detections)
            if len(line.split(' ')) < 9:
                continue
            f_out.write(line + '\n')
            line_ctr += 1

def convert_makesense_data_to_yolo(imgs_dir, labels_dir, class_names, output_file):
    img_paths = glob(os.path.join(imgs_dir, '*' + IMG_EXT))
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

def combine_dataset_paths(data_subsets, output_data_dir, classes):
    os.makedirs(output_data_dir, exist_ok=True)
    for data_subset, out_filename in data_subsets.items():
        output_data_path = os.path.join(output_data_dir, out_filename)
        kitti_imgs_dir = os.path.join(data_subset, 'image_2')
        kitti_labels_dir = os.path.join(data_subset, 'label_2')
        convert_kitti_data_to_yolo(kitti_imgs_dir, kitti_labels_dir, output_data_path, classes)

        makesense_imgs_dir = os.path.join(data_subset, 'makesense_images')
        makesense_labels_dir = os.path.join(data_subset, 'makesense_labels')
        if os.path.isdir(makesense_imgs_dir) and os.path.isdir(makesense_labels_dir):
            makesense_list_path = os.path.join(output_data_dir, 'makesense_tmp.txt')
            convert_makesense_data_to_yolo(
                makesense_imgs_dir,
                makesense_labels_dir,
                classes,
                makesense_list_path)

            with open(output_data_path, 'a') as f_out:
                with open(makesense_list_path, 'r') as f_in:
                    for l_in in f_in:
                        f_out.write(l_in)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, 
        help='The directory containing data.')
    args = parser.parse_args()
    classes_filename = 'classes.names'

    data_subsets = {
        os.path.join(args.data_dir, 'training'): 'train.txt', 
        os.path.join(args.data_dir, 'validation'): 'val.txt',
        os.path.join(args.data_dir, 'testing'): 'test.txt'}
    class_name_path = os.path.join(args.data_dir, classes_filename)
    classes = read_class_names(class_name_path)
    combine_dataset_paths(data_subsets, args.data_dir, classes)
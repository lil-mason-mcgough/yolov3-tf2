import os
import glob

from PIL import Image

OUTPUT_EXT = '.jpg'


def make_dir_exist(dirname):
    try:
        os.makedirs(dirname)
    except (OSError, FileExistsError):
        pass

def convert_img(img_path, dst_path, quality=75):
    img = Image.open(img_path, mode='r')
    img.save(dst_path, quality=quality)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch conversion of files.')
    parser.add_argument('file_pattern', type=str,
        help='glob pattern for finding images to convert.')
    parser.add_argument('dst_root', type=str,
        help='Destination folder to save converted files to.')
    parser.add_argument('-q', '--quality', type=int, default=75,
        help='Quality of output jpeg compression.')
    parser.add_argument('-r', '--recursive', action='store_true',
        help='If true, searches directories recursively.')
    parser.add_argument('-t', '--preserve-tree', action='store_true',
        help='If true, creates folders to match directory structure of input '\
            'files, starting from first appearance of \'**\'. Otherwise, '\
            'writes all files to dst-root.')
    args = parser.parse_args()

    img_paths = glob.glob(args.file_pattern, recursive=args.recursive)
    input_root_dir = os.path.abspath(args.file_pattern.split('**')[0])
    for img_path in img_paths:
        img_path_tree = os.path.dirname(os.path.relpath(img_path, input_root_dir))
        dst_dir = os.path.join(args.dst_root, img_path_tree)
        make_dir_exist(dst_dir)
        img_name = os.path.splitext(os.path.basename(img_path))[0] + OUTPUT_EXT
        dst_path = os.path.join(dst_dir, img_name)
        print('Writing image: {}'.format(dst_path))
        convert_img(img_path, dst_path, quality=args.quality)
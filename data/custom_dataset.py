"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(gpu_ids='0')
        parser.set_defaults(batchSize=32)
        parser.set_defaults(preprocess_mode='none')
        parser.set_defaults(load_size=160)
        parser.set_defaults(crop_size=160)
        parser.set_defaults(display_winsize=160)
        parser.set_defaults(label_nc=255)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(output_nc=3)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(nThreads=0)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(z_dim=256)
        parser.set_defaults(display_freq=1000)
        parser.set_defaults(print_freq=1000)
        parser.set_defaults(save_latest_freq=10000)

        parser.set_defaults(label_dir='/media/m-ssd3/damola/dataset/training2d/mask_z/')
        parser.set_defaults(image_dir='/media/m-ssd3/damola/dataset/training2d/image_z/')

        parser.add_argument('--label_dir', type=str, required=False,
                            help='/media/m-ssd3/damola/feta_dataset/training/mask2d/')
        parser.add_argument('--image_dir', type=str, required=False,
                            help='/media/m-ssd3/damola/feta_dataset/training/image2d/')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave blank if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths

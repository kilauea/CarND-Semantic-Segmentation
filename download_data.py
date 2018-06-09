"""Download data relevant to train the KittiSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
import subprocess

import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd


from six.moves import urllib
from shutil import copy2

import argparse

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

sys.path.insert(1, 'incl')

# Please set kitti_data_url to the download link for the Kitti DATA.
#
# You can obtain by going to this website:
# http://www.cvlibs.net/download.php?file=data_road.zip
#
# Replace 'http://kitti.is.tue.mpg.de/kitti/?????????.???' by the
# correct URL.


#vgg_data_url = 'ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy'


def download(url, dest_directory):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    
    logging.info("Download URL: {}".format(url))
    logging.info("Download DIR: {}".format(dest_directory))
    
    def _progress(count, block_size, total_size):
        prog = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (filename, prog))
        sys.stdout.flush()
    
    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print('')
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='./data',
                        type=str)
    parser.add_argument('--vgg_data_url',
                        default='https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                        type=str)
    parser.add_argument('--kitti_url',
                        default='http://www.cvlibs.net/download.php?file=data_road.zip',
                        type=str)
    parser.add_argument('--kitti_id',
                        default='http://www.cvlibs.net/download.php?file=data_road.zip',
                        type=str)
    args = parser.parse_args()
    
    vgg_data_url = args.vgg_data_url
    kitti_data_url = args.kitti_url
    kitti_data_id = args.kitti_id
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    vgg_data_zip = os.path.join(data_dir, 'vgg.zip')

    # Download VGG DATA
    if not os.path.exists(vgg_data_zip):
        logging.info("Downloading vgg.")
        download(vgg_data_url, data_dir)
        
        # Extract VGG
        logging.info("Extracting vgg.")
        zipfile.ZipFile(vgg_data_zip, 'r').extractall(data_dir)

    data_road_zip = os.path.join(data_dir, 'data_road.zip')

    # Download KITTI DATA
    if not os.path.exists(data_road_zip):
        logging.info("Downloading Kitti Road Data.")
        # Download files from Google Drive
        gdd.download_file_from_google_drive(file_id=kitti_data_id, dest_path=data_road_zip, unzip=True)
        if False:
            if kitti_data_url == '':
                logging.error("Data URL for Kitti Data not provided.")
                url = "http://www.cvlibs.net/download.php?file=data_road.zip"
                logging.error("Please visit: {}".format(url))
                logging.error("and request Kitti Download link.")
                logging.error("Rerun scipt using"
                              "'python download_data.py' --kitti_url [url]")
                exit(1)
            if not kitti_data_url[-19:] == 'kitti/data_road.zip':
                logging.error("Wrong url.")
                url = "http://www.cvlibs.net/download.php?file=data_road.zip"
                logging.error("Please visit: {}".format(url))
                logging.error("and request Kitti Download link.")
                logging.error("Rerun scipt using"
                              "'python download_data.py' --kitti_url [url]")
                exit(1)

            logging.info("Downloading Kitti Road Data.")
            download(kitti_data_url, data_dir)

            # Extract and prepare KITTI DATA
            logging.info("Extracting kitti_road data.")
            zipfile.ZipFile(data_road_zip, 'r').extractall(data_dir)

    logging.info("All data have been downloaded successful.")


if __name__ == '__main__':
    main()


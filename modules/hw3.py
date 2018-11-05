import cv2
import numpy as np
import scipy
from scipy.misc import imread
import _pickle as cPickle
import random
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import date
import base64
from icrawler import ImageDownloader
from icrawler.builtin import FlickrImageCrawler
from six.moves.urllib.parse import urlparse


class MyImageDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in [
                    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppm', 'pgm'
            ]:
                extension = default_ext
        else:
            extension = default_ext
        # works for python3
        filename = base64.b64encode(url_path.encode()).decode()
        return '{}.{}'.format(filename, extension)


img_path = os.getcwd() + "/var"

# if not os.path.exists(img_path):
#     os.makedirs(img_path)
# flickr_crawler = FlickrImageCrawler(
#     'b040ad4b6a95ddaa8ad86f0762ebc828', storage={'root_dir': img_path})
# flickr_crawler.crawl(
#     max_num=100,
#     tags='orvieto cathedral, italy, architecture',
#     group_id='21041011@N00',
#     min_upload_date=date(2008, 5, 1))


# Feature extractor
def feature_extraction(img_path):
    print(img_path)
    img_building = cv2.imread(
        os.path.join("/Users/wuaiwei/Desktop/EECS442/eta/HW_3/data/",
                     'image_of_cathedral.jpg'))
    img_building = cv2.cvtColor(
        img_building,
        cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    (kp1, des1) = sift.detectAndCompute(img_building, None)
    print("des1", des1[:10])

    img_db_dict = dict()
    for i in range(1, 10):
        pic_name = "/00000" + str(i) + ".jpg"
        test_img = cv2.imread(img_path + pic_name)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        (kp2, des2) = sift.detectAndCompute(test_img, None)

        test_img_keypoints = cv2.drawKeypoints(
            test_img,
            kp2,
            test_img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print("disc", test_img_keypoints[:10])
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(
            matches, key=lambda x: x.distance
        )  # Sort matches by distance.  Best come first.
        print("matches", matches[:10])
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img_matches = cv2.drawMatchesKnn(
            img_building, kp1, test_img, kp2, good, flags=2)
        plt.figure(figsize=(16, 16))
        plt.title("test")
        plt.imshow(img_matches)
        plt.show()


def run(img_path):
    feature_extraction(img_path)


run(img_path)
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
from icrawler.builtin import FlickrImageCrawler

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

    orb = cv2.ORB_create(
    )  # OpenCV 3 backward incompatibility: Do not create a detector with `cv2.ORB()`.
    key_points, description = orb.detectAndCompute(img_building, None)
    img_building_keypoints = cv2.drawKeypoints(
        img_building,
        key_points,
        img_building,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw circles.
    plt.figure(figsize=(16, 16))
    plt.title('ORB Interest Points')
    img_db_dict = dict()
    for i in range(1, 10):
        pic_name = "/00000" + str(i) + ".jpg"
        test_img = cv2.imread(img_path + pic_name)
        print("path", img_path + pic_name)
        test_key_points, test_description = orb.detectAndCompute(
            test_img, None)

        test_img_keypoints = cv2.drawKeypoints(
            test_img,
            test_key_points,
            test_img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print("disc", test_img_keypoints[:10])
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(description, test_description)
        matches = sorted(
            matches, key=lambda x: x.distance
        )  # Sort matches by distance.  Best come first.
        # print("matches", matches)
        print("matches", matches[:10])
        # dist = scipy.spatial.distance.cdist(test_description, description,
        #                                     'cosine').reshape(-1)
        # img_db_dict[i] = test_description
        # print("i hope it works", dist)
        img_matches = cv2.drawMatches(
            img_building,
            key_points,
            test_img,
            test_key_points,
            matches[:10],
            test_img,
            flags=2)  # Show top 10 matches
        plt.figure(figsize=(16, 16))
        plt.title("test")
        plt.imshow(img_matches)
        plt.show()


def run(img_path):
    feature_extraction(img_path)


run(img_path)
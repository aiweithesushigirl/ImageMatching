import cv2
import numpy as np
import scipy
from scipy.misc import imread
import random
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import date
import base64
import json
from icrawler import ImageDownloader
from icrawler.builtin import FlickrImageCrawler
from six.moves.urllib.parse import urlparse
import shutil, os


# Overwrite the downloader to obtain meta data of the photo info
class MyImageDownloader(ImageDownloader):
    def process_meta(self, task):
        info_path = os.getcwd() + "/var/info.json"
        a = []
        entry = {
            'number': 1,
            'url': task['file_url'],
            'id': task['meta']['id'],
            'title': task['meta']['title']
        }
        if not os.path.isfile(info_path):
            a.append(entry)
            with open(info_path, mode='w') as f:
                f.write(json.dumps(a, indent=2))
        else:
            with open(info_path) as feedsjson:
                feeds = json.load(feedsjson)
                entry = {
                    'number': len(feeds) + 1,
                    'url': task['file_url'],
                    'id': task['meta']['id'],
                    'title': task['meta']['title']
                }
            feeds.append(entry)
            with open(info_path, mode='w') as f:
                f.write(json.dumps(feeds, indent=2))


img_path = os.getcwd() + "/var"

# Crawl Flickr to obtain max 500 photos of cathedrals in Italy
title_dict = dict()
info_path = os.getcwd() + "/var/info.json"


def get_images(img_path):
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    flickr_crawler = FlickrImageCrawler(
        'b040ad4b6a95ddaa8ad86f0762ebc828',
        downloader_cls=MyImageDownloader,
        downloader_threads=5,
        storage={'root_dir': img_path})
    flickr_crawler.crawl(
        max_num=500,
        tags='florence',
        extras='description',
        group_id='99108923@N00',
        min_upload_date=date(2005, 5, 1))


# group_id='50035595%N00',


# Feature extractor
def feature_extraction(img_path):
    print(img_path)
    img_building = cv2.imread(
        os.path.join("/Users/wuaiwei/Desktop/EECS442/eta/HW_3/data/",
                     'image_of_cathedral.jpg'))
    img_building_gray = cv2.cvtColor(
        img_building,
        cv2.COLOR_BGR2GRAY)  # Convert from cv's BRG default color order to RGB

    orb = cv2.ORB_create()
    '''
    # Plot the feature points
    
    img_building_keypoints = cv2.drawKeypoints(
        img_building,
        key_points,
        img_building,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw circles.
    plt.figure(figsize=(16, 16))
    plt.title('ORB Interest Points')
    plt.imshow(img_building_keypoints)
    plt.show()
    '''

    dist_dict = []

    for i in range(1, 413):
        pic_name = "/" + "0" * (6 - len(str(i))) + str(i) + ".jpg"
        test_img = cv2.imread(img_path + pic_name)
        test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_key_points, test_description = orb.detectAndCompute(
            test_img_gray, None)
        img_building_resize = cv2.resize(
            img_building_gray, (test_img.shape[1], test_img.shape[0]))
        key_points, description = orb.detectAndCompute(img_building_resize,
                                                       None)

        # Below are the code that I use bf.match to match key points
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(description, test_description)
        matches = sorted(
            matches, key=lambda x: x.distance
        )  # Sort matches by distance. smallest come first.

        # draw_params = dict(
        #     singlePointColor=None, matchColor=(255, 0, 0), flags=2)
        # img_matches = cv2.drawMatches(
        #     img_building_resize, key_points, test_img_gray, test_key_points,
        #     matches[:10], None, **draw_params)  # Show top 10 matches
        # plt.figure(figsize=(16, 16))
        # plt.title("test")
        # plt.imshow(img_matches)
        # plt.show()

        src_pts = np.float32([key_points[m.queryIdx].pt
                              for m in matches])[:15].reshape(-1, 1, 2)
        dst_pts = np.float32([test_key_points[m.trainIdx].pt
                              for m in matches])[:15].reshape(-1, 1, 2)
        # compute Homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        print("homo", i)
        size = (img_building_resize.shape[1], img_building_resize.shape[0])
        # project the given image into the shape of the test_img
        dst = cv2.warpPerspective(test_img, M, size)

        print(dst.shape)
        # plt.imshow(dst)
        # plt.show()
        # calculate euclidean distance
        dist = np.linalg.norm(img_building_resize -
                              cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY))
        print("dist", dist)
        dist_dict.append([i, dist])

    dist_dict = sorted(dist_dict, key=lambda x: x[1])
    return (dist_dict[:15])


def find_match(dist_dict):
    # print(dist_dict)
    img_path = os.getcwd() + "/var"
    info_path = os.getcwd() + "/var/info.json"
    with open(info_path) as feedsjson:
        feeds = json.load(feedsjson)
        # print(feeds)
        for i in range(len(dist_dict)):
            for j in range(len(feeds)):
                if feeds[j]['number'] == dist_dict[i][0]:
                    pic_name = "/" + "0" * (6 - len(str(
                        dist_dict[i][0]))) + str(dist_dict[i][0]) + ".jpg"
                    f = img_path + pic_name
                    dest_folder = img_path + "/result/"
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    shutil.copy(f, dest_folder)
                    print(feeds[j]['title'])


def run(img_path):
    # get_images(img_path)
    dist_dict = feature_extraction(img_path)
    find_match(dist_dict)


run(img_path)

# test_img_keypoints = cv2.drawKeypoints(
#     test_img,
#     test_key_points,
#     test_img,
#     color=(255, 0, 0),
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# '''
# # use knn
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# matches = bf.knnMatch(description, test_description, k=2)

# # Apply ratio test
# good = []
# for m, n in matches:
#     if m.distance < 0.89 * n.distance:
#         good.append([m, n])

# good = sorted(
#     good, key=lambda x: x[0].distance**2 + x[1].distance**2
# )  # Sort matches by distance.  Best come first.
# dist = 0
# for j in range(10):
#     dist += good[j][0].distance**2 + good[j][1].distance**2
# dist_dict.append([i, dist])
# '''

# draw_params = dict(
#     singlePointColor=None, matchColor=(255, 0, 0), flags=2)
# img_matches = cv2.drawMatches(img_building, key_points, test_img,
#                               test_key_points, matches[:10], None,
#                               **draw_params)  # Show top 10 matches
# # plt.figure(figsize=(16, 16))
# plt.title("test")
# plt.imshow(img_matches)
# plt.show()

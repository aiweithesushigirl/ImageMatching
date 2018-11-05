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
import json
from icrawler import ImageDownloader
from icrawler.builtin import FlickrImageCrawler
from six.moves.urllib.parse import urlparse


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
'''
# Crawl Flickr to obtain max 500 photos of cathedrals in Italy
title_dict = dict()
info_path = os.getcwd() + "/var/info.json"

if not os.path.exists(img_path):
    os.makedirs(img_path)
flickr_crawler = FlickrImageCrawler(
    'b040ad4b6a95ddaa8ad86f0762ebc828',
    downloader_cls=MyImageDownloader,
    downloader_threads=4,
    storage={'root_dir': img_path})
flickr_crawler.crawl(
    max_num=500,
    tags='florence',
    extras='description',
    group_id='99108923@N00',
    min_upload_date=date(2005, 5, 1))
'''


# Feature extractor
def feature_extraction(img_path):
    print(img_path)
    img_building = cv2.imread(
        os.path.join("/Users/wuaiwei/Desktop/EECS442/eta/HW_3/data/",
                     'image_of_cathedral.jpg'))
    img_building = cv2.cvtColor(
        img_building,
        cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

    orb = cv2.ORB_create()
    key_points, description = orb.detectAndCompute(img_building, None)
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

    for i in range(1, 426):
        pic_name = "/" + "0" * (6 - len(str(i))) + str(i) + ".jpg"
        test_img = cv2.imread(img_path + pic_name)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_key_points, test_description = orb.detectAndCompute(
            test_img, None)

        test_img_keypoints = cv2.drawKeypoints(
            test_img,
            test_key_points,
            test_img,
            color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(description, test_description, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.89 * n.distance:
                good.append([m, n])

        good = sorted(
            good, key=lambda x: x[0].distance**2 + x[1].distance**2
        )  # Sort matches by distance.  Best come first.
        dist = 0
        for j in range(10):
            dist += good[j][0].distance**2 + good[j][1].distance**2
        dist_dict.append([i, dist])
        '''

        # Plot figure

        # Below are the code that I use bf.match to match key points

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(description, test_description)
        '''
        draw_params = dict(
            singlePointColor=None, matchColor=(255, 0, 0), flags=2)
        img_matches = cv2.drawMatches(img_building, key_points, test_img,
                                      test_key_points, good[:10], None,
                                      **draw_params)  # Show top 10 matches
        plt.figure(figsize=(16, 16))
        plt.title("test")
        plt.imshow(img_matches)
        plt.show()
        '''

        # calculate euclidean distance
        dist = 0
        for j in range(10):
            dist += matches[j].distance
        dist_dict.append([i, dist])

    dist_dict = sorted(dist_dict, key=lambda x: x[1])
    return (dist_dict[:10])


def find_match(dist_dict):
    print(dist_dict)
    info_path = os.getcwd() + "/var/info.json"
    with open(info_path) as feedsjson:
        feeds = json.load(feedsjson)
        # print(feeds)
        for i in range(len(dist_dict)):
            for j in range(len(feeds)):

                if feeds[j]['number'] == dist_dict[i][0]:
                    print(feeds[j]['title'])


def run(img_path):

    dist_dict = feature_extraction(img_path)
    find_match(dist_dict)


run(img_path)
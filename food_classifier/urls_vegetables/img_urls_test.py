#template:
#https://github.com/hardikvasa/google-images-download/blob/master/google_images_download/google_images_download.py
import sys
import urllib.request
from urllib.request import Request, urlopen
from urllib.request import URLError, HTTPError
from urllib.parse import quote
import http.client
from http.client import IncompleteRead, BadStatusLine
http.client._MAXHEADERS = 200

import time  # Importing the time library to check the time of code execution
import os
import argparse
import ssl
import datetime
import json
import re
import codecs
import socket
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vegetable', help='vegetable to gather from google images', required=True, type=str)
parser.add_argument('-s', '--search-term', help='google images search term', required=True, type=str)
args = parser.parse_args()
arguments = vars(args)

# Download Page for more than 100 images
def download_extended_page(url, chromedriver_path):
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument("--headless")

    if (chromedriver_path.exists() and not chromedriver_path.is_dir()): print ('yay')

    try:
        browser = webdriver.Chrome(chromedriver_path, chrome_options=options)
    except Exception as e:
        print("Looks like we cannot locate the path the 'chromedriver' or google chrome browser is not "
              "installed on your machine (exception: %s)" % e)
        sys.exit()
    browser.set_window_size(1024, 768)

    # Open the link
    browser.get(url)
    time.sleep(1)
    print("Getting you a lot of images. This may take a few moments...")

    element = browser.find_element_by_tag_name("body")
    # Scroll down
    for i in range(30):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id("smb").click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection

    print("Reached end of Page.")
    time.sleep(0.5)

    source = browser.page_source #page source
    #close the browser
    browser.close()

    return source



def get_all_tabs(page):
    tabs = {}
    while True:
        item,item_name,end_content = get_next_tab(page)
        if item == "no_tabs":
            break
        else:
            if len(item_name) > 100 or item_name == "background-color":
                break
            else:
                tabs[item_name] = item  # Append all the links in the list named 'Links'
                time.sleep(0.1)  # Timer could be used to slow down the request for image downloads
                page = page[end_content:]
    return tabs


def main():

	path = pathlib.Path.cwd()
	veg = arguments['vegetable']
	# dest = path/veg
	# print(f'creating dir: {dest}')
	# dest.mkdir(parents=False, exist_ok=True)

	url = 'https://www.google.com/search?q=' + quote(
		arguments['search_term'].encode('utf-8')) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'

	source = download_extended_page(url, path/'chromedriver.exe')
	tabs = get_all_tabs(source)


	filename = f'urls_{veg}.txt'
	f = open(filename, "w")
	for item in tabs:
		f.write(item+'\n')
	# for i in range(10):
 #    	f.write("This is line %d\r\n" % (i+1))

	f.close()

if __name__ == "__main__":
	main()
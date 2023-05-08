# code for images scraping and for store them locally

from selenium import webdriver
from bs4 import BeautifulSoup as bs
import os
import time
import requests  # to get image from the web
import shutil  # to save them locally

# location of the webdriver, needed for Selenium
wd = webdriver.Chrome("C:\\Users\\Matteo\\Desktop\\chromedriver_win32\\chromedriver.exe")

# images links scraping
url = "https://www.gallerie-estensi.beniculturali.it/collezioni-digitali?oggetto=Moneta&perpage=30&page="

coins = []

for num in range(1, 7):
    # print(url + str(num))

    # obtaining the source pages with Selenium
    wd.get(url + str(num))

    time.sleep(1)  # for debugging reasons

    htmlSource = wd.page_source

    # actual scraping
    soup = bs(htmlSource, "lxml")

    images = soup.find_all(lambda tag: tag.name == "div" and tag.get("class") == ["image"])

    # images[0]
    # type(images[0])

    for coin in images:
        coins.append(coin.img["src"].replace("amp;", ""))

# downloading and storing the images locally
os.mkdir("C:\\Users\\Matteo\\Desktop\\Coins_Dataset")

for image in coins:
    filename = image.split("/")[-1].split("?")[0]  # a trick needed for a problem with the URL

    # opening the url image, this will return the stream content
    r = requests.get(image, stream=True)

    # checking if the image was retrieved successfully
    if r.status_code == 200:
        # setting decode_content value to True, otherwise the downloaded image file's size will be zero
        r.raw.decode_content = True

        # opening a local file with wb(write binary) permission
        with open(str("C:\\Users\\Matteo\\Desktop\\Coins_Dataset\\" + filename), 'wb') as f:
            shutil.copyfileobj(r.raw, f)

            print('Image successfully downloaded: ', filename)
    else:
        print('Image couldn\'t be retrieved.')

print(f"you've successfully downloaded {len(coins)} images.")

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse

def download_images(query, num_images=10, save_folder='images'):
    os.makedirs(save_folder, exist_ok=True)

    search_url = f"https://www.pexels.com/search/{query}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        print(f"Error obtaining the page: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    images = soup.find_all('img')

    image_urls = []
    for img in images:
        img_url = img.get('src')
        if img_url:
            img_url = urljoin(search_url, img_url)
            image_urls.append(img_url)

    count = 0
    for img_url in image_urls[:num_images]:
        try:
            img_data = requests.get(img_url, headers=headers).content

            img_name = os.path.join(save_folder, f'{query}_{count+1}.jpg')

            with open(img_name, 'wb') as f:
                f.write(img_data)
                print(f"Image downloaded: {img_name}")
            count += 1
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True, help="output folder")
    ap.add_argument("--num", type=int, required=True, help="output folder")

    args = vars(ap.parse_args())

    OUTPUT_FOLDER = args["output"]
    NUM_IMAGES = args["num"]

    queries = ['cat', 'dog', 'bird', 'horse']
    for query in queries:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        download_images(query,num_images=NUM_IMAGES,save_folder=os.path.join(OUTPUT_FOLDER,query))

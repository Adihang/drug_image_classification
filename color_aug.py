import csv
import os
import json
from PIL import Image
from tqdm import tqdm

image_files_path = './image'
label_files_path = './1.Training\label'
txt_file_path = "./txtfile.txt"

# JSON 파일 읽기
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data
        
# 배경색 읽기
def back_color(data):
    return data["images"][0]["back_color"]

# 조명색 읽기
def light_color(data):
    return data["images"][0]["light_color"]

# 앞, 뒷면 읽기
def drug_dir(data):
    return data["images"][0]["drug_dir"]

# 약 색깔 읽기
def drug_color(data):
    return data["images"][0]["color_class1"]

# 이미지 크롭
def center_crop_images(path, target_size):
    if os.path.isfile(path):
        # 이미지를 OpenCV를 사용하여 읽어옴
        image = Image.open(path)
        if image is not None:
            # 이미지의 높이와 너비를 가져옴
            width, height = image.size

            # 가운데를 기준으로 자르기
            top = (height - target_size[0]) // 2
            bottom = top + target_size[0]
            left = (width - target_size[1]) // 2
            right = left + target_size[1]

            # 이미지를 가운데 크롭하고 저장
            cropped_image = image.crop((left, top, right, bottom))
            
            # 크롭한 이미지를 원래 파일 경로에 저장
            cropped_image.save(path)
            print(f"{path}를 크롭했습니다.")
        else:
            print(f"Failed to read image: {path}")
    else:
        print(f'{path}가 없습니다.')

csv_file_path = "./csvfile.csv"

# 현재 디렉토리와 그 하위 디렉토리에서 JSON 파일 찾기
csvfile = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csvfile)
for root, dirs, files in tqdm(list(os.walk(label_files_path))):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                data = process_json_file(json_file_path)
                back_c = back_color(data)
                light_c = light_color(data)
                drug_d = drug_dir(data)
                drug_c = drug_color(data)
                csv_writer.writerow([file[:8], drug_c, drug_d, back_c, light_c])
csvfile.close()
print('작업이 완료되었습니다.')
input()
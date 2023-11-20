import os
from tqdm import tqdm
import json
import shutil

# 탐색할 디렉토리 경로 설정
label_dir = './dataset/label'
image_dir = './dataset/secondclass/연질캡슐제_장방형'


def drug_shape(data):
    return data["images"][0]["drug_shape"]

def dl_custom_shape(data):
    return data["images"][0]["dl_custom_shape"]

# JSON 파일 읽기
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

# 디렉토리 내부의 모든 하위 디렉토리 순회
for root, dirs, files in tqdm(list(os.walk(label_dir))):
    if len(files) > 0:
        if root == label_dir:
            continue
        # 폴더 내에 파일이 하나 이상 있는 경우
        file = files[0]  # 첫 번째 파일
        filename = file[:8]
        file_path = os.path.join(root, file)  # 파일의 전체 경로
        json_file_path = os.path.join(root, file)
        data = process_json_file(json_file_path)
        shape = drug_shape(data)
        custom_shape = dl_custom_shape(data)
        if shape == None:
            shape = 'none'
        if shape == '타원형':
            source_folder = os.path.join(image_dir, filename)
            save_dir = os.path.join(image_dir, shape)
            print(f'{source_folder} to {save_dir}')
            try:
                shutil.move(source_folder, save_dir)
            except:
                pass
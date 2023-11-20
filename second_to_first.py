import os
import shutil
from tqdm import tqdm
# 탐색할 디렉토리 경로 설정
root_dir = './dataset/secondclass/연질캡슐제'  # 여기에 탐색할 디렉토리 경로를 지정하세요

# 하위 폴더 및 파일 순회
for root, dirs, files in tqdm(list(os.walk(root_dir))):
    for file in files:
        if file.endswith(".png"):
            # .png 파일을 찾았을 때
            file_path = os.path.join(root, file)  # 파일의 전체 경로
            parent_folder = os.path.dirname(root)  # 1단계 상위 폴더 경로
            new_path = os.path.join(parent_folder, file)  # 새로운 경로

            # 파일 이동
            shutil.move(file_path, new_path)
            print(f'Moved: {file_path} to {new_path}')
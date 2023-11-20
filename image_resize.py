import os
from tqdm import tqdm
from PIL import Image

training_files_path = './train'
# 현재 디렉토리와 그 하위 디렉토리에서 JSON 파일 찾기
for root, dirs, files in tqdm(list(os.walk(training_files_path))):
    for file in files:
        #if file.endswith(".json") or file.endswith(".png"):
        if file.endswith(".png"):
            file_path = os.path.join(root, file)
            # 이미지 열기
            image = Image.open(file_path)
            # 리사이즈
            new_size = (244, 244)
            resized_image = image.resize(new_size)
            resized_image.save(file_path)
            print(f"{file_path}크기변경")
print('작업이 완료되었습니다.')
input()
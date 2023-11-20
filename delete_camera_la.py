import os
from tqdm import tqdm

training_files_path = './dataset'
print('위도 삭제중.....')
# 현재 디렉토리와 그 하위 디렉토리에서 JSON 파일 찾기
for root, dirs, files in tqdm(list(os.walk(training_files_path))):
    for file in files:
        if file.endswith(".json") or file.endswith(".png"):
            file_path = os.path.join(root, file)
            file = list(file.split('_'))
            if (file[5] != '75') and (file[5] !='90'):
                os.remove(file_path)
                print(f"{file_path}삭제")
print('작업이 완료되었습니다.')
input()
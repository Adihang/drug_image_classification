import os
from tqdm import tqdm
import json

# 탐색할 디렉토리 경로 설정
label_dir = './1.Training/label'


def dl_name(data):
    return data["images"][0]["dl_name"]

# JSON 파일 읽기
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

# 디렉토리 내부의 모든 하위 디렉토리 순회
text_line=''
textfile = open("lisettxt.txt", "w", encoding='utf-8')
for root, dirs, files in tqdm(list(os.walk(label_dir))):
    if len(files) > 0:
        if root == label_dir:
            continue
        file = files[0]  # 첫 번째 파일
        if file.endswith(".json"):
            filename = file[:8]
            file_path = os.path.join(root, file)  # 파일의 전체 경로
            json_file_path = os.path.join(root, file)
            data = process_json_file(json_file_path)
            name = dl_name(data)
            name = ''.join(filter(str.isalnum, name))
            text = f"{filename}\n{name}\n"
            text_line += text
            
textfile.write(text_line)
textfile.close()
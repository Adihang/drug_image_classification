from rembg import remove
from PIL import Image
import os
from tqdm import tqdm

image_files_path = './history'

def add_transparent_background(img):
    # 원본 이미지의 가로와 세로 중 더 큰 값을 선택하여 정사각형의 크기 결정
    size = max(img.size)

    # 정사각형의 배경을 갖는 새로운 이미지 생성
    square_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # 원본 이미지를 중앙에 배치
    square_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))

    return square_img


def remove_Background(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_image = add_transparent_background(output_image)
    output_image.save(image_path)
    print(f'{image_path} 배경제거 완료')
    


    
for root, dirs, files in tqdm(list(os.walk(image_files_path))):
    for file in files:
        if file.endswith(".png"):
            image_file_path = os.path.join(root, file)
            remove_Background(image_file_path)
            
            
print('작업 완료')
input()
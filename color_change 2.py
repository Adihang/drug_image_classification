import os
import numpy as np
import json
import cv2


# json에서 배경색 읽어오기
def back_color(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    b_color = data['images'][0]['back_color']
    return b_color


def gray_change_color(image_path):
    change_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    b, g, r = np.clip(cv2.split(change_img), 50, 200)
    # 연회색 배경(_0_2_0_0) -> 검은색 배경(_0_0_0_0) 변경 후 저장
    img1 = cv2.merge((b-30, g-50, r-40))
    split_path1 = image_path.split('_')
    split_path1[2] = '0'
    img1_path = "_".join(split_path1)
    cv2.imwrite(img1_path, img1)

    # 연회색 배경(_0_2_0_0) -> 파란색 배경(_0_1_0_0) 변경 후 저장
    img2 = cv2.merge((b - 10, g - 15, r - 40))
    split_path2 = image_path.split('_')
    split_path2[2] = '1'
    img2_path = "_".join(split_path2)
    cv2.imwrite(img2_path, img2)


# 검은색 배경이미지 색상 변경 (_0_0_x_x_xxx_200)
def black_change_color(image_path):
    change_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    b, g, r = np.clip(cv2.split(change_img), 35, 200)
    # 검은색 배경(_0_0_0_0) -> 연회색 배경(_0_2_0_0) 변경 후 저장
    img1 = cv2.merge((b + 30, g + 50, r + 40))
    split_path1 = image_path.split('_')
    split_path1[2] = '2'
    img1_path = "_".join(split_path1)
    cv2.imwrite(img1_path, img1)

    # 검은색 배경(_0_0_0_0) -> 파란색 배경(_0_1_0_0) 변경 후 저장
    img2 = cv2.merge((b + 40, g + 45, r))
    split_path2 = image_path.split('_')
    split_path2[2] = '1'
    img2_path = "_".join(split_path2)
    cv2.imwrite(img2_path, img2)


# 파란색 배경이미지 색상 변경 (_0_1_x_x_xxx_200)
def blue_change_color(image_path):
    change_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    b, g, r = np.clip(cv2.split(change_img), 40, 200)
    # 파랑색 배경(_0_1_0_0) -> 연회색 배경(_0_2_0_0) 변경 후 저장
    img1 = cv2.merge((b - 40, g - 40, r + 10))
    split_path1 = image_path.split('_')
    split_path1[2] = '2'
    img1_path = "_".join(split_path1)
    cv2.imwrite(img1_path, img1)

    # 파랑색 배경(_0_1_0_0) -> 검은색 배경(_0_0_0_0) 변경 후 저장
    img2 = cv2.merge((b - 10, g + 22, r + 50))
    split_path2 = image_path.split('_')
    split_path2[2] = '0'
    img2_path = "_".join(split_path2)
    cv2.imwrite(img2_path, img2)


# 이미지 파일을 찾아 색상 변경 작업하기
def main():
    image_dir = './labels/'
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            label_path = os.path.join(root, file)
            image_path = label_path.replace('labels','images').replace('_json','').replace('.json','.png')
            try:
                if back_color(label_path) == '연회색 배경':
                    gray_change_color(image_path)
                elif back_color(label_path) == '검은색 배경':
                    black_change_color(image_path)
                else:
                    blue_change_color(image_path)
            except:
                print(f'error file : {image_path}')


if __name__ == "__main__":
    main()

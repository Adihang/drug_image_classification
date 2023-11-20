import os
import glob
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MySportsDataset(Dataset):
    def __init__(self, csv_file_path, image_dir, transforms=None, mode=None):
        # CSV 파일 읽기
        self.csv_file = pd.read_csv(csv_file_path)
        print("원본 CSV 레코드 수: ", len(self.csv_file))
        
        # 이미지 디렉토리 경로 설정
        self.root_dir = image_dir
        
        # 데이터 변환 설정
        self.transforms = transforms
        
        # 클래스 레이블 딕셔너리 초기화
        self.label_dict = {}
        
        # 데이터 모드 설정 (train 또는 valid)
        self.mode = mode
        
        # 이미지 폴더 이름을 기반으로 클래스 레이블 딕셔너리 생성
        sub_folder_names = os.listdir("./1.Training/image")
        for i, sub_folder_name in enumerate(sub_folder_names):
            self.label_dict[sub_folder_name] = i
        
        # 지정한 데이터 모드에 해당하는 레코드만 선택
        self.csv_file = self.csv_file[self.csv_file['data set'] == self.mode]

    def __getitem__(self, item):
        # CSV에서 해당 레코드의 데이터 모드를 가져옴
        csv_mode = self.csv_file.iloc[item, 3]

        # 지정한 데이터 모드와 CSV 데이터의 데이터 모드가 일치하는 경우 이미지 처리
        if self.mode == csv_mode:
            # 이미지 파일 경로 생성
            image_path = os.path.join(self.root_dir, self.csv_file.iloc[item, 1])

            # 이미지 파일 확장자가 이미지인지 확인
            valid_extensions = ('.jpg', 'jpeg', '.png', '.bmp', '.gif', '.tiff')
            if not image_path.lower().endswith(valid_extensions):
                # 이미지가 아닌 파일인 경우, 다음 레코드 처리
                return self.__getitem__(item + 1)

            # 이미지 열기 및 RGB 모드로 변환
            image = Image.open(image_path).convert('RGB')
            
            # 클래스 레이블 가져오기
            label = self.csv_file.iloc[item, 2]
            label = self.label_dict[label]
            
            # 데이터 변환 적용
            if self.transforms is not None:
                image = self.transforms(image)

            return image, label

    def __len__(self):
        # 현재 데이터 모드에 해당하는 레코드 수 반환
        return len(self.csv_file)

if __name__ == "__main__":
    # 데이터셋 클래스의 인스턴스 생성
    test = MySportsDataset(csv_file_path="./ex02_data/sports.csv", image_dir="./ex02_data", mode="train")
    
    # 데이터셋 반복하며 출력
    for i in test:
        print(i)

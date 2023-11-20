
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import gradio as gr
from PIL import Image
from rembg import remove
import os
import numpy as np


# 모델 불러오기
def resnet18_model(data, device):
    train_data = f'./dataset/secondclass/{data}'
    pt_PATH = f'./pt_files/{data}_model.pt'
    if data == 'shape':
        train_data = './dataset/shapeclass/'
    elif data == '연질캡슐제_장방형' or data == '연질캡슐제_타원형':
        train_data = f'./dataset/thirdclass/{data}'
    train_dataset = torchvision.datasets.ImageFolder(root = train_data)
    labels = train_dataset.classes
    num_classes = len(labels)

    model = models.resnet18(weights=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    model.load_state_dict(torch.load(pt_PATH))
    model = model.to(device)
    
    return model, labels

def get_subfolder_names(directory):
    # 디렉토리 내의 모든 하위 폴더 이름을 가져오기
    subfolders = [f.name for f in os.scandir(directory) if f.is_dir()]
    return subfolders

# 데이터 전처리 함수
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 원본 이미지 배경을 늘려 정사각형으로 변환
def add_transparent_background(img):
    # 원본 이미지의 가로와 세로 중 더 큰 값을 선택하여 정사각형의 크기 결정
    size = max(img.size)
    # 정사각형의 배경을 갖는 새로운 이미지 생성
    square_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # 원본 이미지를 중앙에 배치
    square_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return square_img

image_path= './history/input.png'
def remove_Background():
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_image = add_transparent_background(output_image)
    output_image.save(image_path)
    print(f'배경제거 완료')

# Gradio 예측 함수
def predict(img):
    img_pil = Image.fromarray(img)
    img_pil.save(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    remove_Background()
    img_pil = Image.open(image_path)
    img = np.array(img_pil)
    
    # 이미지가 RGBA(4 채널) 형식이라면 RGB(3 채널)로 변환
    if img.shape[2] == 4:
        img = img[:, :, :3]
    
    img = Image.fromarray(img)
    transform_img = transform(img)
    img = transform_img.unsqueeze(0).to(device)  # GPU로 전송
    
    
    
    predicted_label = 'shape'
    class_list = get_subfolder_names('./dataset/secondclass/') + get_subfolder_names('./dataset/thirdclass/')
    class_list.append(predicted_label)
    steps = []
    while predicted_label in class_list:
        model, labels = resnet18_model(predicted_label, device)
        with torch.no_grad():
            model.eval()
            output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_label = labels[predicted.item()]  # 예측된 클래스 레이블
        steps.append(predicted_label)
        
    probabilities = torch.softmax(output, dim=1)[0]  # 클래스별 확률
    top5_prob, top5_catid = torch.topk(probabilities, 5)  # 상위 5개 클래스 확률 및 인덱스
    top5_labels = [labels[i.item()] for i in top5_catid]
    result = "\n".join([f"예측 {i + 1}: {label}, 확률: {prob.item() * 100:.2f}%" for i, (label, prob) in enumerate(zip(top5_labels, top5_prob))])
    return f'{str(steps)}\n'+result

# Gradio UI 생성
iface = gr.Interface(fn=predict, inputs="image", outputs="text", title="Drug Classifier")
iface.launch()
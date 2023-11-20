import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def model_train(data):
    train_data = f'./dataset/thirdclass/{data}'
    pt_PATH = f'./pt_files/{data}_model60.pt'
    if data == 'shape':
        train_data = './dataset/shapeclass/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #모델 설정
    model = models.resnet18(weights=True)

    for param in model.parameters():
        param.requires_grad = False

    #load and preprocess the data
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root = train_data, transform = transform)
    labels = dataset.classes

    model.fc = nn.Linear(in_features=512, out_features=len(labels))

    #test_dataset = torchvision.datasets.ImageFolder(root = val_data, transform=transform)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)



    #show dataset
    print(f"len(num_classes): {len(labels)}")
    print(f"len(train_dataset): {len(train_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    model = model.to(device)


    # 학습 루프
    print(f'{data} train start......')
    num_epochs = 60
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        # 에폭마다 손실 출력
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}")

    # 평가 루프
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"test accuracy: {accuracy:.2f}%")

    #모델 저장
    torch.save(model.state_dict(), pt_PATH)
    print(f"{pt_PATH} 모델 저장")


model_train('연질캡슐제_장방형')
model_train('연질캡슐제_타원형')
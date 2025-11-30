import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# 加载模型架构
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('recapture_detection_model.pth', map_location='cpu'))
model.eval()

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0) # 增加batch维度
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
    # 假设 0 是 fake, 1 是 real (根据class_names顺序)
    confidence = probabilities[0][preds.item()].item()
    result = "Real (真图)" if preds.item() == 1 else "Fake (屏幕翻拍)"
    
    return result, confidence

# 测试
print(predict_image("test_photo.jpg"))
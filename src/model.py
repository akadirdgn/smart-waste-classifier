import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Son katmanı kendi sınıf sayımıza göre değiştiriyoruz (Örn: Cam, Kağıt, Metal, Plastik, Karton, Çöp = 6 sınıf)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":

    dummy_model = get_model(num_classes=6)
    print("Model başarıyla yüklendi!")

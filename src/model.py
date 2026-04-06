import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Transfer Learning için önceden eğitilmiş bir ResNet18 modeli yükler.
    Son katmanı (classifier) atık sınıf sayımıza göre değiştirir.
    """
    # Önceden eğitilmiş ağırlıkları(weights) ile modeli yüklüyoruz.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Tüm katmanların önce eğitilmesini dondurabilir veya tamamen eğitebiliriz.
    # Örnek olarak burada baştan başa eğitime (fine-tuning) açık bırakıyoruz.
    
    # Son katmanı kendi sınıf sayımıza göre değiştiriyoruz (Örn: Cam, Kağıt, Metal, Plastik, Karton, Çöp = 6 sınıf)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test amaçlı modeli oluşturalım
    dummy_model = get_model(num_classes=6)
    print("Model başarıyla yüklendi!")

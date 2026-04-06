import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, target_size=(224, 224)):
    """
    Veri setini okuyup train ve val (veya test) DataLoader nesnelerini döndürür.
    `data_dir` dizini altında train ve val (veya test) klasörlerinin olması beklenir.
    """
    # ResNet18 gibi pre-trained modeller genelde ImageNet ortalamalarıyla normalize edilmeyi bekler
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {}
    dataloaders = {}
    
    # Train veri seti
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms['train'])
        dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
        
    # Val veri seti
    val_dir = os.path.join(data_dir, 'val')
    if os.path.exists(val_dir):
        image_datasets['val'] = datasets.ImageFolder(val_dir, data_transforms['val'])
        dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)

    # Test veri seti
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        image_datasets['test'] = datasets.ImageFolder(test_dir, data_transforms['val'])
        dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

    return dataloaders, image_datasets

if __name__ == "__main__":
    print("Veriseti yardımcı fonksiyonları yüklendi.")

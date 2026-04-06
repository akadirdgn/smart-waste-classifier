import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import get_model
from dataset import get_dataloaders

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cpu', save_dir='models'):
    """
    Modeli verilen Pytorch DataLoader'ları ile eğitir ve en iyi modeli 'save_dir' içerisine kaydeder.
    """
    start_time = time.time()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Kayıpları ve doğrulukları saklamak için dictionary
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Her epoch için eğitim ve doğrulama aşamaları
        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue

            if phase == 'train':
                model.train()  # Modeli eğitim moduna al
            else:
                model.eval()   # Modeli değerlendirme moduna al

            running_loss = 0.0
            running_corrects = 0

            # Verileri modele besle
            loop = tqdm(dataloaders[phase], desc=f"  {phase.capitalize():5s}", leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Parametre gradyanlarını sıfırla
                optimizer.zero_grad()

                # İleri geçiş (forward)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Geri yayılım (backward) ve optimizasyon sadece train modundaysa
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # İstatistikleri güncelle
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Daha iyi bir model bulduysak kaydet
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print()

    time_elapsed = time.time() - start_time
    print(f'Eğitim tamamlandı: {time_elapsed // 60:.0f}dk {time_elapsed % 60:.0f}sn')
    if 'val' in dataloaders:
        print(f'En İyi Doğrulama Başarısı (Validation Acc): {best_acc:4f}')

    # Eğitilmiş en iyi modeli geri döndür
    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history, save_path='training_history.png'):
    """
    Eğitim ve doğrulama (validation) doğruluk ve kayıp oranlarını ekrana çizer, kaydeder.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy Grafiği
    ax1.plot(history['train_acc'], label='Train Accuracy')
    if history['val_acc']:
        ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss Grafiği
    ax2.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Eğitim grafiği '{save_path}' konumuna kaydedildi.")


if __name__ == '__main__':
    # --- Konfigurasyon ---
    SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.dirname(SRC_DIR)
    DATA_DIR   = os.path.join(ROOT_DIR, "data", "split")
    SAVE_DIR   = os.path.join(ROOT_DIR, "models")
    RESULT_DIR = os.path.join(ROOT_DIR, "results")
    NUM_CLASSES = 6
    BATCH_SIZE  = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("  TrashNet - Transfer Learning Egitimi")
    print("=" * 50)
    print(f"  Cihaz : {DEVICE.upper()}")
    print(f"  Sinif : {NUM_CLASSES}")
    print(f"  Batch : {BATCH_SIZE}")
    print()

    # Veri yukle
    dataloaders, image_datasets = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    class_names = image_datasets['train'].classes
    print(f"  Siniflar: {class_names}")
    for split, ds in image_datasets.items():
        print(f"  {split.capitalize():6s}: {len(ds)} goruntu")
    print()

    # --- FAZ 1: Feature Extraction (sadece FC katmani egitilir) ---
    print("-" * 50)
    print("  FAZ 1: Feature Extraction (5 epoch)")
    print("-" * 50)
    model = get_model(NUM_CLASSES).to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model, history = train_model(
        model, dataloaders, criterion, optimizer,
        num_epochs=5, device=DEVICE, save_dir=SAVE_DIR
    )

    # --- FAZ 2: Fine-Tuning (tum model) ---
    print("-" * 50)
    print("  FAZ 2: Fine-Tuning (15 epoch, dusuk LR)")
    print("-" * 50)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model, history2 = train_model(
        model, dataloaders, criterion, optimizer,
        num_epochs=15, device=DEVICE, save_dir=SAVE_DIR
    )

    # Geçmişleri birlestir
    for key in history:
        history[key].extend(history2[key])

    # Grafikleri kaydet
    plot_training_history(
        history,
        save_path=os.path.join(RESULT_DIR, "training_history.png")
    )
    print()
    print("Egitim tamamlandi!")
    print(f"  Model kaydi : {os.path.join(SAVE_DIR, 'best_model.pth')}")
    print(f"  Grafik      : {os.path.join(RESULT_DIR, 'training_history.png')}")
    print("Sonraki adim: python src/evaluate.py")

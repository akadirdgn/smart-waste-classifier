import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, class_names, device='cpu'):
    """
    Egitilmis modeli bir veri seti uzerinde test eder.
    Karisiklik matrisini (confusion matrix) ve siniflandirma raporunu yazdirir.
    """
    model.eval()

    all_preds = []
    all_labels = []

    print("Model degerlendiriliyor...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Siniflandirma raporu
    print("\nSiniflandirma Raporu (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Karisiklik Matrisi
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gercek Deger (Actual)')
    plt.title('Karisiklik Matrisi (Confusion Matrix)')

    return cm, all_labels, all_preds


if __name__ == '__main__':
    # Bulunduğumuz konumdan bağımsız doğru yol hesapla
    SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.dirname(SRC_DIR)
    sys.path.insert(0, SRC_DIR)
    from model import get_model
    from dataset import get_dataloaders

    DATA_DIR   = os.path.join(ROOT_DIR, "data", "split")
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pth")
    RESULT_DIR = os.path.join(ROOT_DIR, "results")
    NUM_CLASSES = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 50)
    print("  TrashNet - Model Degerlendirme")
    print("=" * 50)
    print(f"  Cihaz     : {DEVICE.upper()}")
    print(f"  Model     : {MODEL_PATH}")
    print()

    # Veri yukle
    dataloaders, image_datasets = get_dataloaders(DATA_DIR, batch_size=32)
    class_names = image_datasets['train'].classes
    print(f"  Siniflar : {class_names}")

    if 'val' not in dataloaders:
        print("HATA: Val split bulunamadi!")
        sys.exit(1)

    # Modeli yukle
    model = get_model(NUM_CLASSES).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyasi bulunamadi: {MODEL_PATH}")
        print("Once 'python src/train.py' calistirin.")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("  Model basariyla yuklendi!")
    print()

    # Degerlendirme - val split
    print("--- Validation Seti ---")
    cm, labels, preds = evaluate_model(model, dataloaders['val'], class_names, DEVICE)

    save_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Karisiklik matrisi kaydedildi: {save_path}")

    # Test split varsa
    if 'test' in dataloaders:
        print("\n--- Test Seti ---")
        cm_test, _, _ = evaluate_model(model, dataloaders['test'], class_names, DEVICE)
        save_test = os.path.join(RESULT_DIR, "confusion_matrix_test.png")
        plt.tight_layout()
        plt.savefig(save_test, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Test karisiklik matrisi kaydedildi: {save_test}")

    print()
    print("Degerlendirme tamamlandi!")


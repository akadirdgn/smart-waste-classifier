import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import argparse

def predict_single_image(image_path, model_path, num_classes, class_names, device='cpu'):
    """
    Belirtilen görüntüyü yükler ve eğitilmiş model kullanarak sınıfını tahmin eder.
    """
    # Transform tanımları (Validation/Test sırasında kullanılanların aynısı)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Görüntüyü yükle
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Görüntü okunurken hata oluştu: {e}")
        return None

    # Tenöre çevir ve boyut ekle (Batch boyutu için)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Modeli hazırla
    model = get_model(num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e} \n(Dosya yolunu veya num_classes değerini kontrol edin.)")
        return None
    
    model.eval()

    # Tahmin işlemi
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

    predicted_label = class_names[top_class.item()]
    confidence = top_prob.item() * 100

    print(f"\n--- Tahmin Sonucu ---")
    print(f"Kullanılan Görüntü: {image_path}")
    print(f"Tahmin Edilen Sınıf: {predicted_label} (Güven Skoru: %{confidence:.2f})")

    return predicted_label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tek bir atık görüntüsünü sınıflandırır.')
    parser.add_argument('--image', type=str, required=True, help='Tahmin edilecek görüntünün dosya yolu')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Eğitilmiş model ağırlıklarının dosya yolu (.pth)')
    
    args = parser.parse_args()

    # Örnek TrashNet sınıfları (Eğer sizin klasör isimleriniz farklıysa bu kısmı güncelleyin)
    CLASSES = ["Cam", "Kağıt", "Karton", "Metal", "Organik", "Plastik"]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Kullanılan Cihaz: {device}")

    predict_single_image(args.image, args.model, len(CLASSES), CLASSES, device=device)

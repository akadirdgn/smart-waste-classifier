"""
TrashNet ResNet18 → TFLite Dönüştürücü
=======================================
Kullanım:
  cd C:\\Users\\kadir\\Desktop\\TrashNet
  pip install onnx onnxruntime onnx-tf tensorflow
  python scripts/convert_to_tflite.py

Çıktılar:
  models/best_model.onnx
  models/best_model.tflite
  assets/ml/waste_classifier.tflite
  assets/ml/labels.txt
"""

import os
import sys
import shutil
import torch
import torch.nn as nn
from torchvision import models

# ─── Yollar ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
SRC_DIR     = os.path.join(ROOT_DIR, "src")
MODEL_PTH   = os.path.join(ROOT_DIR, "models", "best_model.pth")
ONNX_PATH   = os.path.join(ROOT_DIR, "models", "best_model.onnx")
TF_DIR      = os.path.join(ROOT_DIR, "models", "tf_savedmodel")
TFLITE_PATH = os.path.join(ROOT_DIR, "models", "best_model.tflite")

# EcoTrack assets klasörü (Flutter projesi)
ECOTRACK_DIR   = os.path.join(os.path.expanduser("~"), "Desktop", "ecotrack")
ASSETS_ML_DIR  = os.path.join(ECOTRACK_DIR, "assets", "ml")

# ─── Sınıf Etiketleri (ImageFolder tarafından alfabetik sıralanır) ──────────────
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = 6
IMG_SIZE    = 224  # ResNet18 giriş boyutu

# ─── 1. Modeli Yükle ────────────────────────────────────────────────────────────
def load_model():
    print("[1/4] PyTorch modeli yükleniyor...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(MODEL_PTH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"      ✓ Yüklendi: {MODEL_PTH}")
    return model

# ─── 2. ONNX'e Dönüştür ────────────────────────────────────────────────────────
def convert_to_onnx(model):
    print("[2/4] ONNX formatına dönüştürülüyor...")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"      ✓ ONNX kaydedildi: {ONNX_PATH}")

    # Doğrula
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("      ✓ ONNX model doğrulandı.")

# ─── 3. TFLite'a Dönüştür (onnx-tf yöntemi) ──────────────────────────────────
def convert_to_tflite():
    print("[3/4] TFLite formatına dönüştürülüyor...")

    try:
        # Yöntem A: onnx-tf (tavsiye edilen)
        from onnx_tf.backend import prepare
        import onnx, tensorflow as tf

        onnx_model = onnx.load(ONNX_PATH)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(TF_DIR)
        print(f"      ✓ TF SavedModel kaydedildi: {TF_DIR}")

        # TFLite dönüşüm
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_DIR)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 quantization
        # converter.target_spec.supported_types = [tf.float16]  # FP16 (boyut ↓)
        tflite_model = converter.convert()

    except Exception as e:
        print(f"      [!] onnx-tf kütüphanesi yüklenemedi veya desteklenmiyor: {e}")
        print("      [!] onnx2tf + tf yöntemi deneniyor...")
        try:
            # Yöntem B: onnx2tf (daha yeni alternatif)
            import subprocess
            print("      Dönüşüm sürüyor (bu işlem 1-2 dakika alabilir)...")
            result = subprocess.run(
                [sys.executable, "-m", "onnx2tf", "-i", ONNX_PATH, "-o", TF_DIR, "-osd"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            print(f"      ✓ onnx2tf ile TF SavedModel oluşturuldu: {TF_DIR}")

            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_saved_model(TF_DIR)
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
        except Exception as e2:
            print(f"\n❌ HATA: TFLite dönüşümü başarısız: {e2}")
            print_manual_instructions()
            sys.exit(1)

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"      ✓ TFLite kaydedildi: {TFLITE_PATH} ({size_mb:.1f} MB)")
    return tflite_model

# ─── 4. EcoTrack Assets'e Kopyala ────────────────────────────────────────────
def copy_to_ecotrack(tflite_model):
    print("[4/4] EcoTrack assets klasörüne kopyalanıyor...")

    os.makedirs(ASSETS_ML_DIR, exist_ok=True)

    # .tflite dosyasını kopyala
    dest_tflite = os.path.join(ASSETS_ML_DIR, "waste_classifier.tflite")
    with open(dest_tflite, "wb") as f:
        f.write(tflite_model)
    print(f"      ✓ {dest_tflite}")

    # labels.txt oluştur
    dest_labels = os.path.join(ASSETS_ML_DIR, "labels.txt")
    with open(dest_labels, "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES))
    print(f"      ✓ {dest_labels}")

    print()
    print("=" * 60)
    print("  DÖNÜŞÜM TAMAMLANDI!")
    print(f"  TFLite Model : {dest_tflite}")
    print(f"  Etiketler    : {dest_labels}")
    print("=" * 60)
    print()
    print("Sonraki adım: EcoTrack klasöründe 'flutter pub get' çalıştırın.")

def print_manual_instructions():
    """onnx-tf kurulu değilse manuel talimatlar."""
    print()
    print("=" * 60)
    print("  MANUEL KURULUM GEREKLİ")
    print("=" * 60)
    print("  Aşağıdaki komutları sırayla çalıştırın:")
    print()
    print("  pip install onnx-tf==1.10.0 tensorflow==2.12.0")
    print("  # veya")
    print("  pip install onnx2tf tensorflow")
    print()
    print("  Sonra tekrar çalıştırın:")
    print("  python scripts/convert_to_tflite.py")

# ─── Ana Akış ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TrashNet → TFLite Dönüştürücü")
    print("=" * 60)
    print(f"  Model   : {MODEL_PTH}")
    print(f"  Sınıflar: {CLASS_NAMES}")
    print()

    if not os.path.exists(MODEL_PTH):
        print(f"❌ HATA: Model dosyası bulunamadı: {MODEL_PTH}")
        sys.exit(1)

    model      = load_model()
    convert_to_onnx(model)
    tflite_model = convert_to_tflite()
    copy_to_ecotrack(tflite_model)

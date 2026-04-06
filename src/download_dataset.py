"""
TrashNet Dataset İndirme Scripti
=================================
Kaynak  : https://www.kaggle.com/datasets/garythung/trashnet
Yöntem  : kagglehub (Kaggle API anahtarı ile)
Sınıflar: cardboard, glass, metal, paper, plastic, trash
"""

import os
import sys
import shutil

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def check_kaggle_credentials():
    """Kaggle API anahtarı mevcut mu kontrol et."""
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        print(f"✅ Kaggle API anahtarı bulundu: {kaggle_json}")
        return True
    else:
        print("❌ Kaggle API anahtarı bulunamadı.")
        print()
        print("=" * 60)
        print("  Kaggle API Anahtarı Nasıl Alınır?")
        print("=" * 60)
        print("  1. https://www.kaggle.com adresine gidin")
        print("  2. Hesabınıza giriş yapın → Profil → Settings")
        print("  3. API bölümünde 'Create New Token' tıklayın")
        print("  4. İndirilen kaggle.json dosyasını şu konuma taşıyın:")
        print(f"       {kaggle_json}")
        print("  5. Bu scripti tekrar çalıştırın.")
        print("=" * 60)
        return False

def download_with_kagglehub():
    """kagglehub ile dataset'i indir."""
    try:
        import kagglehub
    except ImportError:
        print("📦 kagglehub yükleniyor...")
        os.system("pip install kagglehub -q")
        import kagglehub

    print("⬇️  Dataset indiriliyor: garythung/trashnet")
    print("   (Bu işlem internet hızınıza göre birkaç dakika sürebilir...)\n")

    try:
        path = kagglehub.dataset_download("garythung/trashnet")
        print(f"\n✅ Dataset başarıyla indirildi!")
        print(f"   Kaynak konum: {path}")
        return path
    except Exception as e:
        print(f"\n❌ kagglehub indirme hatası: {e}")
        return None

def find_dataset_root(downloaded_path):
    """İndirilen klasörde 'dataset-resized' dizinini bul."""
    for root, dirs, files in os.walk(downloaded_path):
        if "dataset-resized" in dirs:
            return os.path.join(root, "dataset-resized")
        # Doğrudan sınıf klasörleri varsa
        expected_classes = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
        if expected_classes.issubset(set(dirs)):
            return root
    return None

def copy_to_raw(dataset_root):
    """Dataset'i proje dizinine kopyala."""
    dest = os.path.join(RAW_DIR, "dataset-resized")

    if os.path.exists(dest):
        print(f"\n⚠️  Hedef klasör zaten mevcut: {dest}")
        ans = input("   Üzerine yazılsın mı? (e/H): ").strip().lower()
        if ans != "e":
            print("   İşlem iptal edildi.")
            return False
        shutil.rmtree(dest)

    print(f"\n📁 Kopyalanıyor → {dest}")
    shutil.copytree(dataset_root, dest)
    print("✅ Kopyalama tamamlandı!")
    return True

def verify_dataset(dest_dir):
    """Dataset yapısını ve görüntü sayılarını doğrula."""
    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    total = 0

    print("\n" + "=" * 50)
    print("  Dataset Doğrulama")
    print("=" * 50)
    print(f"  {'Sınıf':<12} {'Görüntü Sayısı':>15}")
    print("-" * 50)

    all_ok = True
    for cls in expected_classes:
        cls_path = os.path.join(dest_dir, cls)
        if not os.path.exists(cls_path):
            print(f"  {cls:<12} {'❌ Klasör bulunamadı!':>15}")
            all_ok = False
            continue
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPG"))])
        total += count
        print(f"  {cls:<12} {count:>12} görüntü")

    print("-" * 50)
    print(f"  {'TOPLAM':<12} {total:>12} görüntü")
    print("=" * 50)

    if all_ok:
        print("\n✅ Dataset doğrulandı! Tüm sınıf klasörleri mevcut.\n")
    else:
        print("\n⚠️  Bazı sınıf klasörleri eksik!\n")

    return all_ok

def main():
    print("=" * 60)
    print("  TrashNet Dataset İndirme Aracı")
    print("=" * 60)
    print()

    # 1. Kaggle kimlik bilgilerini kontrol et
    if not check_kaggle_credentials():
        sys.exit(1)

    # 2. data/raw dizinini oluştur
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"\n📂 Hedef dizin: {os.path.abspath(RAW_DIR)}")

    # 3. Eğer dataset zaten mevcutsa kontrol et
    existing = os.path.join(RAW_DIR, "dataset-resized")
    if os.path.exists(existing):
        print(f"\n✅ Dataset zaten mevcut: {existing}")
        print("   Doğrulama yapılıyor...")
        verify_dataset(existing)
        print("   Yeniden indirmek istemiyorsanız scripti sonlandırabilirsiniz.")
        ans = input("   Yeniden indir? (e/H): ").strip().lower()
        if ans != "e":
            sys.exit(0)

    # 4. Download
    downloaded_path = download_with_kagglehub()
    if downloaded_path is None:
        print("\n❌ İndirme başarısız. Kaggle API anahtarınızı kontrol edin.")
        sys.exit(1)

    # 5. dataset-resized klasörünü bul
    dataset_root = find_dataset_root(downloaded_path)
    if dataset_root is None:
        print(f"\n⚠️  'dataset-resized' klasörü bulunamadı.")
        print(f"   İndirilen konum: {downloaded_path}")
        print("   Lütfen klasör yapısını manuel kontrol edin.")
        sys.exit(1)

    print(f"\n📂 Dataset kökü bulundu: {dataset_root}")

    # 6. Proje dizinine kopyala
    if not copy_to_raw(dataset_root):
        sys.exit(1)

    # 7. Doğrula
    verify_dataset(os.path.join(RAW_DIR, "dataset-resized"))

    print("🎉 Dataset hazır! Sonraki adım: prepare_data.py çalıştırın.")
    print("   python src/prepare_data.py")
    print()


if __name__ == "__main__":
    main()

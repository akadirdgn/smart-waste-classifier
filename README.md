# 🗑️ TrashNet — Transfer Learning ile Akıllı Atık Sınıflandırma

> ResNet18 tabanlı Transfer Learning kullanarak atıkları 6 kategoriye ayıran görüntü sınıflandırma projesi.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-%2589%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Dataset](#-dataset)
- [Model Mimarisi](#-model-mimarisi)
- [Sonuçlar](#-sonuçlar)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)

---

## 🎯 Proje Hakkında

Bu proje, **Transfer Learning** yöntemiyle ImageNet üzerinde önceden eğitilmiş bir **ResNet18** modelini atık sınıflandırma görevine uyarlıyor. İki fazlı eğitim stratejisi uygulanmıştır:

| Faz | Strateji | Epoch | LR |
|-----|----------|-------|-----|
| **Faz 1** | Feature Extraction (sadece FC katmanı) | 5 | `1e-3` |
| **Faz 2** | Fine-Tuning (tüm model) | 15 | `1e-4` |

---

## 📦 Dataset

**Kaynak:** [TrashNet — Kaggle (feyzazkefe/trashnet)](https://www.kaggle.com/datasets/feyzazkefe/trashnet)  
**Orijinal:** Gary Thung & Mindy Yang, Stanford University

| Sınıf | Toplam | Train (%70) | Val (%15) | Test (%15) |
|-------|--------|-------------|-----------|------------|
| cardboard | 403 | 282 | 60 | 61 |
| glass | 501 | 350 | 75 | 76 |
| metal | 410 | 287 | 61 | 62 |
| paper | 594 | 415 | 89 | 90 |
| plastic | 482 | 337 | 72 | 73 |
| trash | 137 | 95 | 20 | 22 |
| **Toplam** | **2.527** | **1.766** | **377** | **384** |

---

## 🧠 Model Mimarisi

```
ResNet18 (ImageNet pretrained)
├── Conv + BatchNorm + ReLU
├── MaxPool
├── Layer1 - Layer4 (Residual Blocks)  ← Faz 1'de donduruldu
└── FC: 512 → 6                        ← Her iki fazda da eğitildi
```

---

## 📊 Sonuçlar

### Validation Seti (%93 Accuracy)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| cardboard | 1.00 | 0.92 | 0.96 |
| glass | 0.87 | 0.95 | 0.90 |
| metal | 0.95 | 0.90 | 0.92 |
| paper | 0.92 | 0.98 | 0.95 |
| plastic | 0.96 | 0.90 | 0.93 |
| trash | 0.84 | 0.80 | 0.82 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** |

### Test Seti (%89 Accuracy)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| cardboard | 0.98 | 0.89 | 0.93 |
| glass | 0.84 | 0.83 | 0.83 |
| metal | 0.87 | 0.87 | 0.87 |
| paper | 0.90 | 0.94 | 0.92 |
| plastic | 0.85 | 0.90 | 0.87 |
| trash | 0.95 | 0.86 | 0.90 |
| **Weighted Avg** | **0.89** | **0.89** | **0.89** |

---

## ⚙️ Kurulum

```bash
# Repoyu klonla
git clone https://github.com/abdulkadrdoan/TrashNet.git
cd TrashNet

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 🚀 Kullanım

### 1. Dataset İndir

```bash
# Kaggle API anahtarı gerekli (~/.kaggle/kaggle.json)
python src/download_dataset.py
```

### 2. Train/Val/Test Split Hazırla

```bash
python src/prepare_data.py
```

### 3. Modeli Eğit

```bash
python src/train.py
# Eğitim ~60-80 dk sürer (CPU). GPU varsa ~10 dk.
# En iyi model: models/best_model.pth
# Grafik: results/training_history.png
```

### 4. Modeli Değerlendir

```bash
python src/evaluate.py
# Confusion matrix: results/confusion_matrix.png
# results/confusion_matrix_test.png
```

### 5. Tek Görüntü Tahmin Et

```bash
python src/predict.py --image "yol/goruntu.jpg" --model "models/best_model.pth"
```

---

## 📁 Proje Yapısı

```
TrashNet/
├── data/
│   ├── raw/dataset-resized/      ← İndirilen ham dataset
│   └── split/                    ← Train/Val/Test split
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── best_model.pth            ← Eğitilmiş model ağırlıkları
├── results/
│   ├── training_history.png      ← Loss & Accuracy grafikleri
│   ├── confusion_matrix.png      ← Val seti karışıklık matrisi
│   └── confusion_matrix_test.png ← Test seti karışıklık matrisi
├── src/
│   ├── download_dataset.py       ← Dataset indirme
│   ├── prepare_data.py           ← Train/Val/Test split
│   ├── dataset.py                ← DataLoader yardımcıları
│   ├── model.py                  ← ResNet18 model tanımı
│   ├── train.py                  ← Eğitim scripti
│   ├── evaluate.py               ← Değerlendirme & metrikler
│   └── predict.py                ← Tek görüntü tahmini
├── requirements.txt
└── README.md
```

---

## 📚 Teknolojiler

| Kütüphane | Amaç |
|-----------|------|
| PyTorch | Derin öğrenme framework'ü |
| torchvision | ResNet18 pretrained modeli & transformlar |
| scikit-learn | Metrikler (F1, confusion matrix) |
| matplotlib / seaborn | Görselleştirme |
| kagglehub | Dataset indirme |
| tqdm | İlerleme çubukları |

---

## 👤 Geliştirici

**Abdulkadir Doğan**  
GitHub: [@akadirdgn](https://github.com/akadirdgn)

# Klasyfikacja Guzów Mózgu z Obrazów MRI

Projekt deep learning do klasyfikacji obrazów MRI mózgu w celu wykrywania guzów przy użyciu konwolucyjnych sieci neuronowych (CNN) z PyTorch.

## Przegląd Projektu

Ten projekt implementuje model CNN do klasyfikacji skanów MRI mózgu w różne kategorie, pomagając w wykrywaniu i analizie guzów mózgu. Model wykorzystuje zaawansowane techniki wizji komputerowej do osiągnięcia wysokiej dokładności w klasyfikacji obrazów medycznych.

## Zbiór Danych

- **Źródło**: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
- **Lokalizacja**: `archive-4/brain_tumor_dataset`
- **Podział**: 80% trening, 20% walidacja
- **Rozmiar obrazu**: 64x64 pikseli
- **Klasy**: 2 kategorie (guz/brak guza)

## Architektura Modelu

Model CNN składa się z:
- **Ekstrakcja cech**: 3 bloki konwolucyjne z BatchNorm i Dropout
- **Kanały**: 64 → 128 → 256
- **Klasyfikator**: W pełni połączone warstwy z regularyzacją dropout
- **Wyjście**: 2 klasy

### Kluczowe Funkcje:
- Batch Normalization dla stabilnego treningu
- Warstwy Dropout zapobiegające przeuczeniu
- MaxPooling dla redukcji wymiarowości
- Funkcje aktywacji ReLU

## Wymagania

```python
torch
torchvision
matplotlib
numpy
scikit-learn
seaborn
```

## Użycie

1. **Przygotowanie Danych**: Ładowanie i przetwarzanie obrazów MRI
2. **Trening Modelu**: Trenowanie CNN przez 19 epok
3. **Ewaluacja**: Analiza wydajności modelu z macierzą konfuzji
4. **Wizualizacja**: Przeglądanie predykcji z wynikami pewności

## Wyniki

Model zapewnia:
- Śledzenie dokładności treningu i walidacji
- Analiza macierzy konfuzji
- Wizualizacja predykcji z wynikami pewności
- Raport klasyfikacji z metrykami precision/recall

## Wizualizacja

<img width="1047" height="783" alt="Zrzut ekranu 2025-07-21 o 14 21 59" src="https://github.com/user-attachments/assets/be7c4bc8-7035-46a3-86d2-3c1ea41d5165" />


Projekt zawiera kompleksowe narzędzia wizualizacji:
- Wyświetlanie przykładowych obrazów
- Krzywe strat treningu/walidacji
- Mapa cieplna macierzy konfuzji
- Analiza pewności predykcji
- Przykłady indywidualnych predykcji

## Struktura Plików

```
Brain MRI/
├── brain-tumor-cnn-classifier.ipynb # Główny notebook z kompletnym pipeline
├── archive-4/
│   └── brain_tumor_dataset/   # Katalog ze zbiorem danych
└── runs/                      # Logi TensorBoard
```
## Autor 
Franciszek Łasiński 
---------

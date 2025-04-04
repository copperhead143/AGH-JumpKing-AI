# Dokumentacja Wymagań: AI dla Jump Kinga

## 1. Cele Projektu
- **Główny cel:** Stworzenie modelu AI, który ukończy grę Jump King.
- **Szczegółowe cele:**
  - Zintegrowanie uczenia modelu z kodem gry
  - Stworzenie środowiska treningowego.
  - Wdrożenie algorytmu DQN.
  - Definicja i optymalizacja funkcji nagrody.
  - Monitorowanie i logowanie postępów treningowych.

## 2. Zakres Wymagań
- **Funkcjonalne:**
  - **Integracja z kodem źródłowym gry** 
  - **Algorytmy RL:** Implementacja wybranych metod i mechanizmu replay buffer.
  - **Monitorowanie:** Logi i wizualizacja.
- **Niefunkcjonalne:**
  - Stabilność i odporność na błędy.

## 3. Wymagane Zasoby
  - Python, framework TensorFlow
  - Numpy
  - OpenCV
  - TensorBoard

## 4. Określenie Problemu
- **Problem:**  
  Stworzenie AI, które pokona grę Jump King, optymalizując decyzje w dynamicznym środowisku.
- **Wyzwania:**
  - Definicja funkcji nagrody adekwatnej do postępów w grze.
  - Utrzymanie stabilności procesu uczenia oraz optymalizacja działania agenta.
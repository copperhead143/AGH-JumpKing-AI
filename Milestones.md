# Kamień 1: Dokumentacja Wymagań AI dla Jump Kinga

## 1.1. Cele Projektu
- **Główny cel:** Stworzenie modelu AI, który ukończy grę Jump King.
- **Szczegółowe cele:**
  - Zintegrowanie uczenia modelu z kodem gry
  - Stworzenie środowiska treningowego.
  - Wdrożenie algorytmu DQN.
  - Definicja i optymalizacja funkcji nagrody.
  - Monitorowanie i logowanie postępów treningowych.

## 1.2. Zakres Wymagań
- **Funkcjonalne:**
  - **Integracja z kodem źródłowym gry** 
  - **Algorytmy RL:** Implementacja wybranych metod i mechanizmu replay buffer.
  - **Monitorowanie:** Logi i wizualizacja.
- **Niefunkcjonalne:**
  - Stabilność i odporność na błędy.

## 1.3. Wymagane Zasoby
  - Python, framework TensorFlow
  - Numpy
  - OpenCV
  - TensorBoard

## 1.4. Określenie Problemu
- **Problem:**  
  Stworzenie AI, które pokona grę Jump King, optymalizując decyzje w dynamicznym środowisku.
- **Wyzwania:**
  - Definicja funkcji nagrody adekwatnej do postępów w grze.
  - Utrzymanie stabilności procesu uczenia oraz optymalizacja działania agenta.


---

---

# Kamień 2: Zbiór danych

## 2.1. Gra jako zbiór danych
W projekcie wykorzystujemy klon gry Jump King napisany w Pythonie jako główne źródło danych. Gra została zmodyfikowana tak, aby automatycznie rejestrować wszystkie istotne informacje dotyczące rozgrywki, co umożliwia zbieranie danych niezbędnych do treningu modelu AI. Dane te obejmują:
- Stany gry (np. pozycja postaci, poziom trudności, aktualne przeszkody)
- Akcje podejmowane przez gracza (np. skoki, ruchy w lewo/prawo)
- Wyniki rozgrywki (np. ukończenie poziomu)

## 2.2. Zasady gry
Aby poprawnie przygotować zbiór danych, niezbędne jest zrozumienie zasad rządzących rozgrywką w Jump King:
- **Mechanika rozgrywki:** Gra polega na wykonywaniu precyzyjnych skoków, które mają prowadzić do zdobycia kolejnych poziomów. Każdy ruch gracza wpływa na zmianę stanu gry.
- **Cele i przeszkody:** Gracz musi pokonać szereg poziomów aby osiągnąć jak najdalszy postęp. Zrozumienie tych elementów pozwala na odpowiednie oznaczanie danych, gdzie sukces (przesunięcie się do przodu) jest nagradzany, a błędy lub powtórne próby – karane. W niektórych przypadkach dopuszczalny jest spadek, w celu przejścia dalej. Wynika to ze sposobu zaprojektowania niektórych poziomów.
- **Warunki zwycięstwa:** Ukończenie gry lub osiągnięcie określonego punktu kontrolnego definiuje sukces rozgrywki, co ma bezpośredni wpływ na projektowanie funkcji nagrody.

## 2.3. Zasady uczenia modelu
Proces uczenia modelu AI opiera się na metodach uczenia przez wzmacnianie, a dokładniej na implementacji algorytmu DQN. Kluczowe elementy tego etapu to:
- **Definicja funkcji nagrody:** Funkcja nagrody została zaprojektowana tak, aby premiować postęp w grze, osiąganie kolejnych poziomów oraz unikanie błędów. Odpowiednia definicja nagrody jest kluczowa dla skutecznego treningu modelu.
- **Mechanizm replay buffer:** Zastosowany został mechanizm replay buffer, który przechowuje doświadczenia agenta. Pozwala to na stabilizację procesu uczenia poprzez ponowne wykorzystanie danych z poprzednich epizodów.
- **Strategia eksploracji i eksploatacji:** Wdrożona strategia umożliwia modelowi balansowanie między odkrywaniem nowych rozwiązań a wykorzystaniem już zdobytej wiedzy, co jest kluczowe w dynamicznym środowisku gry.
- **Optymalizacja i tuning:** Parametry algorytmu, takie jak współczynnik uczenia, wielkość batcha oraz częstotliwość aktualizacji, zostały zoptymalizowane w oparciu o przeprowadzone eksperymenty i analizę danych.

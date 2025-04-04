# Kamień 1: Dokumentacja Wymagań AI dla Jump Kinga

## 1.1. Cele Projektu
- **Główny cel:** Stworzenie modelu AI, który ukończy grę *Jump King*.
- **Szczegółowe cele:**
  - Integracja procesu uczenia modelu z kodem gry.
  - Stworzenie dedykowanego środowiska treningowego.
  - Implementacja algorytmu *Deep Q-Network (DQN)*.
  - Definicja oraz optymalizacja funkcji nagrody.
  - Monitorowanie i analiza postępów treningowych.

## 1.2. Zakres Wymagań
- **Funkcjonalne:**
  - **Integracja z kodem źródłowym gry.**
  - **Implementacja algorytmów RL:** Wdrożenie metody DQN z mechanizmem *replay buffer*.
  - **Monitorowanie:** Logowanie i wizualizacja procesu treningowego.
- **Niefunkcjonalne:**
  - Stabilność oraz odporność na błędy.
  - Optymalizacja czasu treningu i wydajności.

## 1.3. Wymagane Zasoby
- **Język programowania:** Python
- **Biblioteki i narzędzia:**
  - TensorFlow
  - NumPy
  - OpenCV
  - TensorBoard

## 1.4. Określenie Problemu
- **Problem:**
  Stworzenie AI, które skutecznie ukończy grę *Jump King*, optymalizując podejmowane decyzje.
- **Wyzwania:**
  - Opracowanie odpowiedniej funkcji nagrody odzwierciedlającej postęp w grze.
  - Stabilizacja procesu uczenia oraz optymalizacja działania agenta.
  - Skuteczne zarządzanie eksploracją i eksploatacją w dynamicznym środowisku gry.

---

# Kamień 2: Zbór Danych

## 2.1. Gra jako Zbór Danych
W projekcie wykorzystujemy klon gry *Jump King* napisany w Pythonie jako główne źródło danych. Gra została zmodyfikowana w celu automatycznego rejestrowania istotnych informacji dotyczących rozgrywki, co umożliwia zbieranie danych niezbędnych do treningu modelu AI. Zebrane dane obejmują:
- Stany gry (np. pozycja postaci, aktualne przeszkody).
- Akcje podejmowane przez gracza (np. skoki, ruchy w lewo/prawo).
- Pokonane etapy gry oraz powtarzające się błędy.

## 2.2. Zasady Gry
Aby poprawnie przygotować zbór danych, niezbędne jest dogłębne zrozumienie zasad rządzących rozgrywką w *Jump King*:
- **Mechanika rozgrywki:**
  - Gra polega na wykonywaniu precyzyjnych skoków, umożliwiających przemieszczanie się po wyżej położonych platformach.
  - Każdy ruch gracza zmienia stan gry i wpływa na kolejne decyzje modelu.
- **Cele i przeszkody:**
  - Gracz musi pokonać szereg etapów, aby osiągnąć szczyt i ukończyć grę.
  - Skuteczna analiza sukcesów i porażek umożliwia lepsze oznaczanie danych.
- **Warunki zwycięstwa:**
  - Sukcesem jest ukończenie gry lub osiągnięcie określonego punktu kontrolnego.
  - Te kryteria są kluczowe dla poprawnej definicji funkcji nagrody.

## 2.3. Zasady Uczenia Modelu
Proces uczenia AI opiera się na metodach *Reinforcement Learning (RL)*, w szczególności na algorytmie *Deep Q-Network (DQN)*. Kluczowe elementy tego procesu to:
- **Definicja funkcji nagrody:**
  - Premiuje postęp w grze oraz osiąganie kolejnych poziomów.
  - Karze za błędy, takie jak duże spadki, ale uwzględnia sytuacje, gdzie spadek może być strategicznie korzystny.
- **Mechanizm *replay buffer*:**
  - Przechowuje doświadczenia agenta, co stabilizuje proces uczenia poprzez ponowne wykorzystanie danych z poprzednich epizodów.
- **Strategia eksploracji i eksploatacji:**
  - Zapewnia równowagę między odkrywaniem nowych strategii a wykorzystywaniem zdobytej wiedzy.
- **Optymalizacja i tuning:**
  - Kluczowe parametry, takie jak współczynnik uczenia (*learning rate*), wielkość *batcha* oraz częstotliwość aktualizacji modelu, są dostosowywane na podstawie eksperymentów i analizy wyników.


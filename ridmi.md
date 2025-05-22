# Kamień 1: Dokumentacja Wymagań AI dla *Jump Kinga*

## 1.1. Cele Projektu
- **Główny cel:**  
  Zaprojektować i wdrożyć moduł AI, który automatycznie planuje i wykonuje precyzyjne skoki w klonie gry *Jump King*, dążąc do przejścia kolejnych poziomów przy minimalnej liczbie prób i utknięć.
- **Cele szczegółowe:**  
  1. **Integracja z kodem gry** – zaimplementować klasę `AdaptiveJumpKingAI` współdziałającą z obiektem `king`.  
  2. **Modelowanie fizyki** – opracować `JumpPhysicsCalculator`, symulujący trajektorie z uwzględnieniem grawitacji, odbić od ścian i sufitów.  
  3. **Planowanie sekwencji skoków** – stworzyć `SmartJumpPlanner`, który ocenia alternatywne trajektorie i wybiera optymalną.  
  4. **Mechanizm adaptacji** – wykrywać sytuacje „stuck” i automatycznie replanując trasę skoku.  
  5. **Monitorowanie i debug** – rozbudować `get_debug_info()` o metryki sukcesu/porażki i narzędzia do wizualizacji przebiegu treningu.

## 1.2. Zakres Wymagań

### 1.2.1. Wymagania funkcjonalne
- **Symulacja trajektorii:**  
  - Obliczanie i wizualizacja punktów lotu  
  - Uwzględnianie odbić i tarcia  
- **Optymalizacja decyzji:**  
  - Ranking skoków wg funkcji `calculate_jump_score`  
  - Wybór trasy minimalizującej ryzyko  
- **Integracja akcji AI:**  
  - Sterowanie metodami `crouch()`, `jump()`, `jump_left()`, `jump_right()`  
- **Adaptacja do błędów:**  
  - Automatyczne wykrywanie utknięć (metoda `is_stuck()`)  
  - Replanujące algorytmy awaryjne

### 1.2.2. Wymagania niefunkcjonalne
- **Wydajność:**  
  - Utrzymanie ≥ 60 FPS podczas symulacji  
- **Stabilność:**  
  - Odporność na błędne odczyty pozycji  
  - Mechanizmy cooldown/resetu  
- **Skalowalność:**  
  - Łatwe dodawanie nowych poziomów i mechanik  
- **Bezpieczeństwo:**  
  - Izolacja AI w wątku pomocniczym  
  - Ograniczenie zużycia pamięci

## 1.3. Wymagane Zasoby
- **Język:** Python ≥ 3.8  
- **Biblioteki:**  
  - `pygame` (rendering + kolizje)  
  - `numpy` / `math` (wektory, obliczenia)  
  - `time` / `logging` (profilowanie + debug)  
- **Środowisko:**  
  - Repozytorium z klonem *Jump King*  

## 1.4. Ryzyka i założenia
- **Założenia:**  
  - Kod gry jest stabilny i dobrze udokumentowany.  
  - `pygame` obsługuje wszystkie potrzebne kolizje.  
- **Ryzyka:**  
  - Niedokładne modelowanie fizyki może prowadzić do nietrafionych skoków.  
  - Przeciążenie CPU przy dużej liczbie symulacji  
  - Potencjalne wycieki pamięci w pętli treningowej  

---

# Kamień 2: Zbiór Danych i Ich Przygotowanie

## 2.1. Źródła i typy danych
1. **Logi skoków** – szczegółowe ścieżki trajektorii wygenerowane przez `JumpPhysicsCalculator`.  
2. **Parametry poziomów** – współrzędne i wymiary platform z obiektów `Level`.  
3. **Metadane eksperymentów** – timestamp, numer próby, ustawienia `planner` (np. współczynniki nagrody).

## 2.2. Przetwarzanie i walidacja
1. **Czyszczenie danych:**  
   - Usunięcie trajektorii zakończonych poza ekranem  
   - Filtrowanie krótkich symulacji (< 10 kroków)  
2. **Normalizacja:**  
   - Skalowanie i zaokrąglanie pozycji do siatki 20 px  
   - Standaryzacja wektorów prędkości  
3. **Etykietowanie:**  
   - „Sukces” vs „porażka” na podstawie detekcji kolizji i osiągnięcia platformy  
   - Dodatkowa etykieta „stuck” dla trajektorii z brakiem postępu przez ≥ 50 kroków  
4. **Walidacja jakości:**  
   - Raportowaie braków i rozkładu etykiet  
   - Wizualizacja rozkładu trajektorii  

## 2.3. Narzędzia i format
- **Skrypty ETL:** Python + `pandas`  
- **Format wyjściowy:**  
  - CSV z kolumnami:  
    - `start_x`, `start_y`, `angle`, `power`  
    - `trajectory_points` (JSON string)  
    - `label` (`success`/`failure`/`stuck`)  
- **Dokumentacja:**  
  - Plik `data_dictionary.md` opisujący każdy atrybut  

## 2.4. Oczekiwane wyniki
- **Dataset przygotowany do treningu:**  
  - Zbalansowany (≤ 60% sukcesów)  
  - Bez braków ani duplikatów  
  - Udokumentowany „data dictionary”  
- **Weryfikacja:**  
  - Skrypt `validate_dataset.py` produkujący raport (PDF/HTML) z KPI  

---

# Kamień 3: Wybór i Implementacja Modelu AI

## 3.1. Wybór algorytmów
- **Rule–based baseline:**  
  - `SmartJumpPlanner` z ręcznie skalibrowaną funkcją `calculate_jump_score`.  
- **Reinforcement Learning (opcja rozwoju):**  
  - *Deep Q-Network (DQN)* w module `AdaptiveJumpKingAI`.  

## 3.2. Prototypowanie i testy
1. **Baseline:**  
   - Porównać rule–based vs losowe skoki na 1000 epizodach.  
2. **Implementacja DQN:**  
   - Sieć konwolucyjna przyjmująca siatkę obrazującą scenę gry.  
   - Replay buffer + ε-greedy policy.  
3. **Metryki oceny:**  
   - **Success Rate** (odsetek ukończonych poziomów)  
   - **Average Trial Length** (liczba prób do ukończenia)  
   - **Average Steps** (kroki symulacji)  

## 3.3. Integracja i walidacja
- **Scenariusze testowe:** 5 poziomów o rosnącej trudności  
- **Testy regresyjne:** Porównanie parametrów `calculate_jump_score` przed/po implementacji DQN  
- **Monitorowanie:**  
  - Dashboard TensorBoard (reward curve, loss)  
  - Dzienniki `get_debug_info()` w formacie JSON  

## 3.4. Oczekiwany rezultat
- **Model rule–based** o success rate ≥ 40% na poziomach 1–3.  
- **Model DQN** (jeśli wdrożony) ≥ 60% success rate na poziomach 1–5.  
- **Pełny raport:**  
  - Porównanie baseline vs DQN  
  - Analiza błędów i propozycje dalszego rozwoju  
- **Kod:**  
  - Moduły `AdaptiveJumpKingAI.py`, `SmartJumpPlanner.py`, testy unitarne i integracyjne  

---

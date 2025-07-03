# Technical Specification Document v2
## Architektura systemu automatycznego odkrywania cech

### Architektura wysokopoziomowa
System wykorzystuje algorytm Monte Carlo Tree Search (MCTS) do inteligentnej eksploracji przestrzeni możliwych transformacji cech. MCTS, znany z zastosowań w grach strategicznych, pozwala na efektywne balansowanie między eksploracją nowych możliwości a eksploatacją obiecujących kierunków. Ewaluacja jakości cech odbywa się przy użyciu biblioteki AutoML, która zapewnia szybkie i wiarygodne oszacowanie wartości predykcyjnej.

### Główne komponenty systemu

#### Silnik przeszukiwania przestrzeni cech (MCTS Engine)
Centralny komponent odpowiedzialny za orchestrację procesu odkrywania cech. Implementuje pełny cykl MCTS:
- **Selekcja węzłów**: Wybór najbardziej obiecującego węzła do eksploracji przy użyciu formuły UCB1 (Upper Confidence Bound)
- **Ekspansja drzewa**: Tworzenie nowych węzłów reprezentujących transformacje cech
- **Symulacja**: Ewaluacja jakości cech przy użyciu modeli ML
- **Propagacja wsteczna**: Aktualizacja statystyk w drzewie na podstawie wyników

Silnik zarządza pamięcią poprzez przycinanie najmniej obiecujących gałęzi i implementuje mechanizmy checkpointingu dla długotrwałych sesji.

#### Moduł ewaluacji ML (AutoML Evaluator)
Komponent wykorzystujący bibliotekę AutoML do szybkiej oceny jakości wygenerowanych cech:
- **Trenowanie modeli**: Automatyczny wybór i trenowanie ensemble modeli gradientowych (GBM, XGBoost, CatBoost)
- **Walidacja krzyżowa**: k-fold cross validation dla stabilnych wyników
- **Metryki oceny**: Wsparcie dla różnych metryk (accuracy, MAP@K, AUC, RMSE)
- **Optymalizacja hiperparametrów**: Automatyczne dostrajanie modeli dla najlepszych wyników
- **Cache wyników**: Zapamiętywanie ewaluacji dla identycznych zestawów cech

#### System zarządzania cechami (Feature Space Manager)
Rozszerzalny framework do definiowania i zarządzania operacjami transformacji:
- **Operacje generyczne**: Uniwersalne transformacje niezależne od domeny
  - Agregacje statystyczne (mean, std, skewness, kurtosis)
  - Transformacje wielomianowe (interakcje, potęgi)
  - Binning i dyskretyzacja (equal-width, quantile-based)
  - Ranking i normalizacja
  - Operacje na danych kategorycznych (encoding, embedding)
  - Przetwarzanie tekstu (bag-of-words, TF-IDF, długość, statystyki)
  - Transformacje czasowe (lag features, rolling statistics)
- **Operacje domenowe**: Specyficzne dla konkretnych typów problemów
  - Wskaźniki agronomiczne dla danych rolniczych
  - Feature engineering dla danych finansowych
  - Transformacje dla danych medycznych
- **Wykrywanie sygnału**: Automatyczne filtrowanie cech bez wartości informacyjnej
- **Kompozycja operacji**: Łączenie prostych operacji w złożone transformacje

#### Warstwa persystencji (Database Layer)
System bazodanowy zapewniający trwałość danych i efektywny dostęp:
- **Baza analityczna**: Kolumnowa baza danych optymalizowana pod kątem zapytań analitycznych
- **Schemat danych**:
  - Tabele sesji: metadane eksploracji, konfiguracja, status
  - Historia eksploracji: wszystkie odwiedzone węzły z wynikami
  - Katalog cech: definicje i metadane wygenerowanych cech
  - Cache ewaluacji: wyniki oceny jakości cech
  - Rejestr zbiorów danych: metadane i hash dla integralności
- **Optymalizacje**:
  - Indeksy na często używanych kolumnach
  - Partycjonowanie po ID sesji dla szybkiego dostępu
  - Kompresja dla efektywnego wykorzystania przestrzeni
- **Transakcyjność**: ACID compliance dla krytycznych operacji

#### Interfejs użytkownika (CLI Interface)
Narzędzia linii poleceń zapewniające pełną kontrolę nad systemem:
- **Główny orchestrator**: Skrypt uruchamiający sesje eksploracji z konfiguracją
- **Manager systemu**: Narzędzie do zarządzania danymi, sesjami i analizy wyników
- **Parsowanie argumentów**: Intuicyjne opcje z walidacją i pomocą kontekstową
- **Formatowanie wyjścia**: Czytelne tabele, progress bars, kolorowe logi
- **Tryb interaktywny**: Możliwość modyfikacji parametrów w trakcie działania

### Przepływ danych w systemie

#### Inicjalizacja sesji
1. Wczytanie konfiguracji (YAML) z parametrami eksploracji
2. Rejestracja lub wczytanie zbioru danych z cache
3. Inicjalizacja drzewa MCTS (nowy lub wznowienie)
4. Przygotowanie modułu ewaluacji z AutoML

#### Cykl eksploracji
1. **Selekcja**: Wybór węzła do ekspansji na podstawie UCB1 score
2. **Generowanie cech**: Aplikacja transformacji dla wybranego węzła
3. **Ewaluacja**: Trenowanie modelu AutoML i ocena jakości
4. **Aktualizacja**: Propagacja wyników w górę drzewa
5. **Persystencja**: Zapis stanu do bazy danych
6. **Iteracja**: Powtórzenie cyklu lub zakończenie

#### Finalizacja
1. Analiza drzewa i wybór najlepszych ścieżek
2. Generowanie kodu Python dla odkrytych cech
3. Tworzenie raportu HTML z wizualizacjami
4. Eksport metryk i statystyk

### Integracje systemowe

#### Biblioteki ML
- **AutoGluon**: Główny framework do automatycznej ewaluacji
- **Gradient Boosting**: LightGBM, XGBoost, CatBoost jako modele bazowe
- **Scikit-learn**: Preprocessing, metryki, utilities
- **PyTorch**: Opcjonalne wsparcie dla modeli głębokich (TabNet)

#### Przetwarzanie danych
- **Pandas**: Główna biblioteka do manipulacji danymi
- **NumPy**: Obliczenia numeryczne i operacje macierzowe
- **SciPy**: Zaawansowane funkcje statystyczne
- **Dask**: Opcjonalne wsparcie dla danych out-of-memory

#### Infrastruktura
- **SQLAlchemy**: Warstwa abstrakcji bazy danych
- **DuckDB**: Embedded baza analityczna
- **Rich**: Terminal UI dla lepszego UX
- **Matplotlib/Seaborn**: Generowanie wykresów

### Architektura warstw (Layered Architecture)

#### Warstwa prezentacji
- CLI parsers i formatters
- Progress monitoring
- Result visualization

#### Warstwa aplikacji
- Session orchestration
- Configuration management
- Export services

#### Warstwa domeny
- MCTS algorithm implementation
- Feature engineering operations
- Evaluation strategies

#### Warstwa infrastruktury
- Database repositories
- File system operations
- Logging and monitoring

### Wzorce projektowe

#### Repository Pattern
Separacja logiki dostępu do danych od logiki biznesowej:
- Abstrakcyjne interfejsy dla operacji CRUD
- Konkretne implementacje dla różnych backendów
- Łatwa testowalność przez mockowanie

#### Factory Pattern
Tworzenie obiektów operacji i ewaluatorów:
- Dynamiczne ładowanie operacji feature engineering
- Konfigurowalny wybór strategii ewaluacji
- Rozszerzalność o nowe typy transformacji

#### Strategy Pattern
Wymienne algorytmy dla różnych aspektów:
- Strategie selekcji węzłów (UCB1, Thompson Sampling)
- Strategie ewaluacji (szybka, dokładna, custom)
- Strategie zarządzania pamięcią

#### Observer Pattern
Monitoring i reagowanie na zdarzenia:
- Progress callbacks dla UI
- Event logging dla debugowania
- Hooki dla rozszerzeń

### Zarządzanie sesjami

#### Stan sesji
- Konfiguracja i parametry
- Drzewo MCTS z pełną historią
- Cache wygenerowanych cech
- Metryki i statystyki

#### Checkpointing
- Automatyczny zapis co N iteracji
- Atomowe operacje zapisu
- Wersjonowanie checkpointów
- Czyszczenie starych stanów

#### Wznowienie sesji
- Odbudowa drzewa z bazy danych
- Weryfikacja integralności
- Kontynuacja od ostatniego stanu
- Merge rezultatów
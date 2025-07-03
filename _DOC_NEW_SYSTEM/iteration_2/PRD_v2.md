# Product Requirements Document v2
## System automatycznego odkrywania cech dla konkurencji ML

### Cel projektu
Stworzenie autonomicznego systemu do automatycznego odkrywania i generowania cech (feature engineering) dla zadań uczenia maszynowego, ze szczególnym fokusem na konkurencje typu Kaggle. System ma wykorzystywać algorytmy przeszukiwania inspirowane sztuczną inteligencją do eksploracji przestrzeni możliwych transformacji danych, automatycznie oceniając ich wartość predykcyjną.

### Główne cele biznesowe
- **Automatyzacja procesu feature engineering**: Redukcja czasu potrzebnego na ręczne tworzenie cech z tygodni do godzin
- **Poprawa wyników modeli**: Odkrywanie nieoczywistych transformacji i interakcji między zmiennymi, które mogą poprawić wyniki o 5-15%
- **Demokratyzacja ML**: Umożliwienie mniej doświadczonym data scientists osiągania wyników na poziomie ekspertów
- **Skalowalność rozwiązań**: Możliwość zastosowania na różnych typach problemów i domenach danych
- **Dokumentacja procesu**: Automatyczne generowanie kodu produkcyjnego dla odkrytych cech

### Kluczowe funkcjonalności

#### System eksploracji przestrzeni cech
- Inteligentne przeszukiwanie możliwych transformacji przy użyciu algorytmu inspirowanego grami strategicznymi
- Balansowanie między eksploracją nowych obszarów a eksploatacją obiecujących kierunków
- Wsparcie dla operacji domenowych (np. specyficzne dla danych czasowych, tekstowych, kategorycznych)
- Możliwość definiowania własnych operacji transformacji

#### Automatyczna ewaluacja jakości
- Szybka ocena wartości predykcyjnej wygenerowanych cech
- Wsparcie dla różnych metryk (accuracy, MAP@K, RMSE, custom metrics)
- Optymalizacja pod kątem konkretnego problemu biznesowego
- Wykrywanie i eliminacja cech bez wartości informacyjnej

#### Zarządzanie sesjami eksploracji
- Możliwość przerwania i wznowienia długotrwałych procesów
- Śledzenie historii eksploracji z pełną reprodukowalnością
- Porównywanie wyników między sesjami
- Eksport najlepszych ścieżek eksploracji

#### System zarządzania danymi
- Rejestracja i wersjonowanie zbiorów danych
- Automatyczne wykrywanie typów kolumn i sugestie transformacji
- Wsparcie dla danych w formatach CSV, Parquet, z możliwością rozszerzenia
- Bezpieczny dostęp do danych z kontrolą uprawnień

#### Analityka i raportowanie
- Wizualizacja procesu eksploracji (drzewo decyzyjne, postęp w czasie)
- Ranking odkrytych cech według wartości predykcyjnej
- Analiza wpływu poszczególnych transformacji
- Generowanie raportów HTML z pełną dokumentacją procesu

### Użytkownicy docelowi

#### Data Scientists w konkurencjach ML
- **Potrzeby**: Szybkie prototypowanie, eksperymentowanie z różnymi podejściami
- **Przypadki użycia**: Kaggle competitions, hackathony, proof-of-concepts
- **Oczekiwania**: Wysoka jakość cech, łatwy eksport do notebooks

#### Zespoły ML w przedsiębiorstwach
- **Potrzeby**: Powtarzalność procesów, integracja z istniejącymi pipeline'ami
- **Przypadki użycia**: Automatyzacja feature engineering dla produkcyjnych modeli
- **Oczekiwania**: Stabilność, skalowalność, dokumentacja

#### Badacze algorytmów AutoML
- **Potrzeby**: Platforma do eksperymentów z nowymi metodami
- **Przypadki użycia**: Publikacje naukowe, rozwój nowych algorytmów
- **Oczekiwania**: Rozszerzalność, dostęp do metryk szczegółowych

### Podstawowe wymagania funkcjonalne

#### Przetwarzanie danych
- Obsługa zbiorów danych od 1MB do 10GB w pamięci
- Wsparcie dla 100+ kolumn i milionów wierszy
- Automatyczna optymalizacja typów danych
- Obsługa brakujących wartości i outlierów

#### Wydajność systemu
- Generowanie 1000+ cech w ciągu godziny
- Ewaluacja pojedynczej cechy w <1 sekundę dla małych zbiorów
- Możliwość równoległego przetwarzania
- Efektywne wykorzystanie pamięci z automatycznym czyszczeniem

#### Integracja i eksport
- Eksport cech jako kod Python (sklearn-compatible)
- Generowanie dokumentacji dla każdej cechy
- API do integracji z zewnętrznymi systemami
- Wsparcie dla formatów wymiany danych (JSON, Parquet)

### Wymagania niefunkcjonalne

#### Użyteczność
- Intuicyjny interfejs CLI z pomocnymi komunikatami
- Czas do pierwszego wyniku <5 minut dla nowych użytkowników
- Kompleksowa dokumentacja z przykładami
- Wsparcie dla różnych poziomów zaawansowania

#### Niezawodność
- Automatyczne zapisywanie postępu co 5 minut
- Odporność na przerwania (graceful shutdown)
- Walidacja danych wejściowych
- Obsługa błędów z informacyjnymi komunikatami

#### Bezpieczeństwo
- Kontrola dostępu do zbiorów danych
- Brak możliwości SQL injection czy path traversal
- Bezpieczne przechowywanie metadanych
- Audit trail dla wszystkich operacji

### Ograniczenia i założenia
- System działa na danych tabelarycznych (structured data)
- Wymaga lokalnego dostępu do danych (no cloud storage w MVP)
- Pojedyncza maszyna (no distributed computing w pierwszej wersji)
- Focus na supervised learning (classification/regression)
# Product Requirements Document v1
## System automatycznego odkrywania cech dla konkurencji ML

### Cel projektu
Stworzenie autonomicznego systemu do automatycznego odkrywania i generowania cech (feature engineering) dla zadań uczenia maszynowego, ze szczególnym fokusem na konkurencje typu Kaggle.

### Główne cele biznesowe
- Automatyzacja procesu feature engineering
- Redukcja czasu potrzebnego na przygotowanie danych
- Poprawa wyników modeli ML poprzez odkrywanie nieoczywistych cech

### Kluczowe funkcjonalności
- Automatyczne przeszukiwanie przestrzeni możliwych transformacji danych
- Ewaluacja jakości wygenerowanych cech przy użyciu modeli ML
- Persystencja i wznowienie długotrwałych sesji eksploracji
- Zarządzanie zbiorami danych i katalogiem cech
- Analityka i raportowanie wyników

### Użytkownicy docelowi
- Data Scientists pracujący nad konkurencjami ML
- Zespoły ML w przedsiębiorstwach
- Badacze algorytmów automatycznego ML

### Podstawowe wymagania
- Wsparcie dla danych tabelarycznych (CSV, Parquet)
- Skalowalność do milionów rekordów
- Możliwość przerwania i wznowienia pracy
- Eksport najlepszych cech do kodu produkcyjnego
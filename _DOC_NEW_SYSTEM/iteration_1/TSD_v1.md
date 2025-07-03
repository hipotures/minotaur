# Technical Specification Document v1
## Architektura systemu automatycznego odkrywania cech

### Architektura wysokopoziomowa
System oparty na algorytmie przeszukiwania drzewa Monte Carlo (MCTS) do eksploracji przestrzeni możliwych transformacji cech, z ewaluacją przy użyciu biblioteki AutoML.

### Główne komponenty

#### Silnik przeszukiwania przestrzeni cech
Implementacja algorytmu MCTS do inteligentnej eksploracji możliwych operacji na danych.

#### Moduł ewaluacji ML
Komponent odpowiedzialny za szybką ocenę jakości wygenerowanych cech przy użyciu modeli uczenia maszynowego.

#### System zarządzania cechami
Katalog operacji transformacji danych z możliwością rozszerzania o nowe domeny.

#### Warstwa persystencji
System bazodanowy do przechowywania stanu eksploracji, wyników i metadanych.

#### Interfejs użytkownika
Narzędzia linii poleceń do interakcji z systemem i zarządzania sesjami.

### Przepływ danych
1. Wczytanie danych źródłowych
2. Generowanie kandydatów transformacji
3. Ewaluacja jakości cech
4. Aktualizacja drzewa przeszukiwania
5. Persystencja wyników
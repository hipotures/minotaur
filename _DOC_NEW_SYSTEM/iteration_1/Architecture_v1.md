# System Architecture Diagrams v1
## Diagramy architektury systemu

### Diagram architektury wysokopoziomowej
```
[Interfejs CLI] → [Warstwa Serwisów] → [Warstwa Logiki Biznesowej] → [Warstwa Danych]
```

### Główne warstwy systemu
- Warstwa prezentacji (CLI)
- Warstwa orchestracji (serwisy)
- Warstwa logiki (MCTS, ewaluacja)
- Warstwa dostępu do danych (repozytoria)
- Warstwa persystencji (baza danych)

### Komponenty i ich relacje
- Moduł MCTS ↔ Moduł Feature Space
- Moduł MCTS → Moduł Ewaluacji
- Wszystkie moduły → Warstwa Bazodanowa
- CLI → Warstwa Serwisów → Repozytoria

### Przepływ sterowania
1. Użytkownik → CLI
2. CLI → Serwis orchestracji
3. Serwis → Silnik MCTS
4. MCTS → Feature Space → Ewaluator
5. Wyniki → Baza danych
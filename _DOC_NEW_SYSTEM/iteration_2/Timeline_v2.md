# Project Timeline v2
## Szczegółowy harmonogram implementacji systemu

### Metodologia realizacji
Projekt będzie realizowany w metodyce Agile z 2-tygodniowymi sprintami. Każda faza kończy się działającym incrementem systemu, który może być przetestowany i zwalidowany z użytkownikami.

### Faza 1: Przygotowanie i fundament (4 tygodnie)

#### Sprint 1-2: Setup architektury i środowiska
**Tydzień 1-2:**
- Konfiguracja repozytorium i struktura projektu
- Setup środowiska developerskiego (Python, UV, pre-commit hooks)
- Implementacja podstawowego loggingu i konfiguracji
- Szkielet aplikacji CLI z podstawowymi komendami
- Setup testów jednostkowych i CI/CD pipeline

**Tydzień 3-4:**
- Implementacja warstwy bazodanowej (DuckDB + SQLAlchemy)
- Podstawowe repozytoria (SessionRepository, DatasetRepository)
- System migracji bazy danych
- Implementacja connection pooling
- Testy integracyjne dla warstwy danych

**Deliverables:**
- ✓ Działające środowisko developerskie
- ✓ Podstawowa struktura aplikacji
- ✓ Warstwa persystencji z testami
- ✓ CI/CD pipeline (GitHub Actions)

### Faza 2: Implementacja core algorytmów (6 tygodni)

#### Sprint 3-4: MCTS Engine
**Tydzień 5-6:**
- Implementacja struktury drzewa MCTS (Node, Tree)
- Algorytm selekcji UCB1
- Mechanizm ekspansji drzewa
- Podstawowa propagacja wsteczna
- Unit testy dla komponentów MCTS

**Tydzień 7-8:**
- Zarządzanie pamięcią i pruning drzewa
- Serializacja/deserializacja stanu
- Checkpointing i recovery
- Integracja z warstwą persystencji
- Performance testy dla dużych drzew

**Deliverables:**
- ✓ Działający silnik MCTS
- ✓ Persystencja stanu drzewa
- ✓ Możliwość wznowienia sesji

#### Sprint 5: Feature Engineering Framework
**Tydzień 9-10:**
- Architektura systemu operacji (base classes)
- Implementacja operacji generycznych:
  - Statistical aggregations
  - Polynomial features
  - Binning operations
  - Ranking transformations
- System rejestracji operacji
- Signal detection (filtrowanie cech bez wartości)
- Feature catalog w bazie danych

**Deliverables:**
- ✓ Rozszerzalny framework operacji
- ✓ Podstawowe transformacje danych
- ✓ Automatyczna detekcja sygnału

### Faza 3: Integracja ML i ewaluacja (4 tygodnie)

#### Sprint 6-7: AutoML Integration
**Tydzień 11-12:**
- Wrapper dla AutoGluon
- Konfiguracja modeli (XGBoost, LightGBM, CatBoost)
- Implementacja metryk (MAP@K, accuracy, RMSE)
- Cache'owanie wyników ewaluacji
- Zarządzanie zasobami (memory, time limits)

**Tydzień 13-14:**
- Integracja ewaluatora z MCTS
- Optymalizacja performance (parallel evaluation)
- Handling różnych typów problemów (classification, regression)
- Comprehensive testing całego pipeline
- Benchmarking na znanych datasetach

**Deliverables:**
- ✓ Zintegrowany system ewaluacji
- ✓ Działający end-to-end pipeline
- ✓ Wyniki porównywalne z baseline

### Faza 4: User Interface i zarządzanie (4 tygodnie)

#### Sprint 8-9: CLI Development
**Tydzień 15-16:**
- Implementacja głównego orchestratora (mcts command)
- Session management commands
- Dataset registration i validation
- Progress monitoring i feedback
- Error handling i recovery

**Tydzień 17-18:**
- Manager tool implementation
- Feature catalog browsing
- Analytics i reporting
- Export functionality (Python code, JSON)
- Comprehensive help system

**Deliverables:**
- ✓ Kompletny interfejs CLI
- ✓ Narzędzia do zarządzania danymi
- ✓ System raportowania

### Faza 5: Testowanie i optymalizacja (4 tygodnie)

#### Sprint 10-11: Quality Assurance
**Tydzień 19-20:**
- Comprehensive test suite completion
- Performance profiling i optimization
- Memory leak detection i fixes
- Security audit (input validation, SQL injection)
- Documentation (user guide, API docs)

**Tydzień 21-22:**
- User acceptance testing
- Bug fixes i stabilizacja
- Performance tuning dla dużych datasetów
- Stress testing (24h runs)
- Release preparation

**Deliverables:**
- ✓ Test coverage >80%
- ✓ Performance benchmarks met
- ✓ Production-ready codebase
- ✓ Complete documentation

### Faza 6: Deployment i launch (2 tygodnie)

#### Sprint 12: Production Release
**Tydzień 23-24:**
- Docker containerization
- Deployment documentation
- Production configuration templates
- Monitoring setup (logs, metrics)
- Release notes i changelog
- Launch preparation

**Deliverables:**
- ✓ Version 1.0 release
- ✓ Docker images
- ✓ Deployment guide
- ✓ Monitoring ready

### Kamienie milowe i punkty kontrolne

| Milestone | Termin | Kryteria sukcesu |
|-----------|--------|------------------|
| M1: Foundation Complete | Tydzień 4 | Środowisko, CI/CD, database layer |
| M2: Core Algorithm Working | Tydzień 10 | MCTS generuje i eksploruje features |
| M3: ML Integration Done | Tydzień 14 | End-to-end pipeline z ewaluacją |
| M4: UI Feature Complete | Tydzień 18 | Wszystkie komendy CLI działają |
| M5: Quality Gates Passed | Tydzień 22 | Testy, performance, security OK |
| M6: Production Release | Tydzień 24 | v1.0 deployed i documented |

### Zarządzanie ryzykiem

#### Identified Risks
1. **AutoGluon performance**: Może być wolniejsze niż oczekiwano
   - Mitigation: Implement fallback to simple models
   - Contingency: Reduce default time limits

2. **Memory usage dla dużych datasetów**: OOM errors
   - Mitigation: Implement streaming processing
   - Contingency: Document size limitations

3. **MCTS convergence**: Może nie znajdować dobrych features
   - Mitigation: Tune exploration parameters
   - Contingency: Provide manual feature hints

4. **Database performance**: Slow queries dla dużych sesji
   - Mitigation: Optimize indexes and queries
   - Contingency: Implement data archiving

### Zależności między zadaniami

```
Foundation ──┐
             ├─→ MCTS Engine ──┐
             │                 ├─→ ML Integration ──┐
Feature Eng ─┘                 │                    ├─→ CLI ──→ Testing ──→ Release
                               │                    │
Database ──────────────────────┴────────────────────┘
```

### Zasoby i zespół

#### Core Team (minimum)
- **Tech Lead**: Architecture, code reviews (1.0 FTE)
- **Backend Developer**: MCTS, features (1.0 FTE)
- **ML Engineer**: AutoGluon, evaluation (0.8 FTE)
- **DevOps**: CI/CD, deployment (0.5 FTE)

#### Extended Team (optimal)
- **QA Engineer**: Test automation (0.8 FTE)
- **Technical Writer**: Documentation (0.5 FTE)
- **UI/UX Designer**: CLI experience (0.3 FTE)

### Budget czasowy

| Faza | Czas | Effort (person-weeks) |
|------|------|----------------------|
| Przygotowanie | 4 tyg | 12 |
| Core algorytmy | 6 tyg | 18 |
| ML integracja | 4 tyg | 12 |
| User interface | 4 tyg | 12 |
| Testing & QA | 4 tyg | 16 |
| Deployment | 2 tyg | 8 |
| **TOTAL** | **24 tyg** | **78 person-weeks** |

### Post-release roadmap

#### Phase 2 (Q2-Q3)
- Web dashboard
- REST API
- Distributed computing (Ray)
- Cloud deployment (K8s)

#### Phase 3 (Q4)
- Multi-user support
- Real-time collaboration
- AutoML beyond tabular
- Plugin marketplace
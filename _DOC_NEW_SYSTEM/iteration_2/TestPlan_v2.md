# Test Plan / Test Strategy v2
## Kompleksowa strategia testowania systemu

### Filozofia testowania
System wymaga rygorystycznego testowania ze względu na złożoność algorytmów ML i krytyczność wyników dla użytkowników. Stosujemy podejście "test pyramid" z największą liczbą szybkich testów jednostkowych u podstawy, mniejszą liczbą testów integracyjnych w środku i najmniejszą liczbą wolnych testów e2e na szczycie.

### Typy testów

#### Testy jednostkowe (Unit Tests)
Testowanie pojedynczych komponentów w izolacji:
- **Coverage target**: >80% dla kodu krytycznego
- **Execution time**: <0.1s per test
- **Frequency**: Przy każdym commit

**Obszary testowania:**
- Algorytm MCTS: selekcja węzłów, UCB1 scoring, tree operations
- Feature operations: poprawność transformacji, edge cases
- Data validation: typy danych, missing values, zakres wartości
- Configuration parsing: walidacja YAML, default values
- Utility functions: formatowanie, konwersje, helpers

**Przykładowe scenariusze:**
- Test UCB1 calculation z różnymi parametrami
- Test feature generation dla pustego DataFrame
- Test obsługi nieprawidłowej konfiguracji
- Test serializacji/deserializacji drzewa MCTS

#### Testy integracyjne (Integration Tests)
Testowanie interakcji między komponentami:
- **Coverage target**: >60% dla głównych flows
- **Execution time**: <10s per test
- **Frequency**: Przed merge do main branch

**Obszary testowania:**
- MCTS + Feature Space: generowanie i ewaluacja cech
- AutoML + Database: cache'owanie wyników
- Session management: save/restore functionality
- CLI + Core: end-to-end command execution
- Database operations: transakcje, rollback

**Przykładowe scenariusze:**
- Pełny cykl MCTS dla małego datasetu
- Wznowienie przerwanej sesji
- Równoczesny dostęp do bazy danych
- Export wyników do różnych formatów

#### Testy wydajnościowe (Performance Tests)
Testowanie zachowania systemu pod obciążeniem:
- **Baseline**: Ustalone metryki wydajności
- **Regression threshold**: <10% degradacji
- **Frequency**: Nightly builds

**Metryki testowane:**
- Czas generowania 1000 cech
- Użycie pamięci dla różnych rozmiarów danych
- Czas ewaluacji AutoML
- Wydajność zapytań do bazy danych
- Czas startu i zamknięcia aplikacji

**Scenariusze obciążeniowe:**
- Dataset 1M rows x 100 columns
- 10k feature operations
- 24h continuous run
- Concurrent session execution

#### Testy end-to-end (E2E Tests)
Testowanie kompletnych scenariuszy użytkownika:
- **Coverage**: Główne user journeys
- **Execution time**: <5 min per scenario
- **Frequency**: Release candidates

**Scenariusze E2E:**
1. **New user onboarding**:
   - Install → Configure → Run first session → View results
2. **Competition workflow**:
   - Load Kaggle data → Explore features → Export best → Submit
3. **Enterprise workflow**:
   - Register datasets → Schedule runs → Generate reports → API access
4. **Recovery scenario**:
   - Start session → Kill process → Resume → Verify continuity

### Strategia testowania per komponent

#### MCTS Engine Testing
```
Unit Tests:
- Node selection logic (UCB1 variants)
- Tree manipulation (add, prune, traverse)
- State serialization
- Memory management

Integration Tests:
- Full MCTS cycle with mock evaluator
- Session persistence and recovery
- Concurrent tree modifications

Performance Tests:
- Tree size vs memory usage
- Selection speed with 10k+ nodes
- Serialization performance
```

#### Feature Engineering Testing
```
Unit Tests:
- Each operation in isolation
- Edge cases (empty, single row, all nulls)
- Type conversions
- Mathematical correctness

Integration Tests:
- Operation chaining
- Signal detection accuracy
- Cache hit/miss scenarios

Property-based Tests:
- Invariants (e.g., rank preserves order)
- Idempotency where applicable
- Output shape consistency
```

#### AutoML Evaluator Testing
```
Unit Tests:
- Metric calculations
- Model configuration
- Result caching logic

Integration Tests:
- Full training pipeline
- Different data types
- Memory constraints

Performance Tests:
- Training time vs data size
- Prediction latency
- Memory usage patterns
```

### Test Data Management

#### Synthetic Data Generation
- **Small datasets**: Programmatically generated for unit tests
- **Statistical properties**: Controlled distributions
- **Edge cases**: Specific patterns to test boundaries
- **Reproducibility**: Seeded random generation

#### Real Data Samples
- **Titanic dataset**: Public, well-understood
- **Iris dataset**: Multi-class classification
- **California housing**: Regression problems
- **Custom datasets**: Domain-specific samples

#### Data Fixtures
```python
@pytest.fixture
def small_dataset():
    """100 rows, mixed types, some nulls"""
    
@pytest.fixture  
def large_dataset():
    """10k rows for performance tests"""
    
@pytest.fixture
def edge_case_dataset():
    """Single column, all nulls, etc."""
```

### Test Environment Setup

#### Local Development
```yaml
test_env:
  database: in-memory DuckDB
  data_size: reduced (1000 rows)
  parallelism: 2 cores max
  timeouts: strict (30s)
```

#### CI Environment
```yaml
ci_env:
  database: file-based DuckDB
  data_size: medium (10k rows)
  parallelism: 4 cores
  matrix:
    - python: [3.9, 3.10, 3.11]
    - os: [ubuntu, macos, windows]
```

#### Performance Environment
```yaml
perf_env:
  database: production-like
  data_size: full datasets
  parallelism: all cores
  monitoring: enabled
```

### Mocking Strategy

#### External Dependencies
- **AutoGluon**: Mock for unit tests, real for integration
- **File system**: tmpdir fixtures
- **Network calls**: Completely mocked
- **Time**: Controlled clock for reproducibility

#### Mock Implementations
```
MockEvaluator:
  - Instant responses
  - Deterministic scores
  - Configurable behavior

MockFeatureGenerator:
  - Fast synthetic features
  - Controllable failures
  - Memory efficient
```

### Test Automation

#### Pre-commit Hooks
- Unit tests for changed files
- Code formatting check
- Type checking
- Security scan

#### CI Pipeline
```yaml
stages:
  - lint: Code quality checks
  - test: Unit + integration tests  
  - build: Package creation
  - perf: Performance regression
  - deploy: Test deployment
```

#### Test Reports
- Coverage reports with fail threshold
- Performance comparison graphs
- Failed test artifacts
- Flaky test detection

### Test Metrics and KPIs

#### Code Coverage
- **Line coverage**: >80% overall
- **Branch coverage**: >70% overall
- **Critical paths**: 100% coverage required

#### Test Quality
- **Flaky test rate**: <1%
- **Average test time**: <0.5s for unit tests
- **Test/code ratio**: ~2:1

#### Defect Metrics
- **Defect escape rate**: <5%
- **Test effectiveness**: >90% defect detection
- **MTTR**: <4 hours for test failures

### Continuous Improvement

#### Test Review Process
- Weekly test failure analysis
- Monthly test suite optimization
- Quarterly strategy review

#### Test Debt Management
- Track untested code
- Prioritize based on risk
- Allocate 20% time for test improvement

#### Learning from Production
- Post-mortem for escaped defects
- Add regression tests
- Update test scenarios
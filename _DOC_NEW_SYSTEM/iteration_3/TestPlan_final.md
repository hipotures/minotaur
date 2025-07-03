# Test Plan / Test Strategy - Final Version
## Kompleksowa strategia testowania systemu z implementacyjnymi detalami

### Filozofia testowania
System wymaga rygorystycznego testowania ze względu na złożoność algorytmów ML i krytyczność wyników dla użytkowników. Stosujemy podejście "test pyramid" z największą liczbą szybkich testów jednostkowych u podstawy, mniejszą liczbą testów integracyjnych w środku i najmniejszą liczbą wolnych testów e2e na szczycie.

**Kluczowe zasady**:
- **Test First**: Pisanie testów przed implementacją (TDD gdzie możliwe)
- **Fast Feedback**: Testy muszą dawać szybką informację zwrotną (<5 minut dla unit tests)
- **Isolation**: Każdy test niezależny, brak shared state
- **Deterministic**: Zawsze ten sam rezultat dla tych samych danych
- **Comprehensive**: Pokrycie happy path, edge cases i error scenarios

### Szczegółowa specyfikacja testów

#### Testy jednostkowe (Unit Tests)

##### Struktura i organizacja
```
tests/unit/
├── test_mcts/
│   ├── test_node.py          # Node operations
│   ├── test_tree.py          # Tree management
│   ├── test_selection.py     # UCB1 algorithm
│   ├── test_expansion.py     # Node expansion
│   └── test_backprop.py      # Backpropagation
├── test_features/
│   ├── test_statistical.py   # Statistical operations
│   ├── test_polynomial.py    # Polynomial features
│   ├── test_binning.py       # Binning strategies
│   └── test_signal.py        # Signal detection
├── test_evaluation/
│   ├── test_metrics.py       # Metric calculations
│   ├── test_cache.py         # Result caching
│   └── test_ensemble.py      # Model ensemble
└── test_utils/
    ├── test_config.py        # Configuration parsing
    ├── test_validation.py    # Data validation
    └── test_formatting.py    # Output formatting
```

##### Przykłady testów jednostkowych

**Test UCB1 Selection**:
```python
class TestUCB1Selection:
    def test_ucb1_calculation_basic(self):
        """Test podstawowego obliczenia UCB1"""
        # Given
        node = create_test_node(visits=10, total_reward=5.0)
        parent_visits = 100
        exploration_constant = 1.4
        
        # When
        ucb_score = calculate_ucb1(node, parent_visits, exploration_constant)
        
        # Then
        expected_avg = 0.5  # 5.0 / 10
        expected_exploration = 1.4 * sqrt(ln(100) / 10)
        expected_score = expected_avg + expected_exploration
        
        assert abs(ucb_score - expected_score) < 1e-6
        
    def test_ucb1_unvisited_node(self):
        """Test UCB1 dla nieodwiedzonego węzła"""
        # Given
        node = create_test_node(visits=0, total_reward=0.0)
        
        # When
        ucb_score = calculate_ucb1(node, parent_visits=100)
        
        # Then
        assert ucb_score == float('inf')
        
    @pytest.mark.parametrize("c_value", [0.1, 0.5, 1.0, 1.4, 2.0])
    def test_ucb1_exploration_parameter(self, c_value):
        """Test wpływu parametru eksploracji"""
        # Test że większe C daje więcej eksploracji
        node = create_test_node(visits=10, total_reward=5.0)
        scores = []
        
        for c in [0.1, 0.5, 1.0, 1.4, 2.0]:
            score = calculate_ucb1(node, 100, c)
            scores.append(score)
            
        # Scores should increase with C
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
```

**Test Feature Generation**:
```python
class TestStatisticalFeatures:
    @pytest.fixture
    def sample_data(self):
        """Fixture z przykładowymi danymi"""
        return pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'B', 'C'],
            'with_nulls': [1, 2, None, 4, 5]
        })
        
    def test_mean_calculation(self, sample_data):
        """Test obliczania średniej"""
        # Given
        operation = MeanOperation()
        
        # When
        result = operation.generate(sample_data, ['numeric'])
        
        # Then
        assert 'numeric_mean' in result.columns
        assert result['numeric_mean'].iloc[0] == 3.0
        
    def test_null_handling(self, sample_data):
        """Test obsługi wartości null"""
        # Given
        operation = MeanOperation()
        
        # When
        result = operation.generate(sample_data, ['with_nulls'])
        
        # Then
        assert 'with_nulls_mean' in result.columns
        assert result['with_nulls_mean'].iloc[0] == 3.0  # (1+2+4+5)/4
        
    def test_empty_dataframe(self):
        """Test dla pustego DataFrame"""
        # Given
        empty_df = pd.DataFrame()
        operation = MeanOperation()
        
        # When/Then
        with pytest.raises(EmptyDataError):
            operation.generate(empty_df, [])
```

**Test Cache Mechanism**:
```python
class TestFeatureCache:
    def test_cache_hit(self):
        """Test trafienia w cache"""
        # Given
        cache = FeatureCache(max_size_mb=100)
        features = pd.DataFrame({'f1': [1, 2, 3]})
        key = 'test_key_123'
        
        # When
        cache.put(key, features)
        retrieved = cache.get(key)
        
        # Then
        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, features)
        
    def test_cache_eviction_lru(self):
        """Test LRU eviction policy"""
        # Given
        cache = FeatureCache(max_size_mb=1)  # Mały cache
        
        # When - dodaj dużo elementów
        for i in range(100):
            large_df = pd.DataFrame(np.random.rand(1000, 100))
            cache.put(f'key_{i}', large_df)
            
        # Then - tylko najnowsze powinny być w cache
        assert cache.get('key_0') is None  # Evicted
        assert cache.get('key_99') is not None  # Still there
        
    def test_cache_ttl_expiration(self):
        """Test wygasania cache po czasie"""
        # Given
        cache = FeatureCache(ttl_seconds=1)
        cache.put('key', pd.DataFrame({'a': [1]}))
        
        # When
        time.sleep(2)
        
        # Then
        assert cache.get('key') is None
```

#### Testy integracyjne (Integration Tests)

##### Struktura testów integracyjnych
```
tests/integration/
├── test_mcts_pipeline/
│   ├── test_full_iteration.py
│   ├── test_session_lifecycle.py
│   └── test_checkpoint_recovery.py
├── test_feature_pipeline/
│   ├── test_end_to_end_generation.py
│   ├── test_operation_chaining.py
│   └── test_parallel_generation.py
├── test_database/
│   ├── test_repository_operations.py
│   ├── test_transactions.py
│   └── test_connection_pool.py
└── test_evaluation/
    ├── test_autogluon_integration.py
    ├── test_model_training.py
    └── test_metric_calculation.py
```

##### Przykłady testów integracyjnych

**Test pełnej iteracji MCTS**:
```python
class TestMCTSIteration:
    @pytest.fixture
    def mcts_setup(self, test_dataset):
        """Setup MCTS z test config"""
        config = {
            'mcts': {
                'exploration_weight': 1.4,
                'max_tree_depth': 3,
                'expansion_threshold': 2
            },
            'autogluon': {
                'time_limit': 10,
                'presets': 'medium_quality_faster_train'
            }
        }
        
        engine = MCTSEngine(config)
        engine.initialize(test_dataset)
        return engine
        
    def test_complete_iteration_cycle(self, mcts_setup):
        """Test pełnego cyklu iteracji MCTS"""
        # Given
        engine = mcts_setup
        initial_nodes = engine.tree.node_count()
        
        # When
        result = engine.run_iteration()
        
        # Then
        assert result.selected_node is not None
        assert result.generated_features is not None
        assert result.evaluation_score >= 0.0
        assert engine.tree.node_count() >= initial_nodes
        
    def test_selection_expansion_evaluation(self, mcts_setup):
        """Test kolejnych faz algorytmu"""
        # Given
        engine = mcts_setup
        
        # When - Selection
        selected = engine.select_node()
        assert selected.is_leaf() or selected.is_expandable()
        
        # When - Expansion
        if selected.is_expandable():
            expanded = engine.expand_node(selected)
            assert len(expanded) > 0
            selected = expanded[0]
            
        # When - Evaluation
        features = engine.generate_features(selected)
        assert features.shape[1] > 0
        
        score = engine.evaluate_features(features)
        assert 0.0 <= score <= 1.0
        
        # When - Backpropagation
        engine.backpropagate(selected, score)
        assert selected.visits == 1
        assert selected.total_reward == score
```

**Test Database Integration**:
```python
class TestDatabaseIntegration:
    @pytest.fixture
    def db_service(self, temp_db):
        """Database service z tymczasową bazą"""
        return DatabaseService(temp_db)
        
    def test_session_persistence_and_recovery(self, db_service):
        """Test zapisu i odczytu sesji"""
        # Given
        session_data = {
            'name': 'test_session',
            'config': {'test': True},
            'status': 'running'
        }
        
        # When - Create
        session_id = db_service.create_session(session_data)
        assert session_id is not None
        
        # When - Update
        db_service.update_session(session_id, {'status': 'completed'})
        
        # When - Retrieve
        retrieved = db_service.get_session(session_id)
        
        # Then
        assert retrieved['name'] == 'test_session'
        assert retrieved['status'] == 'completed'
        assert retrieved['config']['test'] is True
        
    def test_concurrent_access(self, db_service):
        """Test równoczesnego dostępu"""
        # Given
        num_threads = 10
        operations_per_thread = 100
        
        def worker(thread_id):
            for i in range(operations_per_thread):
                db_service.log_metric({
                    'thread_id': thread_id,
                    'value': i
                })
                
        # When
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Then
        total_metrics = db_service.count_metrics()
        expected = num_threads * operations_per_thread
        assert total_metrics == expected
```

#### Testy wydajnościowe (Performance Tests)

##### Benchmarki komponentów
```python
class TestPerformanceBenchmarks:
    @pytest.mark.benchmark(group="feature_generation")
    def test_statistical_features_performance(self, benchmark):
        """Benchmark generowania cech statystycznych"""
        # Given
        data = generate_large_dataset(rows=100_000, cols=50)
        operation = StatisticalAggregation()
        
        # When/Then
        result = benchmark(operation.generate, data)
        
        # Assert performance requirements
        assert benchmark.stats['mean'] < 1.0  # <1s average
        assert benchmark.stats['max'] < 2.0   # <2s worst case
        
    @pytest.mark.benchmark(group="mcts_operations")
    def test_tree_selection_performance(self, benchmark):
        """Benchmark selekcji w dużym drzewie"""
        # Given
        tree = generate_large_tree(nodes=10_000)
        
        # When/Then
        result = benchmark(tree.select_best_node)
        
        # Performance assertions
        assert benchmark.stats['mean'] < 0.01  # <10ms average
        assert benchmark.stats['stddev'] < 0.005  # Low variance
```

##### Load testing
```python
class TestLoadScenarios:
    def test_sustained_load(self, system_under_test):
        """Test długotrwałego obciążenia"""
        # Given
        duration_hours = 1
        target_rps = 10  # requests per second
        
        # When
        start_time = time.time()
        requests_made = 0
        errors = 0
        
        while (time.time() - start_time) < duration_hours * 3600:
            try:
                system_under_test.process_request()
                requests_made += 1
            except Exception:
                errors += 1
                
            # Maintain target RPS
            time.sleep(1.0 / target_rps)
            
        # Then
        error_rate = errors / requests_made
        assert error_rate < 0.01  # <1% errors
        assert requests_made >= target_rps * duration_hours * 3600 * 0.95
```

#### Testy end-to-end (E2E Tests)

##### Scenariusze użytkownika
```python
class TestUserScenarios:
    def test_kaggle_competition_workflow(self, cli_runner, sample_dataset):
        """Test pełnego workflow dla konkurencji"""
        # Step 1: Register dataset
        result = cli_runner.invoke(['manager', 'datasets', 'register',
                                  '--name', 'competition_data',
                                  '--path', sample_dataset])
        assert result.exit_code == 0
        
        # Step 2: Run exploration
        result = cli_runner.invoke(['mcts', 'run',
                                  '--config', 'config/test.yaml',
                                  '--dataset', 'competition_data',
                                  '--iterations', '10'])
        assert result.exit_code == 0
        session_id = extract_session_id(result.output)
        
        # Step 3: Wait for completion
        max_wait = 300  # 5 minutes
        start = time.time()
        while time.time() - start < max_wait:
            result = cli_runner.invoke(['mcts', 'status', session_id])
            if 'completed' in result.output:
                break
            time.sleep(10)
            
        # Step 4: Export results
        result = cli_runner.invoke(['mcts', 'export',
                                  '--session', session_id,
                                  '--format', 'python',
                                  '--output', 'features.py'])
        assert result.exit_code == 0
        assert os.path.exists('features.py')
        
        # Step 5: Validate exported code
        # Import and test the generated code
        spec = importlib.util.spec_from_file_location("features", "features.py")
        features_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(features_module)
        
        # Test that transformer works
        transformer = features_module.FeatureTransformer()
        transformed = transformer.fit_transform(sample_dataset)
        assert transformed.shape[1] > sample_dataset.shape[1]
```

### Test Data Management

#### Fixtures i Factories

```python
# conftest.py
@pytest.fixture(scope="session")
def test_data_dir():
    """Directory z test data"""
    return Path(__file__).parent / "data"
    
@pytest.fixture
def small_classification_dataset():
    """Mały dataset do klasyfikacji"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)
    
@pytest.fixture
def large_regression_dataset():
    """Duży dataset do regresji"""
    X, y = make_regression(
        n_samples=100_000,
        n_features=50,
        n_informative=30,
        noise=0.1,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)
    
class DatasetFactory:
    """Factory dla różnych typów danych"""
    
    @staticmethod
    def create_with_edge_cases():
        """Dataset z edge cases"""
        return pd.DataFrame({
            'all_nulls': [None] * 100,
            'all_same': [1] * 100,
            'high_cardinality': range(100),
            'with_infinities': [float('inf')] + [1.0] * 99,
            'with_zeros': [0] * 50 + [1] * 50,
            'text_column': ['text'] * 100,
            'date_column': pd.date_range('2020-01-01', periods=100)
        })
```

#### Mock Strategies

```python
class MockEvaluator:
    """Mock evaluator dla szybkich testów"""
    
    def __init__(self, base_score=0.5, variance=0.1):
        self.base_score = base_score
        self.variance = variance
        self.call_count = 0
        
    def evaluate(self, features):
        """Deterministyczna ewaluacja"""
        self.call_count += 1
        
        # Deterministic score based on feature count
        feature_bonus = len(features.columns) * 0.01
        noise = hash(str(features.columns.tolist())) % 100 / 1000
        
        score = self.base_score + feature_bonus + noise
        return min(max(score, 0.0), 1.0)
        
class MockFeatureGenerator:
    """Mock generator cech"""
    
    def generate(self, data, operation):
        """Generuje syntetyczne cechy"""
        n_rows = len(data)
        
        if operation == 'statistical':
            return pd.DataFrame({
                f'{operation}_mean': np.random.randn(n_rows),
                f'{operation}_std': np.random.rand(n_rows)
            })
        elif operation == 'polynomial':
            return pd.DataFrame({
                f'{operation}_squared': np.random.randn(n_rows) ** 2,
                f'{operation}_interaction': np.random.randn(n_rows)
            })
        else:
            return pd.DataFrame({
                f'{operation}_feature': np.random.randn(n_rows)
            })
```

### Test Automation Pipeline

#### CI/CD Configuration

```yaml
# .github/workflows/tests.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install UV
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
    - name: Install dependencies
      run: |
        uv pip install -r requirements.txt
        uv pip install -r requirements-test.txt
        
    - name: Run unit tests
      run: |
        pytest tests/unit \
          --cov=src \
          --cov-report=xml \
          --cov-report=term-missing \
          --junit-xml=junit.xml
          
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
          
    steps:
    - uses: actions/checkout@v4
    
    - name: Run integration tests
      run: |
        pytest tests/integration \
          --maxfail=5 \
          --timeout=300
          
  performance-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run performance benchmarks
      run: |
        pytest tests/performance \
          --benchmark-only \
          --benchmark-json=benchmark.json
          
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

#### Test Execution Strategy

```bash
#!/bin/bash
# scripts/run_tests.sh

# Fast feedback loop (for development)
run_fast_tests() {
    pytest tests/unit -x --ff --tb=short
}

# Comprehensive test suite
run_all_tests() {
    # Unit tests with coverage
    pytest tests/unit \
        --cov=src \
        --cov-report=html \
        --cov-report=term-missing \
        --cov-fail-under=80
        
    # Integration tests
    pytest tests/integration \
        --timeout=300 \
        --maxfail=3
        
    # Performance tests
    pytest tests/performance \
        --benchmark-compare \
        --benchmark-histogram
}

# Smoke tests (for deployment)
run_smoke_tests() {
    pytest tests/smoke -v --tb=short
}

# Memory leak detection
run_memory_tests() {
    pytest tests/unit tests/integration \
        --memray \
        --memray-bin-path=.memray
}
```

### Test Quality Metrics

#### Coverage Requirements
```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */migrations/*
    */__pycache__/*
    */venv/*
    
[report]
precision = 2
skip_covered = False
show_missing = True

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

#### Quality Gates
```python
# test_quality_gates.py
def test_code_coverage():
    """Ensure minimum coverage is met"""
    import coverage
    cov = coverage.Coverage()
    cov.load()
    
    total_coverage = cov.report()
    assert total_coverage >= 80.0, f"Coverage {total_coverage}% is below 80%"
    
def test_no_flaky_tests():
    """Ensure no flaky tests in suite"""
    # Run test suite multiple times
    results = []
    for _ in range(3):
        result = pytest.main(['tests/unit', '-x'])
        results.append(result)
        
    # All runs should pass
    assert all(r == 0 for r in results), "Flaky tests detected"
    
def test_performance_regression():
    """Check for performance regressions"""
    current = load_benchmark_results('benchmark_current.json')
    baseline = load_benchmark_results('benchmark_baseline.json')
    
    for test, current_time in current.items():
        baseline_time = baseline.get(test)
        if baseline_time:
            regression = (current_time - baseline_time) / baseline_time
            assert regression < 0.1, f"{test} regressed by {regression*100}%"
```

### Test Debugging Tools

#### Debug Helpers
```python
# test_helpers.py
def debug_dataframe(df, name="DataFrame"):
    """Helper to debug DataFrame w testach"""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Head:\n{df.head()}")
    print(f"Nulls:\n{df.isnull().sum()}")
    print("=" * 50)
    
def save_test_artifact(obj, test_name):
    """Save test artifacts for debugging"""
    artifact_dir = Path("test_artifacts") / test_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(artifact_dir / "data.parquet")
    elif isinstance(obj, dict):
        with open(artifact_dir / "data.json", 'w') as f:
            json.dump(obj, f, indent=2)
    else:
        with open(artifact_dir / "data.pkl", 'wb') as f:
            pickle.dump(obj, f)
```

#### Test Monitoring
```python
# Memory monitoring podczas testów
@pytest.fixture(autouse=True)
def monitor_memory():
    """Monitor memory usage podczas testu"""
    import tracemalloc
    
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()
    
    yield
    
    snapshot_end = tracemalloc.take_snapshot()
    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    
    # Log if memory usage is high
    total_diff = sum(stat.size_diff for stat in top_stats)
    if total_diff > 100 * 1024 * 1024:  # 100MB
        print(f"\nHigh memory usage detected: {total_diff / 1024 / 1024:.1f} MB")
        for stat in top_stats[:5]:
            print(stat)
```

### Continuous Testing Strategy

#### Mutation Testing
```bash
# Run mutation testing
mutmut run --paths-to-mutate=src/ --tests-dir=tests/unit/

# Generate report
mutmut html

# Quality gate: >80% mutations killed
```

#### Property-Based Testing
```python
# test_properties.py
from hypothesis import given, strategies as st

class TestFeatureProperties:
    @given(
        data=st.data(),
        n_rows=st.integers(min_value=10, max_value=1000),
        n_cols=st.integers(min_value=1, max_value=50)
    )
    def test_feature_generation_properties(self, data, n_rows, n_cols):
        """Test properties that should always hold"""
        # Generate random DataFrame
        df = pd.DataFrame(
            data.draw(st.lists(
                st.lists(st.floats(allow_nan=False, allow_infinity=False), 
                        min_size=n_cols, max_size=n_cols),
                min_size=n_rows, max_size=n_rows
            ))
        )
        
        # Apply feature generation
        generator = FeatureGenerator()
        result = generator.transform(df)
        
        # Properties that must hold
        assert len(result) == len(df)  # Same number of rows
        assert result.shape[1] >= df.shape[1]  # At least as many columns
        assert not result.isnull().all().any()  # No columns all null
        assert result.select_dtypes(include=[np.number]).shape[1] > 0  # Has numeric
```

### Test Maintenance

#### Test Review Checklist
- [ ] Test nazwa jasno opisuje co jest testowane
- [ ] Setup jest minimalny i skupiony
- [ ] Assertions są specyficzne i informatywne
- [ ] Test jest deterministyczny
- [ ] Test jest szybki (<0.1s dla unit)
- [ ] Test pokrywa edge cases
- [ ] Test ma cleanup jeśli potrzebny

#### Test Refactoring Guidelines
1. **DRY dla setup, nie dla test logic**
2. **Preferuj wiele małych testów nad jednym dużym**
3. **Używaj descriptive names zamiast komentarzy**
4. **Fixtures dla reusable setup**
5. **Parametrize dla test variants**
6. **Mark slow tests explicitly**

### Test Documentation

#### Test Plan Reviews
- **Weekly**: Review failing tests
- **Monthly**: Coverage analysis
- **Quarterly**: Test suite performance
- **Yearly**: Complete test strategy review

#### Metrics Tracking
```python
# test_metrics.py
def generate_test_report():
    """Generate comprehensive test metrics"""
    return {
        'total_tests': count_tests(),
        'coverage': get_coverage_percentage(),
        'avg_execution_time': get_average_test_time(),
        'flaky_tests': identify_flaky_tests(),
        'slow_tests': find_slow_tests(threshold=1.0),
        'test_to_code_ratio': calculate_test_ratio(),
        'mutation_score': get_mutation_testing_score()
    }
```
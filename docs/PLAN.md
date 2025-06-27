# PLAN.md - MCTS-Driven Automated Feature Engineering System

## 1. Cel Projektu
Stworzenie systemu automatycznego feature engineering dla fertilizer prediction używając:
- **MCTS (Monte Carlo Tree Search)** do inteligentnej eksploracji przestrzeni ficzerów
- **AutoGluon** jako szybkiego ewaluatora jakości zestawów ficzerów  
- **SQLite** do kompleksowego logowania eksploracji i analizy impact ficzerów

## 2. Architektura Systemu

### 2.1 Komponenty Główne
```
fertilizer_models/
├── mcts_feature_discovery/
│   ├── __init__.py
│   ├── mcts_engine.py           # Główny algorytm MCTS
│   ├── feature_space.py         # Definicje operacji na ficzerach
│   ├── autogluon_evaluator.py   # Wrapper AutoGluon dla MCTS
│   ├── discovery_db.py          # SQLite interface
│   ├── llm_generator.py         # LLM-assisted feature generation
│   └── analytics.py             # Dashboard i raporty
├── feature_discovery.db         # SQLite baza danych
├── mcts_config.yaml            # Konfiguracja systemu
└── run_feature_discovery.py    # Główny skrypt uruchomieniowy
```

### 2.2 SQLite Schema
```sql
-- Historia eksploracji MCTS
exploration_history: session_id, iteration, operation_applied, features_before/after, evaluation_score, timing

-- Katalog ficzerów z kodem Python
feature_catalog: feature_name, category, python_code, dependencies, created_by, description

-- Analiza wpływu ficzerów na metrykę
feature_impact: feature_name, baseline_score, with_feature_score, impact_delta, sample_size

-- Performance operacji
operation_performance: operation_name, success_rate, avg_improvement, effectiveness_score

-- Views: top_features, session_summary, discovery_timeline
```

### 2.3 MCTS Node Structure
```python
class FeatureNode:
    - base_features: Set[str]        # Podstawowe ficzery
    - applied_operations: List[str]   # Historia operacji
    - evaluation_score: float        # MAP@3 z AutoGluon
    - visit_count: int               # MCTS visits
    - total_reward: float            # Suma nagród
    - children: List[FeatureNode]    # Dzieci w drzewie
    - ucb1_score: float             # Upper Confidence Bound
```

## 3. Feature Space Definition

### 3.1 Kategorie Operacji
```python
FEATURE_OPERATIONS = {
    'npk_interactions': [
        'npk_ratios', 'npk_products', 'npk_harmony', 'npk_balance_indicators'
    ],
    'environmental_stress': [
        'heat_stress', 'drought_stress', 'optimal_conditions', 'climate_zones'
    ],
    'agricultural_domain': [
        'crop_nutrient_deficits', 'soil_adjustments', 'crop_soil_compatibility'
    ],
    'statistical_aggregations': [
        'groupby_soil_stats', 'groupby_crop_stats', 'rankings', 'z_scores'
    ],
    'feature_transformations': [
        'polynomial_features', 'interaction_terms', 'binning', 'log_transforms'
    ],
    'feature_selection': [
        'remove_low_importance', 'correlation_filter', 'univariate_selection'
    ]
}
```

### 3.2 Lazy Feature Generation
- Ficzery generowane on-demand podczas eksploracji
- Caching wyników w pamięci i na dysku
- Tylko używane kombinacje są obliczane

## 4. MCTS Algorithm Flow

### 4.1 Główna Pętla
```python
for iteration in range(max_iterations):
    # 1. SELECTION: UCB1 do wyboru węzła
    selected_node = mcts_tree.select_best_node()
    
    # 2. EXPANSION: Dodaj nowe operacje ficzerów  
    new_operations = feature_space.get_available_operations(selected_node)
    
    # 3. SIMULATION: AutoGluon evaluation
    for operation in new_operations:
        new_features = apply_operation(selected_node.features, operation)
        score = autogluon_evaluator.evaluate(new_features)
        
        # Log do bazy danych
        db.log_exploration_step(iteration, operation, features, score)
        
    # 4. BACKPROPAGATION: Update nagród w drzewie
    mcts_tree.backpropagate(selected_node, scores)
```

### 4.2 AutoGluon Evaluation Strategy
```python
# Faza Exploration (szybka ewaluacja)
evaluation_config_fast = {
    'time_limit': 30,
    'presets': 'medium_quality_faster_train', 
    'num_bag_folds': 2,
    'holdout_frac': 0.3
}

# Faza Exploitation (dokładniejsza ewaluacja)
evaluation_config_thorough = {
    'time_limit': 120,
    'presets': 'good_quality_faster_inference',
    'num_bag_folds': 3, 
    'holdout_frac': 0.2
}
```

## 5. Implementation Plan - Fazy

### FAZA 1: Core Infrastructure (Days 1-2)
1. ✅ Stworzenie SQLite schema i discovery_db.py
2. ✅ Implementacja podstawowego MCTS w mcts_engine.py
3. ✅ AutoGluon wrapper w autogluon_evaluator.py
4. ✅ Feature space definition w feature_space.py
5. ✅ Konfiguracja w mcts_config.yaml

### FAZA 2: MCTS Algorithm (Days 3-4)  
1. ✅ UCB1 selection algorithm
2. ✅ Node expansion z feature operations
3. ✅ Integration AutoGluon evaluation
4. ✅ Backpropagation i reward updates
5. ✅ Memory management i caching

### FAZA 3: Feature Engineering (Days 5-6)
1. ✅ Lazy feature generation system
2. ✅ Integration z feature_engineering.py (existing)
3. ✅ Feature dependency tracking  
4. ✅ Operation performance monitoring
5. ✅ Blacklist management dla złych kombinacji

### FAZA 4: LLM Integration (Days 7-8)
1. ✅ LLM-assisted feature generation w llm_generator.py
2. ✅ Prompt engineering z historical context
3. ✅ Code generation i validation
4. ✅ Integration z MCTS strategy

### FAZA 5: Analytics & Reporting (Days 9-10)
1. ✅ Dashboard w analytics.py
2. ✅ Feature impact analysis
3. ✅ Session progress tracking
4. ✅ Export best features code
5. ✅ HTML/PDF reporting

## 6. Key Configuration Parameters

### 6.1 MCTS Parameters
```yaml
mcts:
  max_iterations: 200
  exploration_weight: 1.4        # UCB1 C parameter
  max_tree_depth: 8
  expansion_threshold: 5         # Min visits before expansion
  
autogluon:
  fast_eval:
    time_limit: 30
    presets: 'medium_quality_faster_train'
  thorough_eval: 
    time_limit: 180
    presets: 'good_quality_faster_inference'
    
feature_space:
  max_features_per_node: 200     # Limit ficzerów w węźle
  min_improvement_threshold: 0.001  # Min MAP@3 improvement
  operation_timeout: 300         # Max czas na operację
```

### 6.2 Database Configuration
```yaml
database:
  path: 'feature_discovery.db'
  backup_interval: 50            # Backup co N iteracji
  max_history_size: 10000       # Max rekordów w historii
  
logging:
  level: 'INFO'
  save_feature_code: true
  track_timing: true
  export_format: 'html'
```

## 7. Expected Outcomes

### 7.1 Performance Targets
- **Automatization**: 90% reduction w manual feature engineering
- **Discovery**: Znalezienie 5-10 high-impact ficzerów (>0.01 MAP@3 improvement)
- **Efficiency**: Eksploracja 500+ kombinacji ficzerów w 8-12 godzin
- **Reproducibility**: Pełny log + kod dla każdego discovered ficzera

### 7.2 Deliverables
1. **feature_discovery.db** - Kompletna historia eksploracji
2. **best_features.py** - Auto-generated kod najlepszych ficzerów
3. **discovery_report.html** - Analytics dashboard  
4. **mcts_session_XXXXXX.json** - Session backup
5. **Updated feature_engineering.py** - Enriched z discovered ficzerami

### 7.3 Success Metrics
- MAP@3 improvement: >0.02 (target: 0.35+ najlepszy dotąd)
- Feature discovery rate: >2 valuable features per hour of search
- Code quality: All generated features pass validation tests
- System stability: <5% failed evaluations due to technical issues

## 8. Risk Mitigation

### 8.1 Technical Risks
- **Memory explosion**: Lazy loading + aggressive caching cleanup
- **AutoGluon crashes**: Timeout handling + graceful fallbacks  
- **Feature explosion**: Hard limits na liczbę ficzerów per node
- **DB corruption**: Regular backups + transaction safety

### 8.2 Performance Risks  
- **Local optima**: High exploration weight + LLM novelty injection
- **Slow convergence**: Adaptive evaluation time based on progress
- **Resource exhaustion**: Monitoring + automatic cleanup

## 9. Future Extensions

### 9.1 Advanced MCTS Features
- **Neural MCTS**: Replace rollouts z neural network predictions
- **Multi-objective**: Optimize MAP@3 + inference speed + interpretability
- **Parallel MCTS**: Multiple trees exploring different regions

### 9.2 Domain Integration
- **External data**: Weather, soil databases, crop calendars
- **Expert knowledge**: Agricultural domain rules validation
- **Ensemble methods**: Combine MCTS z genetic algorithms

### 9.3 Production Features
- **Real-time monitoring**: Live dashboard during discovery
- **API interface**: REST API dla external integration  
- **Cloud deployment**: Distributed evaluation na AWS/GCP
- **Feature store**: Integration z MLflow/Feast

---

**Timeline**: 10 days total
**Dependencies**: feature_engineering.py (existing), AutoGluon, SQLite3
**Success Criteria**: Automated discovery of features improving MAP@3 by >0.02
# System Architecture Diagrams - Final Version
## Kompletne diagramy architektury systemu

### Diagram architektury wysokopoziomowej z przepływem danych

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface Layer                            │
├─────────────────────────────┬────────────────────┬─────────────────────────┤
│        CLI Parser           │   Output Formatter  │    Interactive REPL      │
│  • Argument validation      │   • Rich tables     │  • Command completion    │
│  • Config loading           │   • Progress bars   │  • History management    │
│  • Command routing          │   • Color coding    │  • State persistence     │
└──────────────┬──────────────┴─────────┬──────────┴──────────┬──────────────┘
               │                        │                      │
┌──────────────▼────────────────────────▼──────────────────────▼──────────────┐
│                           Application Service Layer                          │
├─────────────────────────────┬────────────────────┬─────────────────────────┤
│    Session Orchestrator     │  Export Service    │   Analytics Service     │
│  • Lifecycle management     │  • Code generation │  • Report generation    │
│  • Resource allocation      │  • Multi-format    │  • Visualization        │
│  • Progress monitoring      │  • Documentation   │  • Metrics aggregation  │
└──────────────┬──────────────┴─────────┬──────────┴──────────┬──────────────┘
               │                        │                      │
┌──────────────▼────────────────────────▼──────────────────────▼──────────────┐
│                            Business Logic Layer                              │
├──────────────┬─────────────┬──────────────┬──────────────┬─────────────────┤
│ MCTS Engine  │Feature Space│  AutoML      │  Dataset     │  Config         │
│              │  Manager    │  Evaluator   │  Manager     │  Manager        │
│ • Selection  │ • Registry  │ • Training   │ • Loading    │ • Validation    │
│ • Expansion  │ • Generation│ • Scoring    │ • Caching    │ • Merging       │
│ • Evaluation │ • Validation│ • Caching    │ • Transform  │ • Precedence    │
│ • Backprop   │ • Catalog   │ • Ensemble   │ • Split      │ • Type safety   │
└──────┬───────┴──────┬──────┴───────┬──────┴───────┬──────┴────────┬───────┘
       │              │              │              │                │
┌──────▼──────────────▼──────────────▼──────────────▼────────────────▼───────┐
│                          Data Access Layer (Repository Pattern)              │
├─────────────────────────────┬────────────────────┬─────────────────────────┤
│   SessionRepository         │ FeatureRepository  │  DatasetRepository      │
│   ExplorationRepository     │ EvaluationRepository│ MetricsRepository       │
│                                                                              │
│  • CRUD operations          • Batch operations    • Transaction support     │
│  • Query builders           • Lazy loading        • Connection pooling      │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                            Database Abstraction Layer                         │
├─────────────────────────────┬────────────────────┬─────────────────────────┤
│    Connection Manager       │  Query Optimizer   │   Migration Engine      │
│  • Pool management          │  • Plan caching    │  • Version control      │
│  • Health checks            │  • Index hints     │  • Rollback support     │
│  • Retry logic              │  • Statistics      │  • Schema validation    │
└──────────────┬──────────────┴─────────┬──────────┴──────────┬──────────────┘
               │                        │                      │
┌──────────────▼────────────────────────▼──────────────────────▼──────────────┐
│                          Physical Storage Layer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                    DuckDB (Primary) / PostgreSQL (Future)                    │
│                                                                              │
│  • Columnar storage         • ACID compliance      • Parallel query         │
│  • Compression              • Point-in-time recovery• Partitioning          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component interaction diagram with data flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCTS Feature Discovery Flow                        │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────┐
     │   User   │
     └─────┬────┘
           │ mcts run --config config.yaml
           ▼
     ┌──────────────────┐       ┌─────────────────┐
     │ CLI Orchestrator │──────►│ Config Manager  │
     └────────┬─────────┘       └────────┬────────┘
              │                          │ Validated config
              ▼                          ▼
     ┌──────────────────┐       ┌─────────────────┐       ┌──────────────┐
     │ Session Manager  │◄──────┤   MCTS Engine   │──────►│Dataset Manager│
     └──────────────────┘       └────────┬────────┘       └──────┬───────┘
              │                          │                        │
              │ Session ID               │ Node to expand        │ Data
              ▼                          ▼                        ▼
     ┌──────────────────┐       ┌─────────────────┐       ┌──────────────┐
     │   DB Service     │       │ Feature Manager  │◄──────┤ Data Cache   │
     └──────────────────┘       └────────┬────────┘       └──────────────┘
                                         │
                                         │ Generated features
                                         ▼
                                ┌─────────────────┐
                                │ AutoML Evaluator│
                                └────────┬────────┘
                                         │ Score
                                         ▼
                                ┌─────────────────┐
                                │  Cache Manager  │
                                └────────┬────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
           ┌─────────────────┐                      ┌─────────────────┐
           │ Result Storage  │                      │ Tree Update     │
           └─────────────────┘                      └─────────────────┘
```

### Detailed MCTS algorithm flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MCTS Algorithm Flow                             │
└─────────────────────────────────────────────────────────────────────────────┘

Start
  │
  ▼
┌─────────────────┐
│ Initialize Tree │
│   • Root node   │
│   • Empty stats │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Main Loop (N iterations)        │
│  ┌────────────────────────────────────┐ │
│  │          Selection Phase            │ │
│  │  ┌──────────────────────────────┐  │ │
│  │  │ Current = Root               │  │ │     ┌──────────────────┐
│  │  └──────────┬───────────────────┘  │ │     │ UCB1 Calculation │
│  │             ▼                      │ │     │                  │
│  │  ┌──────────────────────────────┐  │ │     │ Score = Q + C *  │
│  │  │ While not leaf:              │  │ │◄────┤ sqrt(ln(N)/n)    │
│  │  │  • Calculate UCB1 scores     │  │ │     │                  │
│  │  │  • Select best child         │  │ │     │ Q: avg reward    │
│  │  │  • Update path               │  │ │     │ C: exploration   │
│  │  └──────────┬───────────────────┘  │ │     │ N: parent visits │
│  │             ▼                      │ │     │ n: node visits   │
│  └─────────────────────────────────────┘ │     └──────────────────┘
│                                          │
│  ┌────────────────────────────────────┐ │
│  │         Expansion Phase            │ │
│  │  ┌──────────────────────────────┐  │ │     ┌──────────────────┐
│  │  │ If visits > threshold:       │  │ │     │ Feature Space    │
│  │  │  • Get available operations  │  │ │◄────┤ • Statistical    │
│  │  │  • Apply priority weights    │  │ │     │ • Polynomial     │
│  │  │  • Create child nodes        │  │ │     │ • Binning        │
│  │  │  • Select one randomly       │  │ │     │ • Ranking        │
│  │  └──────────┬───────────────────┘  │ │     │ • Temporal       │
│  │             ▼                      │ │     │ • Text           │
│  └─────────────────────────────────────┘ │     │ • Categorical    │
│                                          │     └──────────────────┘
│  ┌────────────────────────────────────┐ │
│  │        Evaluation Phase            │ │
│  │  ┌──────────────────────────────┐  │ │     ┌──────────────────┐
│  │  │ Generate features for node   │  │ │     │ AutoGluon Models │
│  │  │  • Apply transformations     │  │ │────►│ • LightGBM       │
│  │  │  • Validate output           │  │ │     │ • XGBoost        │
│  │  │  • Check cache first         │  │ │     │ • CatBoost       │
│  │  └──────────┬───────────────────┘  │ │     │ • TabNet         │
│  │             ▼                      │ │     └────────┬─────────┘
│  │  ┌──────────────────────────────┐  │ │              │
│  │  │ Train model ensemble         │  │ │              ▼
│  │  │  • Cross-validation          │  │ │     ┌──────────────────┐
│  │  │  • Calculate metrics         │  │ │     │ Metrics:         │
│  │  │  • Normalize score [0,1]     │  │ │◄────┤ • MAP@K          │
│  │  └──────────┬───────────────────┘  │ │     │ • AUC            │
│  │             ▼                      │ │     │ • RMSE           │
│  └─────────────────────────────────────┘ │     │ • Custom         │
│                                          │     └──────────────────┘
│  ┌────────────────────────────────────┐ │
│  │     Backpropagation Phase          │ │
│  │  ┌──────────────────────────────┐  │ │
│  │  │ For each node in path:       │  │ │
│  │  │  • Increment visit count     │  │ │
│  │  │  • Add reward to total       │  │ │
│  │  │  • Update max reward         │  │ │
│  │  │  • Recalculate averages      │  │ │
│  │  └──────────┬───────────────────┘  │ │
│  │             ▼                      │ │
│  └─────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │      Maintenance Phase             │ │
│  │  • Check stopping criteria         │ │
│  │  • Save checkpoint if needed       │ │
│  │  • Prune tree if memory high       │ │
│  │  • Update global statistics        │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Export Results  │
           └─────────────────┘
```

### Database schema with relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Database Entity Relationships                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐          ┌─────────────────────┐
│      datasets       │          │      sessions       │
├─────────────────────┤          ├─────────────────────┤
│ id (PK)             │          │ id (PK)             │
│ name (UNIQUE)       │◄─────────┤ dataset_id (FK)     │
│ path                │    1:N   │ name                │
│ hash                │          │ config_json         │
│ row_count           │          │ status              │
│ column_count        │          │ created_at          │
│ size_bytes          │          │ updated_at          │
│ metadata_json       │          │ completed_at        │
│ registered_at       │          │ best_score          │
└─────────────────────┘          │ total_iterations    │
                                 └──────────┬──────────┘
                                            │ 1
                                            │
                                            │ N
                                 ┌──────────▼──────────┐
                                 │  exploration_tree   │
                                 ├─────────────────────┤
                                 │ id (PK)             │
                                 │ session_id (FK)     │◄─────┐
                                 │ parent_id (FK)───────────────┘ Self-ref
                                 │ node_path           │
                                 │ depth               │
                                 │ visits              │
                                 │ total_reward        │
                                 │ max_reward          │
                                 │ ucb_score           │
                                 │ features_json       │
                                 │ created_at          │
                                 │ last_visited        │
                                 └──────────┬──────────┘
                                            │ N
                                            │
                                            │ M
┌─────────────────────┐          ┌──────────▼──────────┐
│      features       │          │  node_features      │
├─────────────────────┤          ├─────────────────────┤
│ id (PK)             │◄─────────┤ node_id (FK)        │
│ name (UNIQUE)       │    N:M   │ feature_id (FK)     │
│ operation           │          │ position            │
│ parameters_json     │          └─────────────────────┘
│ column_source       │
│ importance_global   │
│ stability_score     │          ┌─────────────────────┐
│ usage_count         │          │    evaluations      │
│ created_at          │          ├─────────────────────┤
│ metadata_json       │          │ id (PK)             │
└─────────────────────┘          │ feature_set_hash    │
                                 │ node_id (FK)        │◄────┐
                                 │ model_type          │     │
                                 │ cv_folds            │     │
                                 │ scores_json         │     │
                                 │ duration_ms         │     │
                                 │ memory_mb           │     │
                                 │ created_at          │     │
                                 └─────────────────────┘     │
                                                              │
┌─────────────────────┐          ┌─────────────────────┐     │
│   session_metrics   │          │  feature_importance │     │
├─────────────────────┤          ├─────────────────────┤     │
│ id (PK)             │          │ id (PK)             │     │
│ session_id (FK)     │          │ evaluation_id (FK)──────────┘
│ iteration           │          │ feature_id (FK)     │
│ timestamp           │          │ importance_score    │
│ best_score          │          │ rank                │
│ nodes_explored      │          └─────────────────────┘
│ features_evaluated  │
│ memory_usage_mb     │
│ cpu_time_seconds    │
└─────────────────────┘

Indexes:
- idx_sessions_dataset (dataset_id)
- idx_sessions_status (status)
- idx_exploration_session_parent (session_id, parent_id)
- idx_exploration_best (session_id, max_reward DESC)
- idx_features_operation (operation)
- idx_features_importance (importance_global DESC)
- idx_evaluations_hash (feature_set_hash)
- idx_metrics_session_iteration (session_id, iteration)
```

### State machine for session lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Session State Machine                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   CREATED   │
                              └──────┬──────┘
                                     │ initialize()
                                     ▼
                              ┌─────────────┐
                         ┌────┤ INITIALIZING├────┐
                         │    └──────┬──────┘    │
                         │           │           │ error
                         │           │ ready     │
                         │           ▼           ▼
                         │    ┌─────────────┐  ┌──────────┐
                         │    │   RUNNING   │  │  FAILED  │
                         │    └──────┬──────┘  └──────────┘
                         │           │
                         │           ├─────────────┐
                  pause()│           │             │
                         │           │ complete()  │ interrupt()
                         ▼           ▼             ▼
                   ┌─────────┐ ┌──────────┐ ┌──────────────┐
                   │ PAUSED  │ │COMPLETED │ │ INTERRUPTED  │
                   └────┬────┘ └──────────┘ └──────┬───────┘
                        │                           │
                        │ resume()                  │ resume()
                        └───────────┬───────────────┘
                                   ▼
                            ┌─────────────┐
                            │  RESUMING   │
                            └──────┬──────┘
                                   │ restored()
                                   ▼
                            ┌─────────────┐
                            │   RUNNING   │
                            └─────────────┘

State Transitions:
- CREATED → INITIALIZING: When session starts
- INITIALIZING → RUNNING: After successful setup
- INITIALIZING → FAILED: On initialization error
- RUNNING → PAUSED: User-initiated pause
- RUNNING → COMPLETED: Normal termination
- RUNNING → INTERRUPTED: Unexpected termination
- RUNNING → FAILED: Unrecoverable error
- PAUSED → RESUMING: User-initiated resume
- INTERRUPTED → RESUMING: Auto or manual recovery
- RESUMING → RUNNING: After state restoration
- RESUMING → FAILED: If restoration fails
```

### Memory management architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Memory Management Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Total System Memory                             │
│                                   (16 GB)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────────────┐ │
│  │   System Reserved  │  │  Application Core  │  │    Feature Cache      │ │
│  │      (2 GB)        │  │      (2 GB)        │  │      (4 GB)           │ │
│  │                    │  │                    │  │                       │ │
│  │ • OS overhead      │  │ • Python runtime   │  │ • LRU cache           │ │
│  │ • Other processes  │  │ • Libraries        │  │ • TTL expiration      │ │
│  │                    │  │ • Base objects     │  │ • Compression         │ │
│  └────────────────────┘  └────────────────────┘  └───────────────────────┘ │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────────────┐ │
│  │   MCTS Tree        │  │   Active Dataset   │  │   Evaluation Buffer   │ │
│  │     (3 GB)         │  │      (3 GB)        │  │       (2 GB)          │ │
│  │                    │  │                    │  │                       │ │
│  │ • Node objects     │  │ • Training data    │  │ • Model objects       │ │
│  │ • Statistics       │  │ • Validation data  │  │ • Predictions         │ │
│  │ • Relationships    │  │ • Transformations  │  │ • Temporary arrays    │ │
│  └─────────┬──────────┘  └────────────────────┘  └───────────────────────┘ │
│            │                                                                 │
│            ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Memory Pressure Manager                       │   │
│  │                                                                      │   │
│  │  if memory_usage > 80%:           if memory_usage > 90%:           │   │
│  │    • Prune weak tree branches       • Aggressive cache eviction    │   │
│  │    • Evict old cache entries        • Reduce batch sizes           │   │
│  │    • Trigger garbage collection     • Swap to disk (emergency)     │   │
│  │    • Reduce operation parallelism   • Pause new operations         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Memory Allocation Strategy:
1. Static allocation for core components
2. Dynamic allocation with hard limits
3. Automatic rebalancing under pressure
4. Emergency swap to disk (performance penalty)
```

### Deployment architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Production Deployment Architecture                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                Load Balancer                                 │
│                           (nginx / HAProxy / ALB)                            │
└───────────────────────┬─────────────────────────┬───────────────────────────┘
                        │                         │
          ┌─────────────▼──────────┐ ┌───────────▼────────────┐
          │   API Gateway Node 1   │ │   API Gateway Node 2   │
          │      (FastAPI)         │ │      (FastAPI)         │
          │                        │ │                        │
          │ • Authentication       │ │ • Rate limiting        │
          │ • Request routing      │ │ • Response caching     │
          │ • API versioning       │ │ • Monitoring hooks     │
          └─────────────┬──────────┘ └───────────┬────────────┘
                        │                         │
┌───────────────────────┴─────────────────────────┴───────────────────────────┐
│                           Kubernetes Cluster (EKS/GKE/AKS)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          Namespace: minotaur-prod                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │   │
│  │  │  MCTS Service  │  │ Feature Service│  │ Evaluation Service │   │   │
│  │  │  Deployment    │  │  Deployment    │  │   Deployment       │   │   │
│  │  │                │  │                │  │                    │   │   │
│  │  │ Replicas: 3    │  │ Replicas: 5    │  │ Replicas: 3        │   │   │
│  │  │ CPU: 4 cores   │  │ CPU: 2 cores   │  │ CPU: 8 cores       │   │   │
│  │  │ RAM: 16 GB     │  │ RAM: 8 GB      │  │ RAM: 32 GB         │   │   │
│  │  │ GPU: Optional  │  │                │  │ GPU: 1x V100       │   │   │
│  │  └────────┬───────┘  └────────┬───────┘  └─────────┬──────────┘   │   │
│  │           │                   │                     │               │   │
│  │           └───────────────────┴─────────────────────┘               │   │
│  │                               │                                     │   │
│  │  ┌────────────────────────────▼─────────────────────────────────┐  │   │
│  │  │                    Service Mesh (Istio)                      │  │   │
│  │  │                                                              │  │   │
│  │  │  • Service discovery       • Circuit breakers               │  │   │
│  │  │  • Load balancing          • Retry policies                 │  │   │
│  │  │  • Encrypted communication • Observability                  │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │   │
│  │  │  ConfigMap     │  │    Secrets     │  │ PersistentVolumes  │   │   │
│  │  │                │  │                │  │                    │   │   │
│  │  │ • App config   │  │ • API keys     │  │ • Dataset storage  │   │   │
│  │  │ • Feature defs │  │ • DB passwords │  │ • Model cache      │   │   │
│  │  │ • Model params │  │ • Certificates │  │ • Session state    │   │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Monitoring Stack                              │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │   │
│  │  │   Prometheus   │  │    Grafana     │  │   Jaeger/Zipkin    │   │   │
│  │  │                │  │                │  │                    │   │   │
│  │  │ • Metrics      │  │ • Dashboards   │  │ • Distributed      │   │   │
│  │  │ • Alerts       │  │ • Alerting     │  │   tracing          │   │   │
│  │  │ • Recording    │  │ • Reports      │  │ • Latency analysis │   │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                ┌─────────────────────┴───────────────────────┐
                │                                             │
      ┌─────────▼──────────┐                       ┌─────────▼──────────┐
      │   Database Cluster │                       │   Object Storage   │
      │   (Aurora/Cloud SQL)│                      │   (S3/GCS/Azure)   │
      │                    │                       │                    │
      │ • Multi-AZ         │                       │ • Feature exports  │
      │ • Read replicas    │                       │ • Model artifacts  │
      │ • Auto-scaling     │                       │ • Backups          │
      └────────────────────┘                       └────────────────────┘
```

### Security architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Security Architecture                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Defense in Depth                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 1: Network Security                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • WAF (Web Application Firewall)    • DDoS Protection              │    │
│  │ • VPC with private subnets          • Network segmentation         │    │
│  │ • Security groups / Network ACLs    • VPN for admin access         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                      ▼                                       │
│  Layer 2: Authentication & Authorization                                     │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • OAuth2 / OIDC integration         • API key management           │    │
│  │ • JWT tokens with expiration        • Role-based access (RBAC)     │    │
│  │ • Multi-factor authentication       • Session management           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                      ▼                                       │
│  Layer 3: Application Security                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Input validation & sanitization   • CSRF protection              │    │
│  │ • SQL injection prevention          • XSS prevention               │    │
│  │ • Path traversal protection         • Secure headers               │    │
│  │ • Rate limiting per user/IP         • Request size limits          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                      ▼                                       │
│  Layer 4: Data Security                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Encryption at rest (AES-256)      • Column-level encryption      │    │
│  │ • Encryption in transit (TLS 1.3)   • Key rotation                 │    │
│  │ • Data masking for PII              • Secure deletion             │    │
│  │ • Backup encryption                 • Access logging               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                      ▼                                       │
│  Layer 5: Runtime Security                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Container scanning                • Resource limits              │    │
│  │ • Runtime protection (Falco)        • Syscall filtering           │    │
│  │ • Process isolation                 • Least privilege             │    │
│  │ • Security policies (OPA)           • Admission controllers       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                      ▼                                       │
│  Layer 6: Monitoring & Response                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • SIEM integration                  • Anomaly detection           │    │
│  │ • Security event logging            • Incident response plan      │    │
│  │ • Vulnerability scanning            • Penetration testing         │    │
│  │ • Compliance auditing               • Forensics capability        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Flow Security                                │
└─────────────────────────────────────────────────────────────────────────────┘

User Request
     │
     ▼
[TLS Termination]
     │
     ▼
[WAF Rules]
     │
     ▼
[Authentication]──────► [Auth Service]──────► [User Database]
     │                        │
     ▼                        ▼
[Authorization]◄────── [Permission Check]
     │
     ▼
[Rate Limiting]──────► [Redis Cache]
     │
     ▼
[Input Validation]
     │
     ▼
[Business Logic]
     │
     ├────► [Audit Logging]──────► [SIEM]
     │
     ▼
[Data Access]──────► [Encrypted Database]
     │
     ▼
[Response Filtering]
     │
     ▼
[Output Encoding]
     │
     ▼
User Response

Security Controls at Each Stage:
- Input: Validation, sanitization, size limits
- Processing: Sandboxing, resource limits, timeouts
- Storage: Encryption, access control, audit trails
- Output: Filtering, encoding, minimal disclosure
```

### Performance monitoring dashboard layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Minotaur Performance Monitoring Dashboard                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┐  ┌────────────────────────────────────┐   │
│  │     System Overview          │  │        Active Sessions             │   │
│  ├─────────────────────────────┤  ├────────────────────────────────────┤   │
│  │ CPU Usage:        ████░ 76% │  │ Running:    ████████████████  12   │   │
│  │ Memory:           ███░░ 61% │  │ Paused:     ████              3    │   │
│  │ Disk I/O:         ██░░░ 42% │  │ Completed:  ████████          8    │   │
│  │ Network:          █░░░░ 23% │  │ Failed:     ██                2    │   │
│  │                             │  │                                    │   │
│  │ Uptime: 15d 7h 23m          │  │ Total Today: 25                    │   │
│  └─────────────────────────────┘  └────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MCTS Exploration Metrics (Last 24h)               │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │  Iterations/hour  ┃                    ╱─────╲                       │   │
│  │             150   ┃                ╱───╯      ╲                      │   │
│  │             100   ┃          ╱─────╯           ╲────╮                │   │
│  │              50   ┃    ╱─────╯                      ╲───             │   │
│  │               0   ┃────┴──────┴──────┴──────┴──────┴──────┴─        │   │
│  │                      00:00   04:00   08:00   12:00   16:00   20:00   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────┐  ┌────────────────────────────────────┐   │
│  │   Feature Generation Stats   │  │      Model Training Times          │   │
│  ├─────────────────────────────┤  ├────────────────────────────────────┤   │
│  │ Total Generated:    45,678   │  │ LightGBM:    ████░░░░░░  12.3s    │   │
│  │ Unique Features:    12,345   │  │ XGBoost:     █████░░░░░  15.7s    │   │
│  │ Signal Detected:     8,901   │  │ CatBoost:    ███████░░░  21.2s    │   │
│  │ Cache Hit Rate:        87%   │  │ TabNet:      ██████████  45.8s    │   │
│  │                             │  │                                    │   │
│  │ Avg Generation Time: 0.23s  │  │ Ensemble Avg: 23.8s                │   │
│  └─────────────────────────────┘  └────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Top Features by Importance                    │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ 1. price_rolling_mean_7d              ████████████████████  0.892    │   │
│  │ 2. category_target_encoding           ███████████████████   0.854    │   │
│  │ 3. text_tfidf_cluster_5               ██████████████████    0.823    │   │
│  │ 4. quantity_x_price_log               █████████████████     0.801    │   │
│  │ 5. user_purchase_frequency_rank       ████████████████      0.756    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────┐  ┌────────────────────────────────────┐   │
│  │     Database Performance     │  │         Error Rate                 │   │
│  ├─────────────────────────────┤  ├────────────────────────────────────┤   │
│  │ Queries/sec:         1,234   │  │ Last Hour:                         │   │
│  │ Avg Latency:         12ms   │  │  Warnings:  ███░░░░░░░░  23       │   │
│  │ Connection Pool:      8/10   │  │  Errors:    █░░░░░░░░░░   5       │   │
│  │ Cache Hit Rate:       92%    │  │  Critical:  ░░░░░░░░░░░   0       │   │
│  │                             │  │                                    │   │
│  │ Slow Queries:           3    │  │ Error Rate: 0.12%                  │   │
│  └─────────────────────────────┘  └────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Error flow and recovery mechanisms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Error Detection and Recovery Flow                       │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────┐
                           │   Operation  │
                           └──────┬───────┘
                                  │
                        ┌─────────▼─────────┐
                        │   Try Execute     │
                        └─────────┬─────────┘
                                  │
                           ┌──────┴──────┐
                           │   Success?  │
                           └──────┬──────┘
                        No ───────┴─────── Yes
                        │                    │
                        ▼                    ▼
                ┌───────────────┐      ┌──────────┐
                │ Error Handler │      │ Continue │
                └───────┬───────┘      └──────────┘
                        │
                ┌───────▼────────┐
                │ Classify Error │
                └───────┬────────┘
                        │
        ┌───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Transient   │ │   Resource   │ │    Logic     │ │   Critical   │
│              │ │              │ │              │ │              │
│ • Network    │ │ • Memory     │ │ • Invalid    │ │ • Corruption │
│ • Timeout    │ │ • CPU        │ │   state      │ │ • Security   │
│ • Lock       │ │ • Disk       │ │ • Assertion  │ │ • Fatal      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Retry     │ │   Degrade    │ │   Rollback   │ │   Shutdown   │
│              │ │              │ │              │ │              │
│ Exponential  │ │ Reduce load  │ │ Restore      │ │ Save state   │
│ backoff:     │ │ Flush cache  │ │ checkpoint   │ │ Alert admin  │
│ 1s→2s→4s→8s  │ │ Smaller batch│ │ Reset state  │ │ Exit clean   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────────────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────────────────────────────────┐
│           Recovery Coordinator           │
│                                          │
│ • Log error with context                 │
│ • Update metrics                         │
│ • Notify monitoring                      │
│ • Trigger compensation                   │
└──────────────┬───────────────────────────┘
                │
         ┌──────┴──────┐
         │  Recovered? │
         └──────┬──────┘
      Yes ───────┴─────── No
      │                    │
      ▼                    ▼
┌──────────┐        ┌──────────────┐
│ Continue │        │ Escalate to  │
│ + Warning│        │ Higher Level │
└──────────┘        └──────────────┘

Recovery Strategies by Component:

MCTS Engine:
- Node corruption → Rebuild from parent
- Tree overflow → Prune weak branches
- State inconsistency → Reload from DB

Feature Generation:
- Transform failure → Skip feature
- Memory error → Reduce batch size
- Timeout → Simplify operation

Model Evaluation:
- Training failure → Fallback model
- GPU error → CPU fallback
- Score anomaly → Re-evaluate

Database:
- Connection lost → Reconnect with backoff
- Transaction fail → Retry with new conn
- Corruption → Restore from backup
```
# System Architecture Diagrams v2
## Szczegółowe diagramy architektury systemu

### Diagram architektury wysokopoziomowej

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Warstwa Prezentacji                          │
├─────────────────────────┬───────────────────────────────────────────┤
│    CLI Orchestrator     │           System Manager CLI              │
│  (Feature Discovery)    │    (Data, Sessions, Analytics)            │
└────────────┬───────────┴──────────────┬────────────────────────────┘
             │                          │
┌────────────▼───────────────────────────▼────────────────────────────┐
│                        Warstwa Aplikacji                            │
├─────────────────────────┬───────────────────────────────────────────┤
│   Session Orchestrator  │     Configuration Manager                 │
│   Export Services       │     Analytics Generator                   │
└────────────┬───────────┴──────────────┬────────────────────────────┘
             │                          │
┌────────────▼───────────────────────────▼────────────────────────────┐
│                         Warstwa Domeny                              │
├──────────────┬─────────────┬──────────────┬────────────────────────┤
│ MCTS Engine  │Feature Space│  AutoML      │   Dataset              │
│              │  Manager    │  Evaluator   │   Manager              │
└──────┬───────┴──────┬──────┴───────┬──────┴────────┬───────────────┘
       │              │              │               │
┌──────▼──────────────▼──────────────▼───────────────▼───────────────┐
│                    Warstwa Infrastruktury                          │
├─────────────────────────┬───────────────────────────────────────────┤
│   Repository Layer      │        Database Abstraction               │
│   (Session, Feature,    │        (Connection Pool,                  │
│    Dataset, Metrics)    │         Query Builder)                    │
└────────────┬───────────┴──────────────┬────────────────────────────┘
             │                          │
┌────────────▼───────────────────────────▼────────────────────────────┐
│                    Warstwa Persystencji                             │
├─────────────────────────────────────────────────────────────────────┤
│              Analytical Database (Column-Store)                     │
│                    with ACID guarantees                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagram komponentów i ich interakcji

```
┌─────────────────────────────┐
│      MCTS Engine            │
│  ┌─────────────────────┐    │
│  │ Selection Strategy  │    │
│  │    (UCB1)          │    │
│  └──────────┬─────────┘    │
│  ┌──────────▼─────────┐    │
│  │ Tree Manager       │    │
│  │ - Node creation    │    │
│  │ - Tree traversal   │    │
│  │ - Memory pruning   │    │
│  └──────────┬─────────┘    │
│  ┌──────────▼─────────┐    │
│  │ State Persistence  │    │
│  │ - Checkpointing    │    │
│  │ - Recovery         │    │
│  └────────────────────┘    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│    Feature Space Manager    │     │    AutoML Evaluator         │
│  ┌─────────────────────┐    │     │  ┌─────────────────────┐    │
│  │ Operation Registry  │    │     │  │ Model Trainer      │    │
│  │ - Generic ops      │    │◄────►│  │ - Ensemble models  │    │
│  │ - Domain ops       │    │     │  │ - Hyperparameter   │    │
│  └──────────┬─────────┘    │     │  │   optimization     │    │
│  ┌──────────▼─────────┐    │     │  └──────────┬─────────┘    │
│  │ Feature Generator  │    │     │  ┌──────────▼─────────┐    │
│  │ - Transform data   │    │     │  │ Result Cache       │    │
│  │ - Signal detection │    │     │  │ - Hash-based       │    │
│  └──────────┬─────────┘    │     │  │ - TTL management   │    │
│  ┌──────────▼─────────┐    │     │  └────────────────────┘    │
│  │ Feature Catalog    │    │     └─────────────────────────────┘
│  │ - Registration     │    │
│  │ - Metadata         │    │
│  └────────────────────┘    │
└─────────────────────────────┘
```

### Diagram przepływu danych

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   Raw Data  │────►│ Data Manager │────►│ Cached Data   │
│  (CSV/Parq) │     │ - Validation │     │  (Database)   │
└─────────────┘     │ - Type opt. │     └───────┬───────┘
                    └──────────────┘             │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     MCTS Exploration Loop                   │
│                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   Select    │────►│   Expand     │────►│  Generate   │ │
│  │   Node      │     │   Tree       │     │  Features   │ │
│  └─────────────┘     └──────────────┘     └──────┬──────┘ │
│         ▲                                         │        │
│         │                                         ▼        │
│  ┌──────┴──────┐     ┌──────────────┐     ┌─────────────┐ │
│  │ Backpropagate│◄────│   Update     │◄────│  Evaluate   │ │
│  │   Results   │     │   Scores     │     │  Quality    │ │
│  └─────────────┘     └──────────────┘     └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │ Persist Results   │
                    │ - Tree state      │
                    │ - Feature catalog │
                    │ - Metrics         │
                    └───────────────────┘
```

### Diagram sekwencji dla sesji eksploracji

```
User        CLI         Session      MCTS        Feature     AutoML      Database
 │           │          Manager      Engine      Space       Evaluator
 │           │            │           │           │           │           │
 │  start    │            │           │           │           │           │
 ├──────────►│            │           │           │           │           │
 │           ├──config───►│           │           │           │           │
 │           │            ├──init────►│           │           │           │
 │           │            │           ├──load────►│           │           │
 │           │            │           │           ├──register─────────────►│
 │           │            │           │           │           │           │
 │           │            │       ┌───┴───┐       │           │           │
 │           │            │       │ Loop  │       │           │           │
 │           │            │       │       │       │           │           │
 │           │            │       ├─select┤       │           │           │
 │           │            │       │       ├─expand►           │           │
 │           │            │       │       │◄─features─────────┤           │
 │           │            │       │       │       │           │           │
 │           │            │       │       ├─evaluate──────────►           │
 │           │            │       │       │       │           ├─train────►│
 │           │            │       │       │       │           │◄──score───┤
 │           │            │       │       │◄──────result──────┤           │
 │           │            │       │       │       │           │           │
 │           │            │       │       ├─update────────────────────────►│
 │           │            │       │       │       │           │           │
 │           │            │       └───┬───┘       │           │           │
 │           │            │           │           │           │           │
 │           │            │◄─complete─┤           │           │           │
 │           │◄──results──┤           │           │           │           │
 │◄──report──┤            │           │           │           │           │
 │           │            │           │           │           │           │
```

### Diagram deploymentu

```
┌─────────────────────────────────────────────────────────────────┐
│                      Production Server                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │   Application   │  │   Database      │  │   File System  │ │
│  │   Container     │  │   Container     │  │   Volume       │ │
│  │                 │  │                 │  │                │ │
│  │ - Python runtime│  │ - Column DB     │  │ - Config files │ │
│  │ - Dependencies  │  │ - Persistent    │  │ - Datasets     │ │
│  │ - Application   │  │   storage       │  │ - Outputs      │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬───────┘ │
│           │                    │                     │         │
│           └────────────────────┴─────────────────────┘         │
│                            Network Bridge                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         CI/CD Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│   Source Code → Build → Test → Package → Deploy → Monitor      │
└─────────────────────────────────────────────────────────────────┘
```

### Diagram modelu danych

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Sessions     │     │   Exploration   │     │    Features     │
├─────────────────┤     │    History      │     ├─────────────────┤
│ id (PK)         │     ├─────────────────┤     │ id (PK)         │
│ name            │     │ id (PK)         │     │ name            │
│ config_json     │◄────┤ session_id (FK) │     │ operation       │
│ status          │     │ iteration       │     │ params_json     │
│ created_at      │     │ node_path       │     │ importance      │
│ updated_at      │     │ features_json   │◄────┤ created_at      │
└─────────────────┘     │ score           │     └─────────────────┘
                        │ timestamp        │
                        └─────────────────┘
                                ▲
┌─────────────────┐             │              ┌─────────────────┐
│    Datasets     │             │              │   Evaluations   │
├─────────────────┤             │              ├─────────────────┤
│ id (PK)         │             │              │ id (PK)         │
│ name            │             └──────────────┤ feature_set_hash│
│ path            │                            │ score           │
│ hash            │                            │ metric          │
│ metadata_json   │                            │ model_type      │
│ registered_at   │                            │ timestamp       │
└─────────────────┘                            └─────────────────┘
```

### Diagram bezpieczeństwa

```
┌─────────────────────────────────────────────────────────────────┐
│                     Security Layers                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │Input         │    │Access        │    │Data          │    │
│  │Validation    │    │Control       │    │Encryption    │    │
│  │              │    │              │    │              │    │
│  │- Path sanitize│   │- Dataset perms│   │- At rest     │    │
│  │- SQL injection│   │- User auth    │   │- In transit  │    │
│  │- Type checking│   │- Audit trail  │   │- Key mgmt    │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │Error         │    │Resource      │    │Monitoring    │    │
│  │Handling      │    │Limits        │    │& Logging     │    │
│  │              │    │              │    │              │    │
│  │- No info leak│    │- Memory caps  │   │- Security events│  │
│  │- Graceful fail│   │- CPU quotas   │   │- Anomaly detect│   │
│  │- Recovery     │   │- Disk limits  │   │- Compliance    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```
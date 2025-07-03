# UX/UI Design Document v2
## Szczegółowy projekt interfejsu użytkownika

### Filozofia UX
System priorytetyzuje produktywność doświadczonych użytkowników przy zachowaniu przystępności dla początkujących. Interfejs CLI został zaprojektowany według zasad:
- **Clarity over cleverness**: Jasne komunikaty zamiast skrótów
- **Progressive disclosure**: Podstawowe opcje widoczne, zaawansowane dostępne
- **Fail gracefully**: Pomocne komunikaty błędów z sugestiami rozwiązań
- **Respect user time**: Szybkie odpowiedzi, progress feedback

### Architektura interfejsu

#### Główne komponenty CLI
1. **Primary orchestrator** (`mcts`): Uruchamianie i zarządzanie sesjami odkrywania
2. **System manager** (`manager`): Administracja danymi, analiza wyników
3. **Quick tools**: Jednorazowe operacje (validate, export, report)

#### Hierarchia komend
```
mcts
├── run          # Start new session
├── resume       # Continue existing session
├── list         # Show sessions
├── status       # Current session info
└── export       # Export results

manager
├── datasets     # Dataset operations
│   ├── register
│   ├── list
│   ├── info
│   └── remove
├── sessions     # Session management
│   ├── list
│   ├── compare
│   ├── analyze
│   └── cleanup
├── features     # Feature catalog
│   ├── list
│   ├── search
│   ├── top
│   └── export
└── analytics    # Reports and insights
    ├── summary
    ├── trends
    ├── compare
    └── export
```

### Detailed Command Specifications

#### Session Discovery Commands

**Starting new session:**
```bash
mcts run --config config.yaml [options]

Options:
  --name TEXT           Session name (auto-generated if not provided)
  --dataset TEXT        Registered dataset name
  --iterations INT      Max iterations (default: from config)
  --time-limit HOURS    Max runtime in hours
  --test-mode          Use mock evaluator for testing
  --gpu/--no-gpu       Enable/disable GPU acceleration
  --verbose            Detailed logging output
  --quiet              Minimal output
```

**Resuming session:**
```bash
mcts resume [SESSION_ID] [options]

Options:
  --last               Resume most recent session
  --additional INT     Run N more iterations
  --extend HOURS       Extend time limit by N hours
  --checkpoint         Force checkpoint before resuming
```

**Session information:**
```bash
mcts list [options]

Options:
  --limit INT          Show last N sessions (default: 10)
  --active            Only show active sessions
  --date-from DATE    Filter by start date
  --sort-by FIELD     Sort by: date|score|iterations
  --format FORMAT     Output format: table|json|csv
```

#### Data Management Commands

**Dataset registration:**
```bash
manager datasets register [options]

Options:
  --name TEXT         Dataset identifier (required)
  --path PATH         Path to data file (required)
  --auto             Auto-detect file type and columns
  --description TEXT  Dataset description
  --target TEXT      Target column name
  --test-path PATH   Optional test set path
  --validate         Validate data before registration
  --force            Overwrite if exists
```

**Dataset operations:**
```bash
manager datasets list [--format table|json]
manager datasets info DATASET_NAME [--detailed]
manager datasets validate DATASET_NAME
manager datasets remove DATASET_NAME [--force]
manager datasets search KEYWORD
```

#### Feature Analysis Commands

**Feature exploration:**
```bash
manager features list [options]

Options:
  --dataset TEXT      Filter by dataset
  --session TEXT      Filter by session
  --top N            Show top N by importance
  --category TEXT    Filter by category (statistical|polynomial|...)
  --min-score FLOAT  Minimum importance score
```

**Feature search:**
```bash
manager features search "PATTERN" [options]

Options:
  --regex            Use regex pattern
  --in-name          Search in feature names
  --in-description   Search in descriptions
  --case-sensitive   Case sensitive search
```

#### Analytics Commands

**Performance analytics:**
```bash
manager analytics summary [SESSION_ID]
manager analytics trends --last-days 30
manager analytics compare SESSION1 SESSION2
manager analytics export --format html --output report.html
```

### User Interaction Patterns

#### Progress Indication
```
Starting MCTS exploration...
Dataset: playground-series-s5e6 (50000 rows × 25 columns)
Config: config/mcts_config_s5e6.yaml

Iteration 15/100 [████████████████░░░░░░░░░░░░░░░░] 15% | ETA: 2h 15m
├─ Current best score: 0.3342 (MAP@3)
├─ Features evaluated: 234
├─ Nodes explored: 89
└─ Memory usage: 1.2 GB / 16.0 GB

Latest discovery: "price_std_by_category_rank" (score: 0.3156)
```

#### Interactive Prompts
```
Dataset validation failed. Issues found:
- Missing values in column 'age': 1,234 rows (2.5%)
- Suspicious values in 'price': 5 negative values
- Duplicate rows: 12 exact duplicates found

How would you like to proceed?
1) Fix automatically (fill missing, remove negatives, drop duplicates)
2) Fix interactively (decide per issue)
3) Continue anyway (not recommended)
4) Cancel operation

Your choice [1-4]: _
```

#### Error Messages
```
Error: Could not resume session 'mcts_20250629_142350'

Possible reasons:
- Session file corrupted or incomplete
- Database connection failed
- Incompatible version (session v1.2, current v1.3)

Suggestions:
- Try: mcts list --active  (to see available sessions)
- Try: mcts resume --last  (to resume most recent)
- Try: manager sessions repair SESSION_ID

For more help: mcts resume --help
```

### Visual Design Elements

#### Table Formatting
```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Session ID        ┃ Status  ┃ Best Score ┃ Features ┃ Runtime     ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
│ mcts_20250702_... │ RUNNING │ 0.3342     │ 234      │ 2h 15m      │
│ mcts_20250701_... │ DONE    │ 0.3298     │ 567      │ 5h 32m      │
│ mcts_20250630_... │ FAILED  │ 0.3156     │ 123      │ 1h 05m      │
└───────────────────┴─────────┴────────────┴──────────┴─────────────┘
```

#### Color Coding
- **Green**: Success, improvements, positive outcomes
- **Yellow**: Warnings, attention needed, suboptimal
- **Red**: Errors, failures, critical issues
- **Blue**: Information, hints, suggestions
- **Cyan**: Headers, categories, groupings
- **Magenta**: Highlights, important values

#### Tree Visualization
```
MCTS Exploration Tree (depth: 4, nodes: 89)
│
├─● price_features (0.3342) ✓ BEST
│  ├─○ price_mean_by_category (0.3298)
│  ├─● price_std_by_category_rank (0.3156)
│  │  └─○ price_cv_by_region (0.3089)
│  └─○ price_log_transform (0.2967)
│
├─◐ temporal_features (0.3234) 
│  ├─● day_of_week_encoding (0.3189)
│  └─○ month_seasonality (0.3045)
│
└─○ interaction_features (0.3123)
   └─○ price_x_quantity (0.3089)

Legend: ● Explored  ◐ Partial  ○ Not explored  ✓ Best path
```

### Accessibility Features

#### Keyboard Navigation
- **Tab completion**: Commands, options, file paths
- **History**: Up/down arrows for command history
- **Shortcuts**: Ctrl+C (interrupt), Ctrl+D (exit), Ctrl+L (clear)
- **Search**: Ctrl+R for reverse history search

#### Output Control
```bash
# Verbosity levels
mcts run --quiet          # Minimal output
mcts run                  # Standard output  
mcts run --verbose        # Detailed output
mcts run --debug          # Debug logging

# Output formats
manager features list --format table  # Human readable
manager features list --format json   # Machine readable
manager features list --format csv    # Spreadsheet ready
```

#### Help System
```bash
# General help
mcts --help
manager --help

# Command help
mcts run --help
manager datasets register --help

# Interactive help
mcts> help              # In REPL mode
mcts> help run          # Specific command
mcts> examples          # Show examples
```

### Workflow Examples

#### Typical Data Scientist Workflow
```bash
# 1. Setup dataset
manager datasets register \
  --name sales_2025 \
  --path data/train.csv \
  --target revenue \
  --auto

# 2. Configure and run
mcts run \
  --config configs/quick_test.yaml \
  --dataset sales_2025 \
  --name "initial_exploration"

# 3. Monitor progress
watch mcts status

# 4. Analyze results
manager features top --session initial_exploration
manager analytics summary

# 5. Export best features
mcts export \
  --session initial_exploration \
  --top 20 \
  --format python \
  --output best_features.py
```

#### Production Pipeline Integration
```bash
# Automated script
#!/bin/bash
SESSION=$(mcts run \
  --config prod_config.yaml \
  --iterations 1000 \
  --quiet \
  --format json | jq -r '.session_id')

# Wait for completion
while [ $(mcts status $SESSION --format json | jq -r '.status') == "RUNNING" ]; do
  sleep 60
done

# Export results
mcts export --session $SESSION --format python --output features.py
```

### Error Handling and Recovery

#### Graceful Interruption
```
^C
Interrupt received. Saving progress...
✓ Checkpoint saved (iteration 45/100)
✓ Database synchronized
✓ Temporary files cleaned

Session 'mcts_20250702_142350' can be resumed with:
  mcts resume mcts_20250702_142350

Goodbye!
```

#### Automatic Recovery
```
Warning: Previous session terminated unexpectedly
Found checkpoint at iteration 45/100

Would you like to:
1) Resume from checkpoint (recommended)
2) Start fresh
3) View session details

Your choice [1-3]: _
```

### Future UI Enhancements

#### Web Dashboard (Phase 2)
- Real-time exploration visualization
- Interactive feature importance plots
- Drag-and-drop dataset upload
- Collaborative session sharing

#### REST API (Phase 2)
- Programmatic access to all CLI functions
- WebSocket support for live updates
- OpenAPI specification
- Client SDKs (Python, R, Julia)

#### Jupyter Integration (Phase 3)
- Magic commands for notebook users
- Interactive widgets for configuration
- Inline visualizations
- Export to notebook cells
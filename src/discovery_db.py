"""
SQLite Database Interface for MCTS Feature Discovery

Comprehensive logging and analytics for feature exploration sessions.
Supports session management, feature impact analysis, and performance tracking.
"""

import sqlite3
import json
import uuid
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureDiscoveryDB:
    """SQLite database interface for MCTS feature discovery logging and analytics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize database with configuration parameters."""
        self.config = config
        self.db_config = config['database']
        self.db_path = self.db_config['path']
        self.backup_path = self.db_config['backup_path']
        self.session_id = self._get_or_create_session_id()
        
        # Ensure backup directory exists
        Path(self.backup_path).mkdir(parents=True, exist_ok=True)
        
        self.init_database()
        self._migrate_database()
        self._setup_cleanup_scheduler()
        
        logger.info(f"Initialized FeatureDiscoveryDB with session_id: {self.session_id}")
    
    def _get_or_create_session_id(self) -> str:
        """Get existing session ID or create new one based on session mode."""
        session_mode = self.config['session']['mode']
        
        if session_mode == 'new':
            return str(uuid.uuid4())
        elif session_mode in ['continue', 'resume_best']:
            # Check if specific session ID was provided
            resume_session_id = self.config['session'].get('resume_session_id')
            if resume_session_id:
                # Validate that the session exists in database
                if os.path.exists(self.db_path):
                    try:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT session_id FROM sessions 
                                WHERE session_id = ? LIMIT 1
                            """, (resume_session_id,))
                            result = cursor.fetchone()
                            if result:
                                logger.info(f"Resuming specific session: {resume_session_id[:8]}...")
                                return resume_session_id
                            else:
                                logger.warning(f"Specified session not found: {resume_session_id[:8]}...")
                    except sqlite3.Error as e:
                        logger.warning(f"Could not validate session: {e}")
            
            # Try to get last session ID if no specific session provided
            if os.path.exists(self.db_path):
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT session_id FROM sessions 
                            ORDER BY start_time DESC LIMIT 1
                        """)
                        result = cursor.fetchone()
                        if result:
                            logger.info(f"Resuming most recent session: {result[0][:8]}...")
                            return result[0]
                except sqlite3.Error as e:
                    logger.warning(f"Could not resume session: {e}")
            
            # Fallback to new session
            logger.info("No existing session found, starting new session")
            return str(uuid.uuid4())
        else:
            raise ValueError(f"Unknown session mode: {session_mode}")
    
    def init_database(self):
        """Initialize database schema with all tables, indexes, and views."""
        schema_sql = """
        -- Main exploration history table
        CREATE TABLE IF NOT EXISTS exploration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            parent_node_id INTEGER,
            operation_applied TEXT NOT NULL,
            features_before TEXT NOT NULL,  -- JSON list of features before operation
            features_after TEXT NOT NULL,   -- JSON list of features after operation
            evaluation_score REAL NOT NULL, -- MAP@3 score
            evaluation_time REAL NOT NULL,  -- Time in seconds
            autogluon_config TEXT,          -- JSON config used
            mcts_ucb1_score REAL,
            node_visits INTEGER DEFAULT 1,
            is_best_so_far BOOLEAN DEFAULT FALSE,
            memory_usage_mb REAL,
            notes TEXT
        );
        
        -- Feature catalog with Python code
        CREATE TABLE IF NOT EXISTS feature_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT UNIQUE NOT NULL,
            feature_category TEXT NOT NULL,
            python_code TEXT NOT NULL,
            dependencies TEXT,              -- JSON list of required features
            description TEXT,
            created_by TEXT DEFAULT 'mcts', -- 'mcts', 'llm', 'manual'
            creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            computational_cost REAL DEFAULT 1.0,
            data_type TEXT DEFAULT 'float64'
        );
        
        -- Feature impact analysis
        CREATE TABLE IF NOT EXISTS feature_impact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT NOT NULL,
            baseline_score REAL NOT NULL,
            with_feature_score REAL NOT NULL,
            impact_delta REAL NOT NULL,
            impact_percentage REAL NOT NULL,
            evaluation_context TEXT,        -- JSON: other features in set
            sample_size INTEGER DEFAULT 1,
            confidence_interval TEXT,       -- JSON: [lower, upper] 95% CI
            statistical_significance REAL, -- p-value if available
            first_discovered DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_evaluated DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT NOT NULL
        );
        
        -- Operation performance tracking
        CREATE TABLE IF NOT EXISTS operation_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_name TEXT NOT NULL,
            operation_category TEXT,
            total_applications INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            avg_improvement REAL DEFAULT 0.0,
            best_improvement REAL DEFAULT 0.0,
            worst_result REAL DEFAULT 0.0,
            avg_execution_time REAL DEFAULT 0.0,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
            effectiveness_score REAL DEFAULT 0.0,
            session_id TEXT NOT NULL
        );
        
        -- Session metadata
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            session_name TEXT,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            total_iterations INTEGER DEFAULT 0,
            best_score REAL DEFAULT 0.0,
            config_snapshot TEXT,           -- JSON of config used
            status TEXT DEFAULT 'active',   -- 'active', 'completed', 'interrupted'
            strategy TEXT DEFAULT 'default',  -- MCTS strategy used
            is_test_mode BOOLEAN DEFAULT FALSE,  -- Test mode flag
            notes TEXT
        );
        
        -- System performance logs
        CREATE TABLE IF NOT EXISTS system_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            memory_usage_mb REAL,
            cpu_usage_percent REAL,
            disk_usage_mb REAL,
            gpu_memory_mb REAL,
            active_nodes INTEGER,
            evaluation_queue_size INTEGER
        );
        """
        
        # Indexes for performance
        indexes_sql = """
        CREATE INDEX IF NOT EXISTS idx_exploration_session ON exploration_history(session_id);
        CREATE INDEX IF NOT EXISTS idx_exploration_score ON exploration_history(evaluation_score DESC);
        CREATE INDEX IF NOT EXISTS idx_exploration_timestamp ON exploration_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_exploration_iteration ON exploration_history(session_id, iteration);
        CREATE INDEX IF NOT EXISTS idx_feature_name ON feature_catalog(feature_name);
        CREATE INDEX IF NOT EXISTS idx_feature_category ON feature_catalog(feature_category);
        CREATE INDEX IF NOT EXISTS idx_impact_delta ON feature_impact(impact_delta DESC);
        CREATE INDEX IF NOT EXISTS idx_impact_session ON feature_impact(session_id);
        CREATE INDEX IF NOT EXISTS idx_operation_effectiveness ON operation_performance(effectiveness_score DESC);
        CREATE INDEX IF NOT EXISTS idx_operation_session ON operation_performance(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time);
        """
        
        # Views for analytics
        views_sql = """
        -- Top performing features across all sessions
        CREATE VIEW IF NOT EXISTS top_features AS
        SELECT 
            fc.feature_name,
            fc.feature_category,
            fi.impact_delta,
            fi.impact_percentage,
            fi.with_feature_score,
            fi.sample_size,
            fc.python_code,
            fc.computational_cost,
            fi.session_id
        FROM feature_catalog fc
        JOIN feature_impact fi ON fc.feature_name = fi.feature_name
        WHERE fi.impact_delta > 0
        ORDER BY fi.impact_delta DESC;
        
        -- Session summary statistics
        CREATE VIEW IF NOT EXISTS session_summary AS
        SELECT 
            eh.session_id,
            s.session_name,
            s.start_time,
            s.end_time,
            COUNT(*) as total_iterations,
            MIN(eh.evaluation_score) as min_score,
            MAX(eh.evaluation_score) as max_score,
            MAX(eh.evaluation_score) - MIN(eh.evaluation_score) as improvement,
            AVG(eh.evaluation_time) as avg_eval_time,
            SUM(eh.evaluation_time) as total_eval_time,
            s.status
        FROM exploration_history eh
        JOIN sessions s ON eh.session_id = s.session_id
        GROUP BY eh.session_id;
        
        -- Feature discovery timeline
        CREATE VIEW IF NOT EXISTS discovery_timeline AS
        SELECT 
            DATE(eh.timestamp) as discovery_date,
            eh.session_id,
            eh.operation_applied,
            eh.evaluation_score,
            fi.impact_delta,
            ROW_NUMBER() OVER (PARTITION BY eh.session_id ORDER BY eh.timestamp) as discovery_order
        FROM exploration_history eh
        LEFT JOIN feature_impact fi ON eh.operation_applied = fi.feature_name
        WHERE eh.is_best_so_far = TRUE
        ORDER BY eh.timestamp;
        
        -- Operation effectiveness ranking
        CREATE VIEW IF NOT EXISTS operation_ranking AS
        SELECT 
            operation_name,
            operation_category,
            total_applications,
            success_count,
            ROUND(100.0 * success_count / total_applications, 2) as success_rate,
            avg_improvement,
            best_improvement,
            effectiveness_score,
            COUNT(DISTINCT session_id) as used_in_sessions
        FROM operation_performance
        WHERE total_applications > 0
        GROUP BY operation_name
        ORDER BY effectiveness_score DESC;
        """
        
        with sqlite3.connect(self.db_path) as conn:
            # Execute all schema creation
            conn.executescript(schema_sql)
            conn.executescript(indexes_sql)
            conn.executescript(views_sql)
            
            # Set database configuration
            conn.execute(f"PRAGMA synchronous = {self.db_config['sync_mode']}")
            conn.execute(f"PRAGMA journal_mode = {self.db_config['journal_mode']}")
            
            # Register current session
            self._register_session(conn)
        
        logger.info("Database schema initialized successfully")
    
    def _migrate_database(self):
        """Apply database migrations for schema updates."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if strategy column exists in sessions table
                cursor.execute("PRAGMA table_info(sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'strategy' not in columns:
                    logger.info("Adding 'strategy' column to sessions table...")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN strategy TEXT DEFAULT 'default'")
                    conn.commit()
                    logger.info("Migration completed: added strategy column")
                
                if 'is_test_mode' not in columns:
                    logger.info("Adding 'is_test_mode' column to sessions table...")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN is_test_mode BOOLEAN DEFAULT FALSE")
                    conn.commit()
                    logger.info("Migration completed: added is_test_mode column")
                
        except sqlite3.Error as e:
            logger.warning(f"Database migration failed: {e}")
    
    def _register_session(self, conn: sqlite3.Connection):
        """Register current session in sessions table."""
        session_name = self.config['session'].get('session_name')
        if not session_name:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Detect test mode
        is_test_mode = self.config.get('testing', {}).get('use_mock_evaluator', False)
        
        conn.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, session_name, config_snapshot, status, is_test_mode
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            self.session_id,
            session_name, 
            json.dumps(self.config),
            'active',
            is_test_mode
        ))
        
        logger.info(f"Registered session: {session_name}")
    
    def log_exploration_step(self, 
                           iteration: int,
                           operation: str,
                           features_before: List[str],
                           features_after: List[str],
                           score: float,
                           eval_time: float,
                           autogluon_config: Dict,
                           ucb1_score: float = None,
                           parent_node_id: int = None,
                           memory_usage_mb: float = None) -> int:
        """Log a single MCTS exploration step."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if this is the best score so far in this session
            cursor.execute("""
                SELECT MAX(evaluation_score) FROM exploration_history 
                WHERE session_id = ?
            """, (self.session_id,))
            
            current_best = cursor.fetchone()[0] or 0.0
            is_best = score > current_best
            
            # Log the exploration step
            cursor.execute("""
                INSERT INTO exploration_history (
                    session_id, iteration, parent_node_id, operation_applied,
                    features_before, features_after, evaluation_score,
                    evaluation_time, autogluon_config, mcts_ucb1_score,
                    is_best_so_far, memory_usage_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id, iteration, parent_node_id, operation,
                json.dumps(features_before), json.dumps(features_after),
                score, eval_time, json.dumps(autogluon_config),
                ucb1_score, is_best, memory_usage_mb
            ))
            
            # Update session statistics
            cursor.execute("""
                UPDATE sessions SET 
                    total_iterations = ?,
                    best_score = MAX(best_score, ?)
                WHERE session_id = ?
            """, (iteration, score, self.session_id))
            
            step_id = cursor.lastrowid
            
            # Trigger backup if needed
            if iteration % self.db_config['backup_interval'] == 0:
                self._create_backup()
                
        logger.debug(f"Logged exploration step {iteration}: {operation} -> {score:.5f}")
        return step_id
    
    def register_feature(self,
                        name: str,
                        category: str,
                        python_code: str,
                        dependencies: List[str] = None,
                        description: str = "",
                        created_by: str = "mcts",
                        computational_cost: float = 1.0) -> int:
        """Register a new feature in the catalog."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO feature_catalog (
                    feature_name, feature_category, python_code,
                    dependencies, description, created_by, computational_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name, category, python_code,
                json.dumps(dependencies or []), description, created_by, computational_cost
            ))
            
            feature_id = cursor.lastrowid
            
        logger.debug(f"Registered feature: {name} ({category})")
        return feature_id
    
    def update_feature_impact(self,
                            feature_name: str,
                            baseline_score: float,
                            with_feature_score: float,
                            context_features: List[str] = None) -> None:
        """Update impact analysis for a feature."""
        
        impact_delta = with_feature_score - baseline_score
        impact_percentage = (impact_delta / baseline_score) * 100 if baseline_score > 0 else 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if record exists for this feature in current session
            cursor.execute("""
                SELECT id, sample_size, impact_delta FROM feature_impact 
                WHERE feature_name = ? AND session_id = ?
            """, (feature_name, self.session_id))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record with running average
                record_id, old_sample_size, old_impact = existing
                new_sample_size = old_sample_size + 1
                
                # Running average of impact
                new_avg_impact = ((old_impact * old_sample_size) + impact_delta) / new_sample_size
                new_avg_percentage = (new_avg_impact / baseline_score) * 100 if baseline_score > 0 else 0
                
                cursor.execute("""
                    UPDATE feature_impact SET
                        baseline_score = ?,
                        with_feature_score = ?,
                        impact_delta = ?,
                        impact_percentage = ?,
                        evaluation_context = ?,
                        sample_size = ?,
                        last_evaluated = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    baseline_score, with_feature_score, new_avg_impact,
                    new_avg_percentage, json.dumps(context_features or []),
                    new_sample_size, record_id
                ))
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO feature_impact (
                        feature_name, baseline_score, with_feature_score,
                        impact_delta, impact_percentage, evaluation_context,
                        sample_size, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_name, baseline_score, with_feature_score,
                    impact_delta, impact_percentage, 
                    json.dumps(context_features or []), 1, self.session_id
                ))
        
        logger.debug(f"Updated impact for {feature_name}: {impact_delta:+.5f}")
    
    def update_operation_performance(self,
                                   operation_name: str,
                                   category: str,
                                   improvement: float,
                                   execution_time: float,
                                   success: bool = True) -> None:
        """Update performance statistics for an operation."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get existing stats
            cursor.execute("""
                SELECT total_applications, success_count, avg_improvement, 
                       best_improvement, worst_result, avg_execution_time
                FROM operation_performance 
                WHERE operation_name = ? AND session_id = ?
            """, (operation_name, self.session_id))
            
            existing = cursor.fetchone()
            
            if existing:
                (total_apps, success_count, avg_imp, best_imp, 
                 worst_result, avg_exec_time) = existing
                
                # Update statistics
                new_total = total_apps + 1
                new_success = success_count + (1 if success else 0)
                new_avg_imp = ((avg_imp * total_apps) + improvement) / new_total
                new_best_imp = max(best_imp, improvement)
                new_worst = min(worst_result, improvement)
                new_avg_time = ((avg_exec_time * total_apps) + execution_time) / new_total
                
                # Calculate effectiveness score (success rate * avg improvement)
                effectiveness = (new_success / new_total) * max(0, new_avg_imp)
                
                cursor.execute("""
                    UPDATE operation_performance SET
                        total_applications = ?,
                        success_count = ?,
                        avg_improvement = ?,
                        best_improvement = ?,
                        worst_result = ?,
                        avg_execution_time = ?,
                        effectiveness_score = ?,
                        last_used = CURRENT_TIMESTAMP
                    WHERE operation_name = ? AND session_id = ?
                """, (
                    new_total, new_success, new_avg_imp, new_best_imp,
                    new_worst, new_avg_time, effectiveness, 
                    operation_name, self.session_id
                ))
            else:
                # Create new record
                effectiveness = (1 if success else 0) * max(0, improvement)
                
                cursor.execute("""
                    INSERT INTO operation_performance (
                        operation_name, operation_category, total_applications,
                        success_count, avg_improvement, best_improvement,
                        worst_result, avg_execution_time, effectiveness_score,
                        session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    operation_name, category, 1, (1 if success else 0),
                    improvement, improvement, improvement, execution_time,
                    effectiveness, self.session_id
                ))
        
        logger.debug(f"Updated performance for {operation_name}: {improvement:+.5f}")
    
    def get_best_features(self, limit: int = 10, session_id: str = None) -> List[Dict]:
        """Get top performing features by impact delta."""
        
        if session_id is None:
            session_id = self.session_id
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM top_features 
                WHERE session_id = ? OR ? IS NULL
                ORDER BY impact_delta DESC
                LIMIT ?
            """, (session_id, session_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_session_progress(self, session_id: str = None) -> Dict:
        """Get current session statistics."""
        
        if session_id is None:
            session_id = self.session_id
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM session_summary 
                WHERE session_id = ?
            """, (session_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else {}
    
    def get_operation_rankings(self, session_id: str = None) -> List[Dict]:
        """Get operation effectiveness rankings."""
        
        if session_id is None:
            session_id = self.session_id
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM operation_ranking 
                WHERE session_id = ? OR ? IS NULL
                ORDER BY effectiveness_score DESC
            """, (session_id, session_id))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_feature_code(self, feature_name: str) -> Optional[str]:
        """Get Python code for a specific feature."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT python_code FROM feature_catalog 
                WHERE feature_name = ? AND is_active = TRUE
            """, (feature_name,))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _create_backup(self) -> None:
        """Create a backup of the current database."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(
                self.backup_path, 
                f"feature_discovery_backup_{timestamp}.db"
            )
            
            shutil.copy2(self.db_path, backup_file)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info(f"Database backup created: {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files beyond the configured limit."""
        try:
            backup_files = list(Path(self.backup_path).glob("feature_discovery_backup_*.db"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            max_backups = self.db_config['max_backup_files']
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def _setup_cleanup_scheduler(self) -> None:
        """Setup automatic cleanup of old data based on retention policy."""
        if not self.db_config.get('auto_cleanup', True):
            return
            
        # This would typically be implemented with a background thread
        # For now, we'll do cleanup on database operations
        pass
    
    def export_best_features_code(self, output_file: str, limit: int = 20) -> None:
        """Export Python code for the best discovered features."""
        
        best_features = self.get_best_features(limit)
        
        with open(output_file, 'w') as f:
            f.write("# Auto-generated best features from MCTS exploration\n")
            f.write(f"# Generated at: {datetime.now()}\n")
            f.write(f"# Session: {self.session_id}\n\n")
            
            f.write("import numpy as np\nimport pandas as pd\n\n")
            
            for i, feature in enumerate(best_features, 1):
                f.write(f"# {i}. {feature['feature_name']} - Impact: +{feature['impact_delta']:.5f}\n")
                f.write(f"# Category: {feature['feature_category']}\n")
                f.write(f"# Improvement: {feature['impact_percentage']:+.2f}%\n")
                f.write(feature['python_code'])
                f.write("\n\n")
        
        logger.info(f"Exported {len(best_features)} best features to {output_file}")
    
    def close_session(self, status: str = 'completed') -> None:
        """Close the current session and mark it as completed."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET 
                    end_time = CURRENT_TIMESTAMP,
                    status = ?
                WHERE session_id = ?
            """, (status, self.session_id))
        
        logger.info(f"Session {self.session_id} closed with status: {status}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.close_session('interrupted')
        else:
            self.close_session('completed')
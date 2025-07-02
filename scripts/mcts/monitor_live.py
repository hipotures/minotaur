#!/usr/bin/env python3
"""
MCTS Live Monitoring Script

Real-time monitoring of active MCTS sessions showing live updates
as the algorithm runs. Displays current progress, best scores, and tree growth.

Usage:
    python scripts/mcts/monitor_live.py [SESSION_ID] [--refresh N]
    python scripts/mcts/monitor_live.py --latest --refresh 5
"""

# Suppress verbose logging FIRST, before any imports that might trigger DB connections
import logging
logging.getLogger('DB').setLevel(logging.WARNING)
logging.getLogger('db').setLevel(logging.WARNING)  # Database connection manager logger
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('src').setLevel(logging.WARNING)

import sys
import time
import argparse
import signal
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.discovery_db import FeatureDiscoveryDB
from src.utils import create_session_resolver, SessionResolutionError, add_universal_optional_session_args, validate_optional_session_args


def load_default_config() -> Dict[str, Any]:
    """Load default configuration from base config file."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'mcts_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


class MCTSMonitor:
    """Real-time MCTS session monitor."""
    
    def __init__(self, db: FeatureDiscoveryDB, session_id: str, refresh_interval: int = 10):
        self.db = db
        self.session_id = session_id
        self.refresh_interval = refresh_interval
        self.running = False
        self.last_iteration = -1
        self.start_time = time.time()
        
        # Statistics tracking
        self.iteration_history = []
        self.score_history = []
        self.best_score = 0.0
        self.total_evaluations = 0
        
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        query = """
        SELECT status, start_time, end_time, total_iterations, best_score
        FROM sessions WHERE session_id = ?
        """
        result = self.db.db_service.connection_manager.execute_query(query, params=(self.session_id,), fetch='one')
        
        if not result:
            return {"error": "Session not found"}
        
        return {
            "status": result[0],
            "start_time": result[1],
            "end_time": result[2],
            "total_iterations": result[3],
            "best_score": result[4]
        }
    
    def get_latest_activity(self) -> Dict[str, Any]:
        """Get latest MCTS activity."""
        query = """
        SELECT MAX(iteration) as latest_iteration,
               COUNT(*) as total_records,
               COUNT(DISTINCT mcts_node_id) as unique_nodes,
               MAX(evaluation_score) as current_best_score,
               AVG(evaluation_time) as avg_eval_time,
               SUM(evaluation_time) as total_eval_time
        FROM exploration_history 
        WHERE session_id = ?
        """
        
        result = self.db.db_service.connection_manager.execute_query(query, params=(self.session_id,), fetch='one')
        
        if not result or result[0] is None:
            return {"error": "No activity found"}
        
        # Get recent activity (last 10 iterations)
        recent_query = """
        SELECT iteration, operation_applied, evaluation_score, evaluation_time
        FROM exploration_history 
        WHERE session_id = ? AND iteration >= ?
        ORDER BY iteration DESC
        LIMIT 10
        """
        
        recent_iteration = max(0, result[0] - 9)
        recent_records = self.db.db_service.connection_manager.execute_query(recent_query, params=(self.session_id, recent_iteration), fetch='all')
        
        return {
            "latest_iteration": result[0],
            "total_records": result[1],
            "unique_nodes": result[2],
            "max_depth": 0,  # Not available in current schema
            "current_best_score": result[3] or 0,
            "avg_eval_time": round(result[4] or 0, 2),
            "total_eval_time": round(result[5] or 0, 2),
            "recent_activity": [
                {
                    "iteration": r[0],
                    "operation": r[1] or "root",
                    "score": round(r[2] or 0, 5),
                    "eval_time": round(r[3] or 0, 2)
                }
                for r in recent_records
            ]
        }
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation usage statistics."""
        query = """
        SELECT operation_applied, 
               COUNT(*) as count,
               AVG(evaluation_score) as avg_score,
               MAX(evaluation_score) as best_score
        FROM exploration_history 
        WHERE session_id = ? AND operation_applied != 'root'
        GROUP BY operation_applied
        ORDER BY avg_score DESC
        LIMIT 5
        """
        
        records = self.db.db_service.connection_manager.execute_query(query, params=(self.session_id,), fetch='all')
        
        return {
            operation: {
                "count": count,
                "avg_score": round(avg_score or 0, 5),
                "best_score": round(best_score or 0, 5)
            }
            for operation, count, avg_score, best_score in records
        }
    
    def get_tree_growth_stats(self) -> Dict[str, Any]:
        """Get tree growth statistics."""
        query = """
        SELECT 1 as tree_depth, COUNT(*) as node_count
        FROM exploration_history 
        WHERE session_id = ?
        """
        
        records = self.db.db_service.connection_manager.execute_query(query, params=(self.session_id,), fetch='all')
        
        depth_distribution = {depth: count for depth, count in records}
        
        return {
            "depth_distribution": depth_distribution,
            "max_depth": max(depth_distribution.keys()) if depth_distribution else 0,
            "total_nodes": sum(depth_distribution.values())
        }
    
    def check_mcts_log_activity(self) -> Dict[str, Any]:
        """Check recent MCTS log activity."""
        log_file = Path("logs/mcts.log")
        
        if not log_file.exists():
            return {"error": "MCTS log file not found"}
        
        try:
            # Get file modification time and size
            stat = log_file.stat()
            
            # Read last few lines for activity check
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            recent_lines = lines[-10:] if len(lines) >= 10 else lines
            
            # Count recent MCTS phases
            selection_count = sum(1 for line in recent_lines if "SELECTION PHASE START" in line)
            expansion_count = sum(1 for line in recent_lines if "EXPANSION PHASE START" in line)
            backprop_count = sum(1 for line in recent_lines if "BACKPROPAGATION PHASE START" in line)
            
            return {
                "file_size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S"),
                "recent_activity": {
                    "selection_phases": selection_count,
                    "expansion_phases": expansion_count,
                    "backpropagation_phases": backprop_count
                },
                "last_log_line": lines[-1].strip() if lines else "No logs"
            }
            
        except Exception as e:
            return {"error": f"Failed to read log: {e}"}
    
    def calculate_performance_metrics(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if "error" in activity:
            return {}
        
        current_iteration = activity.get("latest_iteration", 0)
        
        # Update tracking
        if current_iteration > self.last_iteration:
            self.iteration_history.append(current_iteration)
            current_score = activity.get("current_best_score", 0)
            self.score_history.append(current_score)
            
            if current_score > self.best_score:
                self.best_score = current_score
            
            self.last_iteration = current_iteration
        
        # Calculate rates
        elapsed_time = time.time() - self.start_time
        iterations_per_minute = (current_iteration / elapsed_time * 60) if elapsed_time > 0 else 0
        
        # Improvement tracking
        improvements = sum(1 for i in range(1, len(self.score_history)) 
                          if self.score_history[i] > self.score_history[i-1])
        
        return {
            "iterations_per_minute": round(iterations_per_minute, 2),
            "total_improvements": improvements,
            "time_since_start": round(elapsed_time, 1),
            "estimated_completion": "N/A"  # Would need max_iterations from config
        }
    
    def display_status(self):
        """Display current status in a formatted way."""
        # Clear screen
        print("\033[2J\033[H", end="")
        
        # Header
        print("=" * 80)
        print(f"🔴 MCTS LIVE MONITOR - Session {self.session_id}")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Refresh: {self.refresh_interval}s")
        print("=" * 80)
        
        # Get data
        session_status = self.get_session_status()
        activity = self.get_latest_activity()
        operations = self.get_operation_stats()
        tree_stats = self.get_tree_growth_stats()
        log_activity = self.check_mcts_log_activity()
        performance = self.calculate_performance_metrics(activity)
        
        # Session Status
        if "error" not in session_status:
            print(f"📊 Session Status: {session_status.get('status', 'Unknown')}")
            print(f"🎯 Best Score: {session_status.get('best_score', 0)}")
            print(f"⏱️  Start Time: {session_status.get('start_time', 'N/A')}")
        
        # Current Activity
        if "error" not in activity:
            print(f"\n🚀 Current Activity:")
            print(f"  Iteration: {activity.get('latest_iteration', 0)}")
            print(f"  Total Records: {activity.get('total_records', 0)}")
            print(f"  Unique Nodes: {activity.get('unique_nodes', 0)}")
            print(f"  Max Tree Depth: {activity.get('max_depth', 0)}")
            print(f"  Current Best Score: {activity.get('current_best_score', 0)}")
            print(f"  Avg Eval Time: {activity.get('avg_eval_time', 0)}s")
        else:
            print(f"\n❌ Activity Error: {activity['error']}")
        
        # Performance Metrics
        if performance:
            print(f"\n⚡ Performance:")
            print(f"  Iterations/min: {performance.get('iterations_per_minute', 0)}")
            print(f"  Total Improvements: {performance.get('total_improvements', 0)}")
            print(f"  Runtime: {performance.get('time_since_start', 0)}s")
        
        # Top Operations
        if operations:
            print(f"\n🎯 Top Operations:")
            for i, (op_name, stats) in enumerate(list(operations.items())[:3]):
                print(f"  {i+1}. {op_name}: {stats['avg_score']} avg ({stats['count']} uses)")
        
        # Tree Growth
        if tree_stats:
            print(f"\n🌳 Tree Structure:")
            print(f"  Max Depth: {tree_stats['max_depth']}")
            print(f"  Total Nodes: {tree_stats['total_nodes']}")
            depth_dist = tree_stats['depth_distribution']
            if depth_dist:
                print(f"  Depth Distribution: {dict(list(depth_dist.items())[:5])}")
        
        # Recent Activity
        if "error" not in activity and activity.get("recent_activity"):
            print(f"\n📈 Recent Activity (last 5):")
            for event in activity["recent_activity"][:5]:
                print(f"  Iter {event['iteration']}: {event['operation']} "
                      f"(score: {event['score']})")
        
        # Log Activity
        if "error" not in log_activity:
            print(f"\n📝 MCTS Logging:")
            print(f"  File Size: {log_activity['file_size']} bytes")
            print(f"  Last Modified: {log_activity['last_modified']}")
            recent = log_activity['recent_activity']
            print(f"  Recent Phases: {recent['selection_phases']} sel, "
                  f"{recent['expansion_phases']} exp, {recent['backpropagation_phases']} back")
        
        print(f"\n{'='*80}")
        print("Press Ctrl+C to stop monitoring")
    
    def start(self):
        """Start monitoring loop."""
        self.running = True
        
        print(f"🔴 Starting live monitoring for session: {self.session_id}")
        print(f"⏱️  Refresh interval: {self.refresh_interval} seconds")
        print("Press Ctrl+C to stop...\n")
        
        try:
            while self.running:
                self.display_status()
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n\n❌ Monitoring error: {e}")
            self.running = False
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


def get_latest_session_id(db: FeatureDiscoveryDB) -> Optional[str]:
    """Get the latest session ID."""
    query = "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1"
    result = db.db_service.connection_manager.execute_query(query, fetch='one')
    return result[0] if result else None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n\n🛑 Received interrupt signal, stopping monitor...")
    sys.exit(0)


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Live MCTS session monitoring")
    
    # Add universal optional session argument
    add_universal_optional_session_args(parser, "Live monitoring")
    
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration and initialize database
    try:
        config = load_default_config()
        
        # Temporarily change logging level to prevent console handlers in scripts
        original_log_level = config.get('logging', {}).get('level', 'INFO')
        config['logging']['level'] = 'WARNING'  # This prevents DEBUG console handlers
        
        db = FeatureDiscoveryDB(config, read_only=True)
        
        # Restore original log level in config
        config['logging']['level'] = original_log_level
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return 1
    
    # Resolve session using universal resolver (with optional fallback)
    try:
        session_identifier = validate_optional_session_args(args)
        
        # Temporarily change logging level for session resolver too
        config['logging']['level'] = 'WARNING'
        resolver = create_session_resolver(config)
        session_info = resolver.resolve_session(session_identifier)
        
        # Restore log level
        config['logging']['level'] = original_log_level
        session_id = session_info.session_id
        
        print(f"🔴 Monitoring session: {session_id[:8]}... ({session_info.session_name})")
        print(f"   Started: {session_info.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Close resolver connection to avoid conflicts
        resolver.connection_manager.close()
        
    except SessionResolutionError as e:
        print(f"❌ Session resolution failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error resolving session: {e}")
        return 1
    
    # Validate refresh interval
    if args.refresh < 1:
        print("❌ Refresh interval must be at least 1 second")
        return 1
    
    try:
        # Create and start monitor
        monitor = MCTSMonitor(db, session_id, args.refresh)
        monitor.start()
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return 1
    
    finally:
        if hasattr(db, 'close'):
            db.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
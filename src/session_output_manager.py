"""
Session Output Manager - Centralized session-based file organization

Provides unified session directory structure for all MCTS outputs:
- Reports and analytics
- Exports and results
- Logs and timing data
- Cache and temporary files
"""

import os
import json
import yaml
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SessionOutputManager:
    """Manages session-based output directory structure."""
    
    def __init__(self, session_name: str, session_id: str, base_output_dir: str = "outputs"):
        """Initialize session output manager."""
        self.session_name = session_name
        self.session_id = session_id
        self.base_output_dir = Path(base_output_dir)
        
        # Create main session directory
        self.session_dir = self.base_output_dir / "sessions" / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subdirectories
        self.subdirs = {
            'reports': self.session_dir / 'reports',
            'exports': self.session_dir / 'exports', 
            'logs': self.session_dir / 'logs',
            'cache': self.session_dir / 'cache',
            'metadata': self.session_dir / 'metadata'
        }
        
        # Create all subdirectories
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"📁 Session output directory initialized: {self.session_dir}")
    
    def get_reports_dir(self) -> Path:
        """Get reports directory for analytics and charts."""
        return self.subdirs['reports']
    
    def get_exports_dir(self) -> Path:
        """Get exports directory for results and feature code."""
        return self.subdirs['exports']
    
    def get_logs_dir(self) -> Path:
        """Get logs directory for session-specific logs."""
        return self.subdirs['logs']
    
    def get_cache_dir(self) -> Path:
        """Get cache directory for temporary session data."""
        return self.subdirs['cache']
    
    def get_metadata_dir(self) -> Path:
        """Get metadata directory for config snapshots."""
        return self.subdirs['metadata']
    
    def get_session_dir(self) -> Path:
        """Get main session directory."""
        return self.session_dir
    
    def save_metadata(self, config: Dict[str, Any], environment_info: Dict[str, Any] = None) -> None:
        """Save session metadata including config snapshot."""
        metadata_dir = self.get_metadata_dir()
        
        # Save config snapshot
        config_file = metadata_dir / 'config_snapshot.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Save session info
        session_info = {
            'session_name': self.session_name,
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'session_directory': str(self.session_dir),
            'subdirectories': {name: str(path) for name, path in self.subdirs.items()}
        }
        
        if environment_info:
            session_info['environment'] = environment_info
        
        session_file = metadata_dir / 'session_info.json'
        with open(session_file, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        logger.info(f"📋 Session metadata saved to: {metadata_dir}")
    
    def get_analytics_paths(self) -> Dict[str, Path]:
        """Get standard analytics file paths."""
        reports_dir = self.get_reports_dir()
        return {
            'html_report': reports_dir / 'mcts_analytics_report.html',
            'score_progression': reports_dir / 'score_progression.png',
            'operation_performance': reports_dir / 'operation_performance.png', 
            'timing_analysis': reports_dir / 'timing_analysis.png',
            'exploration_history': reports_dir / 'exploration_history.csv',
            'summary_statistics': reports_dir / 'summary_statistics.json',
            'timing_analysis_json': reports_dir / 'timing_analysis.json'
        }
    
    def get_export_paths(self) -> Dict[str, Path]:
        """Get standard export file paths."""
        exports_dir = self.get_exports_dir()
        return {
            'best_features_code': exports_dir / 'best_features_discovered.py',
            'discovery_results': exports_dir / 'discovery_session.json',
            'discovery_report': exports_dir / 'discovery_report.html',
            'features_export': exports_dir / 'features_export.csv',
            'sessions_export': exports_dir / 'sessions_export.csv'
        }
    
    def get_log_paths(self) -> Dict[str, Path]:
        """Get standard log file paths."""
        logs_dir = self.get_logs_dir()
        return {
            'session_log': logs_dir / 'session.log',
            'timing_data': logs_dir / 'timing_data.json'
        }
    
    def get_cache_paths(self) -> Dict[str, Path]:
        """Get standard cache file paths."""
        cache_dir = self.get_cache_dir()
        return {
            'features_cache': cache_dir / 'features.duckdb',
            'train_data': cache_dir / 'train.parquet',
            'test_data': cache_dir / 'test.parquet'
        }
    
    def create_session_summary(self, results: Dict[str, Any]) -> Path:
        """Create a session summary file with key results."""
        summary_file = self.session_dir / 'SESSION_SUMMARY.md'
        
        # Extract key metrics
        search_results = results.get('search_results', {})
        final_eval = results.get('final_evaluation', {})
        timing_stats = results.get('timing_statistics', {})
        
        summary_content = f"""# Session Summary: {self.session_name}

## Basic Information
- **Session ID**: `{self.session_id}`
- **Session Name**: `{self.session_name}`
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Runtime**: {results.get('total_runtime', 'N/A'):.2f}s

## Results
- **Best Score**: {search_results.get('best_score', 'N/A')}
- **Total Iterations**: {search_results.get('total_iterations', 'N/A')}
- **Total Evaluations**: {search_results.get('total_evaluations', 'N/A')}

## Best Features Path
```
{' → '.join(results.get('best_features_path', []))}
```

## Files Generated
- **Reports**: `{self.subdirs['reports'].relative_to(self.base_output_dir)}/`
- **Exports**: `{self.subdirs['exports'].relative_to(self.base_output_dir)}/`
- **Logs**: `{self.subdirs['logs'].relative_to(self.base_output_dir)}/`
- **Cache**: `{self.subdirs['cache'].relative_to(self.base_output_dir)}/`
- **Metadata**: `{self.subdirs['metadata'].relative_to(self.base_output_dir)}/`

## Quick Access
- View analytics: `{self.subdirs['reports']/  'mcts_analytics_report.html'}`
- View best features: `{self.subdirs['exports'] / 'best_features_discovered.py'}`
- View session log: `{self.subdirs['logs'] / 'session.log'}`
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"📄 Session summary created: {summary_file}")
        return summary_file
    
    @staticmethod
    def cleanup_old_sessions(base_output_dir: str = "outputs", days_old: int = 30) -> None:
        """Clean up session directories older than specified days."""
        sessions_dir = Path(base_output_dir) / "sessions"
        if not sessions_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                try:
                    # Extract date from session name (session_YYYYMMDD_HHMMSS)
                    date_part = session_dir.name.split('_')[1]
                    session_date = datetime.strptime(date_part, '%Y%m%d')
                    
                    if session_date < cutoff_date:
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(f"🗑️  Cleaned old session: {session_dir.name}")
                        
                except (ValueError, IndexError):
                    # Skip directories that don't match expected format
                    continue
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned {cleaned_count} old session directories")
    
    def __str__(self) -> str:
        """String representation of the session output manager."""
        return f"SessionOutputManager(session={self.session_name}, dir={self.session_dir})"
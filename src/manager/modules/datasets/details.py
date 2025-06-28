"""
Details Command - Show comprehensive information about a specific dataset.

Provides detailed dataset information including:
- Basic metadata (name, ID, description, files)
- Usage statistics (sessions, features, performance)
- File information (paths, sizes, checksums)
- Configuration details (columns, types)
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from .base import BaseDatasetsCommand


class DetailsCommand(BaseDatasetsCommand):
    """Handle --details command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the show dataset details command."""
        try:
            dataset_identifier = args.details
            dataset = self.find_dataset_by_identifier(dataset_identifier)
            
            if not dataset:
                self.print_error(f"Dataset '{dataset_identifier}' not found.")
                self.print_info("List available datasets: python manager.py datasets --list")
                return
            
            # Get comprehensive dataset information
            details = self._gather_dataset_details(dataset)
            
            # Output in requested format
            if getattr(args, 'format', 'table') == 'json':
                self._output_json(details)
            else:
                self._output_detailed_view(details)
                
        except Exception as e:
            self.print_error(f"Failed to show dataset details: {e}")
    
    def _gather_dataset_details(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive dataset information."""
        dataset_id = dataset['dataset_id']
        
        # Get sessions using this dataset (simplified for now)
        sessions = []
        try:
            sessions = self.session_service.repository.get_sessions_by_dataset(dataset_id)
        except Exception:
            sessions = []
        
        # Get feature information (simplified for now)
        features = []
        try:
            features = self.feature_service.repository.get_features_by_dataset(dataset_id)
        except Exception:
            features = []
        
        # Calculate statistics
        stats = self._calculate_dataset_statistics(dataset, sessions, features)
        
        # Get file information
        file_info = self._get_file_information(dataset)
        
        return {
            'basic_info': dataset,
            'statistics': stats,
            'file_info': file_info,
            'sessions': self._summarize_sessions(sessions),
            'features': self._summarize_features(features),
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_dataset_statistics(self, dataset: Dict[str, Any], 
                                    sessions: list, features: list) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        stats = {
            'session_count': len(sessions),
            'feature_count': len(features),
            'total_size_bytes': dataset.get('size_bytes', 0),
            'last_used': None,
            'success_rate': 0.0,
            'avg_score': 0.0,
            'best_score': 0.0
        }
        
        if sessions:
            # Find last used date
            recent_session = max(sessions, key=lambda s: s.get('start_time', ''), default=None)
            if recent_session:
                stats['last_used'] = recent_session.get('start_time')
            
            # Calculate performance statistics
            completed_sessions = [s for s in sessions if s.get('status') == 'completed']
            if completed_sessions:
                stats['success_rate'] = len(completed_sessions) / len(sessions) * 100
                
                scores = [s.get('best_score', 0) for s in completed_sessions if s.get('best_score')]
                if scores:
                    stats['avg_score'] = sum(scores) / len(scores)
                    stats['best_score'] = max(scores)
        
        return stats
    
    def _get_file_information(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed file information for the dataset."""
        file_info = {
            'train_file': self._get_file_details(dataset.get('train_path')),
            'test_file': self._get_file_details(dataset.get('test_path')),
            'submission_file': self._get_file_details(dataset.get('submission_path')),
            'validation_file': self._get_file_details(dataset.get('validation_path'))
        }
        
        # Remove None entries
        return {k: v for k, v in file_info.items() if v is not None}
    
    def _get_file_details(self, file_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get details for a specific file."""
        if not file_path:
            return None
        
        try:
            import os
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                return {
                    'path': file_path,
                    'size_bytes': stat.st_size,
                    'size_formatted': self.format_size(stat.st_size),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'exists': True
                }
            else:
                return {
                    'path': file_path,
                    'exists': False
                }
        except Exception:
            return {
                'path': file_path,
                'exists': False,
                'error': 'Cannot access file'
            }
    
    def _summarize_sessions(self, sessions: list) -> Dict[str, Any]:
        """Summarize session information."""
        if not sessions:
            return {'count': 0, 'recent': []}
        
        # Sort by start time (most recent first)
        sorted_sessions = sorted(sessions, 
                               key=lambda s: s.get('start_time', ''), 
                               reverse=True)
        
        # Get recent sessions (last 5)
        recent_sessions = []
        for session in sorted_sessions[:5]:
            recent_sessions.append({
                'session_id': session.get('session_id', '')[:8],
                'start_time': session.get('start_time'),
                'status': session.get('status', 'unknown'),
                'best_score': session.get('best_score'),
                'total_iterations': session.get('total_iterations', 0)
            })
        
        return {
            'count': len(sessions),
            'recent': recent_sessions,
            'status_breakdown': self._get_session_status_breakdown(sessions)
        }
    
    def _get_session_status_breakdown(self, sessions: list) -> Dict[str, int]:
        """Get breakdown of session statuses."""
        breakdown = {}
        for session in sessions:
            status = session.get('status', 'unknown')
            breakdown[status] = breakdown.get(status, 0) + 1
        return breakdown
    
    def _summarize_features(self, features: list) -> Dict[str, Any]:
        """Summarize feature information."""
        if not features:
            return {'count': 0, 'top_features': []}
        
        # Sort by impact (highest first)
        sorted_features = sorted(features, 
                               key=lambda f: f.get('avg_impact', 0), 
                               reverse=True)
        
        # Get top features (top 5)
        top_features = []
        for feature in sorted_features[:5]:
            top_features.append({
                'name': feature.get('feature_name'),
                'avg_impact': feature.get('avg_impact'),
                'usage_count': feature.get('usage_count', 0),
                'success_rate': feature.get('success_rate', 0)
            })
        
        return {
            'count': len(features),
            'top_features': top_features
        }
    
    def _output_detailed_view(self, details: Dict[str, Any]) -> None:
        """Output dataset details in formatted view."""
        basic = details['basic_info']
        stats = details['statistics']
        files = details['file_info']
        sessions = details['sessions']
        features = details['features']
        
        # Basic Information
        print(f"\n📊 DATASET DETAILS")
        print("=" * 60)
        print(f"📋 Name: {basic['dataset_name']}")
        print(f"🔑 ID: {basic['dataset_id']}")
        print(f"📝 Description: {basic.get('description', 'No description')}")
        print(f"🏷️  Competition: {basic.get('competition_name', 'N/A')}")
        print(f"📅 Registered: {basic.get('created_at', 'Unknown')}")
        
        # Configuration
        print(f"\n⚙️  CONFIGURATION")
        print("-" * 30)
        print(f"🎯 Target Column: {basic.get('target_column', 'N/A')}")
        print(f"🆔 ID Column: {basic.get('id_column', 'N/A')}")
        print(f"🔒 Dataset Hash: {basic.get('dataset_hash', 'N/A')[:16]}...")
        
        # File Information
        print(f"\n📁 FILES")
        print("-" * 30)
        for file_type, file_info in files.items():
            if file_info:
                status = "✅" if file_info.get('exists', False) else "❌"
                size = file_info.get('size_formatted', 'Unknown')
                print(f"{status} {file_type.replace('_', ' ').title()}: {size}")
                if not file_info.get('exists', False):
                    print(f"   Path: {file_info.get('path', 'N/A')}")
        
        # Statistics
        print(f"\n📈 USAGE STATISTICS")
        print("-" * 30)
        print(f"📊 Total Sessions: {stats['session_count']}")
        print(f"🧪 Features Generated: {stats['feature_count']}")
        print(f"💾 Total Size: {self.format_size(stats['total_size_bytes'])}")
        
        if stats['last_used']:
            print(f"🕒 Last Used: {stats['last_used'][:10]}")
        else:
            print(f"🕒 Last Used: Never")
        
        if stats['session_count'] > 0:
            print(f"✅ Success Rate: {stats['success_rate']:.1f}%")
            if stats['avg_score'] > 0:
                print(f"📊 Average Score: {stats['avg_score']:.5f}")
                print(f"🏆 Best Score: {stats['best_score']:.5f}")
        
        # Recent Sessions
        if sessions['recent']:
            print(f"\n🕒 RECENT SESSIONS")
            print("-" * 30)
            for session in sessions['recent']:
                status_icon = {"completed": "✅", "failed": "❌", "running": "⏳"}.get(session['status'], "❔")
                score_info = f" (Score: {session['best_score']:.5f})" if session.get('best_score') else ""
                print(f"{status_icon} {session['session_id']} - {session['start_time'][:10]}{score_info}")
        
        # Top Features
        if features['top_features']:
            print(f"\n🧪 TOP FEATURES")
            print("-" * 30)
            for feature in features['top_features']:
                impact = feature.get('avg_impact', 0)
                impact_icon = "📈" if impact > 0 else "📉" if impact < 0 else "➖"
                print(f"{impact_icon} {feature['name']}: {impact:+.5f} impact")
        
        # Quick Actions
        print(f"\n💡 QUICK ACTIONS")
        print("-" * 30)
        dataset_name = basic['dataset_name']
        print(f"   View sessions: python manager.py datasets --sessions {dataset_name}")
        print(f"   Show statistics: python manager.py datasets --stats")
        print(f"   Update metadata: python manager.py datasets --update {dataset_name}")
    
    def _output_json(self, details: Dict[str, Any]) -> None:
        """Output dataset details in JSON format."""
        self.print_json(details, f"Dataset Details: {details['basic_info']['dataset_name']}")
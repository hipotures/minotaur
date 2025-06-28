"""
Convergence command for analytics module
"""

from typing import Dict, Any, Optional, List
from .base import BaseAnalyticsCommand
from manager.core.utils import format_number, format_duration


class ConvergenceCommand(BaseAnalyticsCommand):
    """Analyze convergence patterns for a session."""
    
    def execute(self, session_id: str) -> None:
        """Execute convergence analysis command.
        
        Args:
            session_id: Session ID to analyze
        """
        # Get convergence data from service
        convergence_data = self.service.analyze_convergence(session_id)
        
        # Output in requested format
        self.output(convergence_data, title=f"Convergence Analysis - {session_id[:8]}...")
    
    def _format_text(self, data: Dict[str, Any], title: Optional[str] = None) -> str:
        """Format convergence data as text."""
        lines = []
        
        if 'error' in data:
            return f"❌ {data['error']}"
        
        lines.append("📊 CONVERGENCE ANALYSIS")
        lines.append("=" * 50)
        
        # Session info
        lines.append(f"\nSession ID: {data['session_id']}")
        lines.append(f"Total Iterations: {format_number(data['total_iterations'])}")
        lines.append(f"Final Score: {data['final_score']:.5f}")
        lines.append(f"Iterations to Best: {format_number(data['iterations_to_best'])}")
        lines.append(f"Improvement Rate: {data['improvement_rate']:.6f}/iteration")
        
        # Plateau analysis
        plateaus = data.get('plateaus', [])
        if plateaus:
            lines.append(f"\n🏔️  PLATEAUS DETECTED: {len(plateaus)}")
            for i, plateau in enumerate(plateaus, 1):
                lines.append(
                    f"   {i}. Iterations {plateau['start']}-{plateau['end']} "
                    f"(duration: {plateau['duration']}, score: {plateau['score']:.5f})"
                )
        
        # Convergence visualization
        convergence_points = data.get('convergence_data', [])
        if convergence_points:
            lines.append("\n📈 CONVERGENCE CHART:")
            chart = self._create_convergence_chart(convergence_points)
            lines.append(chart)
        
        # Key milestones
        if convergence_points:
            milestones = self._extract_milestones(convergence_points)
            if milestones:
                lines.append("\n🎯 KEY MILESTONES:")
                for milestone in milestones:
                    lines.append(
                        f"   Iteration {milestone['iteration']}: "
                        f"Score {milestone['score']:.5f} "
                        f"({milestone['description']})"
                    )
        
        # Convergence statistics
        if len(convergence_points) >= 10:
            lines.append("\n📊 CONVERGENCE STATISTICS:")
            
            # Calculate phases
            early = convergence_points[:len(convergence_points)//3]
            middle = convergence_points[len(convergence_points)//3:2*len(convergence_points)//3]
            late = convergence_points[2*len(convergence_points)//3:]
            
            early_improvement = sum(p['score'] - p['best_score'] for p in early[1:]) / len(early)
            middle_improvement = sum(p['score'] - p['best_score'] for p in middle[1:]) / len(middle)
            late_improvement = sum(p['score'] - p['best_score'] for p in late[1:]) / len(late)
            
            lines.append(f"   Early Phase Improvement Rate: {early_improvement:.6f}")
            lines.append(f"   Middle Phase Improvement Rate: {middle_improvement:.6f}")
            lines.append(f"   Late Phase Improvement Rate: {late_improvement:.6f}")
            
            # Convergence speed
            halfway_score = (convergence_points[0]['best_score'] + convergence_points[-1]['best_score']) / 2
            halfway_iter = next((i for i, p in enumerate(convergence_points) 
                               if p['best_score'] >= halfway_score), len(convergence_points)//2)
            convergence_speed = halfway_iter / len(convergence_points)
            
            if convergence_speed < 0.3:
                speed_desc = "Fast"
            elif convergence_speed < 0.6:
                speed_desc = "Moderate"
            else:
                speed_desc = "Slow"
            
            lines.append(f"   Convergence Speed: {speed_desc} ({convergence_speed:.2%} to halfway)")
        
        return '\n'.join(lines)
    
    def _create_convergence_chart(self, points: List[Dict[str, Any]], 
                                 width: int = 60, height: int = 15) -> str:
        """Create ASCII chart showing convergence."""
        if not points:
            return "No data available"
        
        # Extract scores
        scores = [p['score'] for p in points]
        best_scores = [p['best_score'] for p in points]
        
        min_score = min(min(scores), min(best_scores))
        max_score = max(max(scores), max(best_scores))
        score_range = max_score - min_score
        
        if score_range == 0:
            return f"Flat score at {min_score:.5f}"
        
        # Sample points if too many
        if len(points) > width:
            step = len(points) // width
            sampled_scores = scores[::step][:width]
            sampled_best = best_scores[::step][:width]
        else:
            sampled_scores = scores
            sampled_best = best_scores
        
        # Create chart
        chart_lines = []
        
        # Scale to chart height
        for row in range(height - 1, -1, -1):
            line = []
            threshold = min_score + (score_range * row / (height - 1))
            
            for i in range(len(sampled_scores)):
                if sampled_best[i] >= threshold:
                    line.append('█')  # Best score
                elif sampled_scores[i] >= threshold:
                    line.append('░')  # Current score
                else:
                    line.append(' ')
            
            # Add axis label
            if row == height - 1:
                label = f"{max_score:.5f} |"
            elif row == 0:
                label = f"{min_score:.5f} |"
            else:
                label = "         |"
            
            chart_lines.append(label + ''.join(line))
        
        # Add bottom axis
        chart_lines.append("         " + "+" + "-" * len(sampled_scores))
        chart_lines.append("         0" + " " * (len(sampled_scores) - 10) + f"{len(points)} iterations")
        chart_lines.append("\n         Legend: █ = Best Score, ░ = Current Score")
        
        return '\n'.join(chart_lines)
    
    def _extract_milestones(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key milestones from convergence data."""
        if not points:
            return []
        
        milestones = []
        
        # First improvement
        first_improvement = next((i for i, p in enumerate(points[1:], 1) 
                                if p['score'] > points[0]['score']), None)
        if first_improvement:
            milestones.append({
                'iteration': first_improvement,
                'score': points[first_improvement]['score'],
                'description': 'First improvement'
            })
        
        # 50% of final score
        initial_score = points[0]['score']
        final_score = points[-1]['best_score']
        halfway_score = initial_score + (final_score - initial_score) * 0.5
        
        halfway_point = next((i for i, p in enumerate(points) 
                            if p['best_score'] >= halfway_score), None)
        if halfway_point:
            milestones.append({
                'iteration': halfway_point,
                'score': points[halfway_point]['best_score'],
                'description': '50% of improvement'
            })
        
        # 90% of final score
        ninety_score = initial_score + (final_score - initial_score) * 0.9
        ninety_point = next((i for i, p in enumerate(points) 
                           if p['best_score'] >= ninety_score), None)
        if ninety_point:
            milestones.append({
                'iteration': ninety_point,
                'score': points[ninety_point]['best_score'],
                'description': '90% of improvement'
            })
        
        # Best score achieved
        best_iter = max(enumerate(points), key=lambda x: x[1]['score'])[0]
        milestones.append({
            'iteration': best_iter,
            'score': points[best_iter]['score'],
            'description': 'Best score achieved'
        })
        
        return sorted(milestones, key=lambda x: x['iteration'])
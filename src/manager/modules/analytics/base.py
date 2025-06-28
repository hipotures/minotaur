"""
Base class for analytics commands
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import csv
import io
from pathlib import Path


class BaseAnalyticsCommand(ABC):
    """Base class for all analytics command handlers."""
    
    def __init__(self, context: Dict[str, Any]):
        """Initialize command with context.
        
        Args:
            context: Command context containing args, manager, service, etc.
        """
        self.context = context
        self.args = context['args']
        self.manager = context['manager']
        self.service = context['service']
        self.format = context.get('format', 'text')
        self.output_path = context.get('output')
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Execute the command."""
        pass
    
    def output(self, data: Any, title: Optional[str] = None) -> None:
        """Output data in the requested format.
        
        Args:
            data: Data to output
            title: Optional title for the output
        """
        if self.format == 'json':
            output_str = json.dumps(data, indent=2, default=str)
        elif self.format == 'csv' and isinstance(data, list) and data:
            output_str = self._format_csv(data)
        elif self.format == 'html':
            output_str = self._format_html(data, title)
        else:
            output_str = self._format_text(data, title)
        
        if self.output_path:
            # Write to file
            output_file = Path(self.output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(output_str)
            print(f"✅ Output written to: {output_file}")
        else:
            # Print to console
            print(output_str)
    
    def _format_csv(self, data: list) -> str:
        """Format data as CSV.
        
        Args:
            data: List of dictionaries to format
            
        Returns:
            CSV string
        """
        if not data or not isinstance(data[0], dict):
            return str(data)
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    
    def _format_html(self, data: Any, title: Optional[str] = None) -> str:
        """Format data as HTML.
        
        Args:
            data: Data to format
            title: Optional title
            
        Returns:
            HTML string
        """
        html_parts = ['<!DOCTYPE html><html><head>']
        html_parts.append('<meta charset="utf-8">')
        html_parts.append('<style>')
        html_parts.append('''
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; font-weight: bold; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric { font-size: 24px; font-weight: bold; color: #0066cc; }
            .section { margin: 20px 0; padding: 20px; background-color: #f5f5f5; border-radius: 5px; }
        ''')
        html_parts.append('</style>')
        
        if title:
            html_parts.append(f'<title>{title}</title>')
        
        html_parts.append('</head><body>')
        
        if title:
            html_parts.append(f'<h1>{title}</h1>')
        
        # Convert data to HTML based on type
        if isinstance(data, dict):
            html_parts.append(self._dict_to_html(data))
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            html_parts.append(self._list_to_html_table(data))
        else:
            html_parts.append(f'<pre>{json.dumps(data, indent=2, default=str)}</pre>')
        
        html_parts.append('</body></html>')
        return '\n'.join(html_parts)
    
    def _dict_to_html(self, data: dict, level: int = 2) -> str:
        """Convert dictionary to HTML sections."""
        html_parts = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                html_parts.append(f'<h{level}>{key.replace("_", " ").title()}</h{level}>')
                html_parts.append('<div class="section">')
                html_parts.append(self._dict_to_html(value, level + 1))
                html_parts.append('</div>')
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                html_parts.append(f'<h{level}>{key.replace("_", " ").title()}</h{level}>')
                html_parts.append(self._list_to_html_table(value))
            else:
                html_parts.append(f'<p><strong>{key.replace("_", " ").title()}:</strong> ')
                if isinstance(value, (int, float)) and key in ['score', 'count', 'total']:
                    html_parts.append(f'<span class="metric">{value}</span>')
                else:
                    html_parts.append(str(value))
                html_parts.append('</p>')
        
        return '\n'.join(html_parts)
    
    def _list_to_html_table(self, data: list) -> str:
        """Convert list of dictionaries to HTML table."""
        if not data:
            return '<p>No data available</p>'
        
        html_parts = ['<table>']
        
        # Header
        html_parts.append('<thead><tr>')
        for key in data[0].keys():
            html_parts.append(f'<th>{key.replace("_", " ").title()}</th>')
        html_parts.append('</tr></thead>')
        
        # Body
        html_parts.append('<tbody>')
        for row in data:
            html_parts.append('<tr>')
            for value in row.values():
                html_parts.append(f'<td>{value}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        return '\n'.join(html_parts)
    
    @abstractmethod
    def _format_text(self, data: Any, title: Optional[str] = None) -> str:
        """Format data as text. Must be implemented by subclasses."""
        pass
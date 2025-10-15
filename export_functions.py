import os
import pandas as pd
from typing import List, Dict, Any
from fpdf import FPDF
from datetime import datetime
import plotly.io as pio
import json
from pathlib import Path

class MovementDataExporter:
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize the exporter
        
        Args:
            output_dir: Directory to save exports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_pdf_report(self, analysis_results: Dict[str, Any],
                         athlete_name: str,
                         attempt_num: int,
                         plots: List[Any]) -> str:
        """
        Export analysis results as PDF report
        
        Args:
            analysis_results: Dictionary containing analysis results
            athlete_name: Name of the athlete
            attempt_num: Attempt number
            plots: List of plot figures
            
        Returns:
            str: Path to the exported PDF file
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Movement Analysis Report - {athlete_name}', ln=True)
        pdf.cell(0, 10, f'Attempt {attempt_num}', ln=True)
        pdf.ln(10)
        
        # Quality Metrics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Quality Metrics', ln=True)
        pdf.set_font('Arial', '', 10)
        
        metrics = analysis_results['quality_metrics']
        pdf.cell(0, 10, f'Interpolation Quality: {metrics["interpolation_quality"]:.2f}', ln=True)
        pdf.cell(0, 10, f'Data Quality Score: {metrics["data_quality_score"]:.2f}', ln=True)
        pdf.cell(0, 10, f'Technical Stability: {metrics["technical_stability"]:.2f}', ln=True)
        pdf.cell(0, 10, f'Noise Level: {metrics["noise_level"]:.2f} mm', ln=True)
        pdf.cell(0, 10, f'Gap Count: {metrics["gap_count"]}', ln=True)
        pdf.cell(0, 10, f'Critical Zone Gaps: {metrics["critical_zone_gaps"]}', ln=True)
        pdf.ln(10)
        
        # Zone Analysis
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Zone Analysis', ln=True)
        pdf.set_font('Arial', '', 10)
        
        zones = analysis_results['zone_analysis']
        for zone_name, zone_data in zones.items():
            pdf.cell(0, 10, f'{zone_name} Zone:', ln=True)
            pdf.cell(0, 10, f'Mean Velocity: {zone_data["mean_velocity"]:.2f} m/s', ln=True)
            pdf.cell(0, 10, f'Step Length: {zone_data["step_length_mean"]:.2f} mm', ln=True)
            pdf.cell(0, 10, f'Acceleration: {zone_data["acceleration_mean"]:.2f} m/s²', ln=True)
            pdf.ln(5)
        
        # Add plots
        for i, plot in enumerate(plots):
            plot_path = os.path.join(self.output_dir, f'plot_{i}.png')
            pio.write_image(plot, plot_path)
            pdf.image(plot_path, x=10, y=None, w=190)
            pdf.ln(10)
        
        # Save PDF
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path = os.path.join(self.output_dir, f'report_{athlete_name}_{attempt_num}_{timestamp}.pdf')
        pdf.output(pdf_path)
        
        return pdf_path
        
    def export_excel(self, analysis_results: Dict[str, Any],
                    athlete_name: str,
                    attempt_num: int) -> str:
        """
        Export analysis results as Excel file
        
        Args:
            analysis_results: Dictionary containing analysis results
            athlete_name: Name of the athlete
            attempt_num: Attempt number
            
        Returns:
            str: Path to the exported Excel file
        """
        # Create DataFrames for each section
        metrics_df = pd.DataFrame([analysis_results['quality_metrics']])
        zones_df = pd.DataFrame(analysis_results['zone_analysis']).T
        
        # Create Excel writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = os.path.join(self.output_dir, f'analysis_{athlete_name}_{attempt_num}_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path) as writer:
            metrics_df.to_excel(writer, sheet_name='Quality Metrics')
            zones_df.to_excel(writer, sheet_name='Zone Analysis')
            
            # Add raw data if available
            if 'raw_data' in analysis_results:
                pd.DataFrame(analysis_results['raw_data']).to_excel(writer, sheet_name='Raw Data')
        
        return excel_path
        
    def export_presentation(self, analysis_results: Dict[str, Any],
                          athlete_name: str,
                          attempt_num: int,
                          plots: List[Any]) -> str:
        """
        Export analysis results as presentation material
        
        Args:
            analysis_results: Dictionary containing analysis results
            athlete_name: Name of the athlete
            attempt_num: Attempt number
            plots: List of plot figures
            
        Returns:
            str: Path to the exported presentation file
        """
        # Create presentation directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        presentation_dir = os.path.join(self.output_dir, f'presentation_{athlete_name}_{attempt_num}_{timestamp}')
        os.makedirs(presentation_dir, exist_ok=True)
        
        # Save plots
        for i, plot in enumerate(plots):
            plot_path = os.path.join(presentation_dir, f'slide_{i+1}.png')
            pio.write_image(plot, plot_path)
        
        # Save analysis results as JSON
        results_path = os.path.join(presentation_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        # Create README
        readme_path = os.path.join(presentation_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(f'Movement Analysis Presentation - {athlete_name}\n')
            f.write(f'Attempt {attempt_num}\n\n')
            f.write('Contents:\n')
            f.write('1. Analysis Results (analysis_results.json)\n')
            f.write('2. Movement Profile (slide_1.png)\n')
            f.write('3. Velocity Profile (slide_2.png)\n')
            f.write('4. Zone Analysis (slide_3.png)\n')
            f.write('5. Quality Metrics (slide_4.png)\n')
        
        return presentation_dir

if __name__ == "__main__":
    # Example usage
    exporter = MovementDataExporter()
    
    # Example analysis results
    analysis_results = {
        'quality_metrics': {
            'interpolation_quality': 0.85,
            'data_quality_score': 0.92,
            'technical_stability': 0.78,
            'noise_level': 150.5,
            'gap_count': 2,
            'critical_zone_gaps': 1
        },
        'zone_analysis': {
            '11-6m': {
                'mean_velocity': 7.2,
                'step_length_mean': 2.1,
                'acceleration_mean': 0.5
            },
            '6-1m': {
                'mean_velocity': 8.1,
                'step_length_mean': 2.3,
                'acceleration_mean': 0.3
            }
        }
    }
    
    # Export all formats
    pdf_path = exporter.export_pdf_report(analysis_results, "Test Athlete", 1, [])
    excel_path = exporter.export_excel(analysis_results, "Test Athlete", 1)
    presentation_dir = exporter.export_presentation(analysis_results, "Test Athlete", 1, []) 
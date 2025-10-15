import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import json
from pathlib import Path

OSP_COLORS = {
    'red': '#e30613',
    'black': '#000000',
    'white': '#ffffff',
    'gold': '#ffd700',
    'gray': '#f5f5f5',
    'light_gray': '#e8e8e8',
    'green': '#28a745',
    'yellow': '#ffc107',
}

@dataclass
class QualityMetrics:
    interpolation_quality: float
    data_quality_score: float
    technical_stability: float
    noise_level: float
    gap_count: int
    critical_zone_gaps: int

@dataclass
class ZoneAnalysis:
    mean_velocity: float
    velocity_std: float
    step_length_mean: float
    step_length_std: float
    acceleration_mean: float
    acceleration_std: float
    data_points: int
    gaps: List[Dict[str, Any]]

class MovementAnalysisDashboard:
    def __init__(self, analyzer):
        """
        Initialize the dashboard with the movement data analyzer
        
        Args:
            analyzer: Instance of MovementDataAnalyzer
        """
        self.analyzer = analyzer
        self.file_list = self.get_file_list()
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_dashboard_layout()
        
    def get_file_list(self):
        """Scan input folders and create file list with metadata."""
        file_data = []
        for folder in self.analyzer.folders:
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith('.dat'):
                    fpath = os.path.join(folder, fname)
                    try:
                        takeoff_point, data, athlete_name, attempt_num = self.analyzer.read_data_file(fpath)
                        gap_analysis = self.analyzer.analyze_gaps_until_takeoff(fpath)
                        
                        # Determine quality
                        num_gaps = gap_analysis['number_of_gaps']
                        num_gaps_6_1 = len(gap_analysis['gaps_6_1'])
                        num_gaps_11_6 = len(gap_analysis['gaps_11_6'])
                        
                        if num_gaps_6_1 > 0:
                            status = 'rot'
                            quality = 'Kritisch'
                        elif num_gaps_11_6 > 0:
                            status = 'gelb'
                            quality = 'Achtung'
                        elif num_gaps > 0:
                            status = 'grün'
                            quality = 'OK'
                        else:
                            status = 'grün'
                            quality = 'Sehr gut'
                        
                        file_data.append({
                            'filepath': fpath,
                            'filename': fname,
                            'folder': os.path.basename(folder),
                            'athlete': athlete_name,
                            'attempt': attempt_num,
                            'gaps': num_gaps,
                            'gaps_critical': num_gaps_6_1,
                            'gaps_warning': num_gaps_11_6,
                            'quality': quality,
                            'status': status,
                            'takeoff_point': takeoff_point
                        })
                    except Exception as e:
                        print(f"Error processing {fname}: {e}")
                        continue
        
        return sorted(file_data, key=lambda x: (x['folder'], x['athlete'], x['attempt']))

    def calculate_interpolation_quality(self, original_data: List[float], 
                                     interpolated_data: List[float], 
                                     gap_ranges: List[Tuple[int, int]]) -> float:
        """Calculate the quality of interpolation based on smoothness and continuity"""
        if not gap_ranges:
            return 1.0
            
        try:
            original_velocities = np.diff(original_data)
            interpolated_velocities = np.diff(interpolated_data)
            
            smoothness_score = max(0, 1 - np.mean(np.abs(np.diff(interpolated_velocities))) / 100)
            
            continuity_scores = []
            for start, end in gap_ranges:
                if end < len(original_data):
                    continuity_scores.append(abs(interpolated_data[end] - original_data[end]))
            
            if continuity_scores:
                continuity_score = max(0, 1 - np.mean(continuity_scores) / np.max(original_data))
            else:
                continuity_score = 1.0
            
            return (smoothness_score + continuity_score) / 2
        except:
            return 0.5

    def calculate_data_quality(self, data: List[float], takeoff_point: float) -> float:
        """Calculate overall data quality score"""
        try:
            noise_level = np.std(np.diff(data))
            noise_score = max(0, 1 - min(noise_level / 1000, 1))
            
            gaps = self.analyzer.check_for_gaps(data, takeoff_point)
            gap_score = max(0, 1 - min(len(gaps) / 5, 1))
            
            velocities = self.analyzer.calculate_velocity(data)
            valid_velocities = [v for v in velocities if not np.isnan(v)]
            if valid_velocities:
                velocity_score = max(0, 1 - min(np.std(valid_velocities) / 5, 1))
            else:
                velocity_score = 0
            
            return (noise_score + gap_score + velocity_score) / 3
        except:
            return 0.5

    def calculate_technical_stability(self, data: List[float]) -> float:
        """Calculate technical stability score"""
        try:
            step_pattern = self.analyzer.analyze_step_pattern(data)
            step_stability = max(0, 1 - min(step_pattern['std_step_size'] / 100, 1))
            
            velocities = self.analyzer.calculate_velocity(data)
            valid_velocities = [v for v in velocities if not np.isnan(v)]
            if valid_velocities:
                velocity_stability = max(0, 1 - min(np.std(valid_velocities) / 5, 1))
            else:
                velocity_stability = 0
            
            accelerations = np.diff(velocities)
            valid_accelerations = [a for a in accelerations if not np.isnan(a)]
            if valid_accelerations:
                acceleration_stability = max(0, 1 - min(np.std(valid_accelerations) / 2, 1))
            else:
                acceleration_stability = 0
            
            return (step_stability + velocity_stability + acceleration_stability) / 3
        except:
            return 0.5

    def analyze_zone(self, data: List[float], takeoff_point: float,
                    zone_start: float, zone_end: float) -> ZoneAnalysis:
        """Analyze a specific zone in the movement data"""
        try:
            zone_start_idx = None
            zone_end_idx = None
            
            for i, d in enumerate(data):
                if d >= takeoff_point - zone_start and zone_start_idx is None:
                    zone_start_idx = i
                if d >= takeoff_point - zone_end:
                    zone_end_idx = i
                    break
                    
            if zone_start_idx is None or zone_end_idx is None or zone_start_idx >= zone_end_idx:
                return None
                
            zone_data = data[zone_start_idx:zone_end_idx]
            
            if len(zone_data) < 2:
                return None
            
            velocities = self.analyzer.calculate_velocity(zone_data)
            valid_velocities = [v for v in velocities if not np.isnan(v)]
            
            if len(valid_velocities) < 2:
                return None
            
            accelerations = np.diff(valid_velocities)
            step_lengths = np.diff(zone_data)
            
            gaps = []
            for gap in self.analyzer.check_for_gaps(data, takeoff_point):
                if zone_start_idx <= gap[0] <= zone_end_idx:
                    gaps.append({
                        'index': gap[0],
                        'start_value': gap[1],
                        'end_value': gap[2],
                        'difference': gap[3]
                    })
            
            return ZoneAnalysis(
                mean_velocity=np.mean(valid_velocities),
                velocity_std=np.std(valid_velocities),
                step_length_mean=np.mean(step_lengths),
                step_length_std=np.std(step_lengths),
                acceleration_mean=np.mean(accelerations) if len(accelerations) > 0 else 0,
                acceleration_std=np.std(accelerations) if len(accelerations) > 0 else 0,
                data_points=len(zone_data),
                gaps=gaps
            )
        except Exception as e:
            print(f"Error in analyze_zone: {e}")
            return None

    def generate_interactive_plots(self, data, takeoff_point, athlete_name, attempt_num, 
                                   interpolated_data=None, gap_ranges=None, ssa_ranges=None, 
                                   show_ssa=False, gaps=None):
        """Generate interactive plotly figures"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Distanz-Profil', 'Geschwindigkeits-Profil'),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )
        
        # Original data
        fig.add_trace(go.Scatter(
            y=data, 
            name='Messdaten', 
            line=dict(color='blue', width=2),
            hovertemplate='Frame: %{x}<br>Distanz: %{y:.0f} mm<extra></extra>'
        ), row=1, col=1)
        
        # SSA interpolation areas
        if show_ssa and ssa_ranges:
            for (start, end) in ssa_ranges:
                fig.add_vrect(
                    x0=start, x1=end, 
                    fillcolor='#a259d9', 
                    opacity=0.2, 
                    line_width=0, 
                    row=1, col=1
                )
            fig.add_trace(go.Scatter(
                x=[None], y=[None], 
                mode='markers',
                marker=dict(size=10, color='#a259d9'),
                name='Interpoliert (SSA)',
                showlegend=True
            ), row=1, col=1)
        
        # Takeoff line
        fig.add_hline(
            y=takeoff_point, 
            line_dash="dash", 
            line_color=OSP_COLORS['red'],
            line_width=2,
            annotation_text=f"Absprung ({takeoff_point/1000:.2f}m)",
            annotation_position="right",
            row=1, col=1
        )
        
        # Critical zones
        zone_11_6_start = takeoff_point - 11000
        zone_11_6_end = takeoff_point - 6000
        zone_6_1_start = takeoff_point - 6000
        zone_6_1_end = takeoff_point - 1000
        
        fig.add_hrect(
            y0=zone_11_6_start, y1=zone_11_6_end,
            fillcolor=OSP_COLORS['gold'], 
            opacity=0.1,
            line_width=0,
            annotation_text="11-6m Zone",
            annotation_position="left",
            row=1, col=1
        )
        
        fig.add_hrect(
            y0=zone_6_1_start, y1=zone_6_1_end,
            fillcolor=OSP_COLORS['red'], 
            opacity=0.08,
            line_width=0,
            annotation_text="6-1m Zone (kritisch)",
            annotation_position="left",
            row=1, col=1
        )
        
        # Velocity profile
        velocities = self.analyzer.calculate_velocity(data)
        v_plot = np.array(velocities)
        nans = np.isnan(v_plot)
        if np.any(~nans):
            if np.any(nans):
                v_plot[nans] = np.interp(
                    np.flatnonzero(nans), 
                    np.flatnonzero(~nans), 
                    v_plot[~nans]
                )
            
            fig.add_trace(go.Scatter(
                y=v_plot, 
                name='Geschwindigkeit', 
                line=dict(color='green', width=2),
                hovertemplate='Frame: %{x}<br>Geschw.: %{y:.2f} m/s<extra></extra>'
            ), row=2, col=1)
            
            # Mean velocity line
            mean_vel = np.nanmean(velocities)
            fig.add_hline(
                y=mean_vel,
                line_dash="dash",
                line_color='red',
                line_width=1,
                annotation_text=f"Ø {mean_vel:.2f} m/s",
                annotation_position="right",
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{athlete_name} - Versuch {attempt_num}',
            title_font_size=20,
            title_font_color=OSP_COLORS['red'],
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=12)
            ),
            height=800,
            hovermode='x unified',
            plot_bgcolor=OSP_COLORS['white'],
            paper_bgcolor=OSP_COLORS['white'],
            font=dict(family='Arial, sans-serif', size=12)
        )
        
        fig.update_xaxes(title_text='Messpunkt', row=1, col=1, gridcolor=OSP_COLORS['light_gray'])
        fig.update_xaxes(title_text='Messpunkt', row=2, col=1, gridcolor=OSP_COLORS['light_gray'])
        fig.update_yaxes(title_text='Distanz (mm)', row=1, col=1, gridcolor=OSP_COLORS['light_gray'])
        fig.update_yaxes(title_text='Geschwindigkeit (m/s)', row=2, col=1, gridcolor=OSP_COLORS['light_gray'])
        
        return fig

    def setup_dashboard_layout(self):
        """Setup the dashboard layout with 2-column design"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Img(
                    src='https://osp-hessen.de/wp-content/uploads/2021/03/OSP_Logo_2021_RGB.png',
                    style={'height': '60px', 'marginRight': '20px'}
                ),
                html.H1(
                    'Anlaufanalyse Dashboard',
                    style={
                        'color': OSP_COLORS['red'],
                        'fontWeight': 'bold',
                        'fontSize': '2rem',
                        'display': 'inline-block',
                        'verticalAlign': 'middle',
                        'margin': 0
                    }
                ),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'background': OSP_COLORS['white'],
                'padding': '15px 30px',
                'borderBottom': f'4px solid {OSP_COLORS["gold"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
            }),
            
            # Main content: 2-column layout
            html.Div([
                # Left column: File list (30%)
                html.Div([
                    html.H3('Versuche', style={
                        'color': OSP_COLORS['black'],
                        'marginBottom': '15px',
                        'fontSize': '1.3rem'
                    }),
                    dash_table.DataTable(
                        id='file-table',
                        columns=[
                            {"name": "Athlet", "id": "athlete"},
                            {"name": "Versuch", "id": "attempt"},
                            {"name": "Lücken", "id": "gaps"},
                            {"name": "Qualität", "id": "quality"},
                        ],
                        data=self.file_list,
                        style_table={
                            'overflowY': 'auto',
                            'height': 'calc(100vh - 200px)',
                            'borderRadius': '8px'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '12px',
                            'fontSize': '0.95rem',
                            'fontFamily': 'Arial, sans-serif'
                        },
                        style_header={
                            'backgroundColor': OSP_COLORS['gold'],
                            'color': OSP_COLORS['black'],
                            'fontWeight': 'bold',
                            'fontSize': '1rem',
                            'position': 'sticky',
                            'top': 0,
                            'zIndex': 1
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{quality} = "Kritisch"'},
                                'backgroundColor': '#f8d7da',
                                'color': 'black'
                            },
                            {
                                'if': {'filter_query': '{quality} = "Achtung"'},
                                'backgroundColor': '#fff3cd',
                                'color': 'black'
                            },
                            {
                                'if': {'filter_query': '{quality} = "Sehr gut"'},
                                'backgroundColor': '#d4edda',
                                'color': 'black'
                            },
                            {
                                'if': {'state': 'selected'},
                                'backgroundColor': OSP_COLORS['light_gray'],
                                'border': f'2px solid {OSP_COLORS["red"]}'
                            }
                        ],
                        sort_action='native',
                        filter_action='native',
                        row_selectable='single',
                        selected_rows=[0],
                        page_action='none'
                    )
                ], style={
                    'width': '30%',
                    'padding': '20px',
                    'backgroundColor': OSP_COLORS['gray'],
                    'borderRight': f'2px solid {OSP_COLORS["light_gray"]}'
                }),
                
                # Right column: Details (70%)
                html.Div([
                    html.Div(id='detail-view')
                ], style={
                    'width': '70%',
                    'padding': '20px',
                    'overflowY': 'auto',
                    'height': 'calc(100vh - 120px)'
                })
            ], style={
                'display': 'flex',
                'height': 'calc(100vh - 100px)'
            })
        ], style={
            'fontFamily': 'Arial, sans-serif',
            'background': OSP_COLORS['white'],
            'minHeight': '100vh',
            'margin': 0,
            'padding': 0
        })

        @self.app.callback(
            Output('detail-view', 'children'),
            [Input('file-table', 'selected_rows'),
             Input('file-table', 'data')]
        )
        def update_detail_view(selected_rows, table_data):
            if not selected_rows or not table_data:
                return html.Div("Wähle einen Versuch aus der Liste", style={
                    'textAlign': 'center',
                    'padding': '50px',
                    'color': OSP_COLORS['gray'],
                    'fontSize': '1.2rem'
                })
            
            selected_file = table_data[selected_rows[0]]
            filepath = selected_file['filepath']
            
            try:
                # Load data
                takeoff_point, data, athlete_name, attempt_num = self.analyzer.read_data_file(filepath)
                gap_analysis = self.analyzer.analyze_gaps_until_takeoff(filepath)
                gaps = gap_analysis['gaps']
                
                # SSA interpolation
                ssa_filled, ssa_ranges = self.analyzer.fill_gaps_with_ssa(data, gaps)
                show_ssa = bool(ssa_ranges)
                
                # Zones
                zone_11_6 = self.analyze_zone(data, takeoff_point, 11000, 6000)
                zone_6_1 = self.analyze_zone(data, takeoff_point, 6000, 1000)
                
                # Quality metrics
                quality_metrics = QualityMetrics(
                    interpolation_quality=self.calculate_interpolation_quality(data, ssa_filled, ssa_ranges) if show_ssa else 1.0,
                    data_quality_score=self.calculate_data_quality(data, takeoff_point),
                    technical_stability=self.calculate_technical_stability(data),
                    noise_level=np.std(np.diff(data)),
                    gap_count=len(gaps),
                    critical_zone_gaps=len(zone_6_1.gaps) if zone_6_1 else 0
                )
                
                # Generate plot
                fig = self.generate_interactive_plots(
                    data, takeoff_point, athlete_name, attempt_num,
                    interpolated_data=ssa_filled, gap_ranges=ssa_ranges,
                    ssa_ranges=ssa_ranges, show_ssa=show_ssa, gaps=gaps
                )
                
                # Status badge
                status = selected_file['status']
                if status == 'rot':
                    badge_color = OSP_COLORS['red']
                    badge_text = '🔴 KRITISCH'
                elif status == 'gelb':
                    badge_color = OSP_COLORS['yellow']
                    badge_text = '🟡 ACHTUNG'
                else:
                    badge_color = OSP_COLORS['green']
                    badge_text = '🟢 GUT'
                
                return html.Div([
                    # Status badge
                    html.Div([
                        html.Span(badge_text, style={
                            'background': badge_color,
                            'color': 'white',
                            'padding': '8px 20px',
                            'borderRadius': '20px',
                            'fontWeight': 'bold',
                            'fontSize': '1.1rem',
                            'marginRight': '15px'
                        }),
                        html.Span(f"Sampling Rate: {self.analyzer.sampling_rate} Hz", style={
                            'fontSize': '1rem',
                            'color': OSP_COLORS['black']
                        })
                    ], style={'marginBottom': '20px'}),
                    
                    # Plot
                    dcc.Graph(figure=fig, style={'marginBottom': '20px'}),
                    
                    # Metrics in 3 columns
                    html.Div([
                        # Column 1: Quality Metrics
                        html.Div([
                            html.H4('Qualitätsmetriken', style={'color': OSP_COLORS['red'], 'marginBottom': '10px'}),
                            html.P(f'Interpolationsgüte: {quality_metrics.interpolation_quality:.2%}'),
                            html.P(f'Datenqualität: {quality_metrics.data_quality_score:.2%}'),
                            html.P(f'Technische Stabilität: {quality_metrics.technical_stability:.2%}'),
                            html.P(f'Rauschlevel: {quality_metrics.noise_level:.1f} mm'),
                            html.P(f'Anzahl Lücken: {quality_metrics.gap_count}'),
                            html.P(f'Kritische Lücken: {quality_metrics.critical_zone_gaps}')
                        ], style={
                            'flex': '1',
                            'padding': '15px',
                            'backgroundColor': OSP_COLORS['gray'],
                            'borderRadius': '8px',
                            'marginRight': '10px'
                        }),
                        
                        # Column 2: Zone 11-6m
                        html.Div([
                            html.H4('Zone 11-6m', style={'color': OSP_COLORS['gold'], 'marginBottom': '10px'}),
                            html.P(f'Ø Geschwindigkeit: {zone_11_6.mean_velocity:.2f} m/s') if zone_11_6 else html.P('Keine Daten'),
                            html.P(f'Schrittlänge: {zone_11_6.step_length_mean:.1f} mm') if zone_11_6 else None,
                            html.P(f'Beschleunigung: {zone_11_6.acceleration_mean:.2f} m/s²') if zone_11_6 else None,
                            html.P(f'Lücken: {len(zone_11_6.gaps)}') if zone_11_6 else None
                        ], style={
                            'flex': '1',
                            'padding': '15px',
                            'backgroundColor': OSP_COLORS['gray'],
                            'borderRadius': '8px',
                            'marginRight': '10px'
                        }),
                        
                        # Column 3: Zone 6-1m
                        html.Div([
                            html.H4('Zone 6-1m', style={'color': OSP_COLORS['red'], 'marginBottom': '10px'}),
                            html.P(f'Ø Geschwindigkeit: {zone_6_1.mean_velocity:.2f} m/s') if zone_6_1 else html.P('Keine Daten'),
                            html.P(f'Schrittlänge: {zone_6_1.step_length_mean:.1f} mm') if zone_6_1 else None,
                            html.P(f'Beschleunigung: {zone_6_1.acceleration_mean:.2f} m/s²') if zone_6_1 else None,
                            html.P(f'Lücken: {len(zone_6_1.gaps)}') if zone_6_1 else None
                        ], style={
                            'flex': '1',
                            'padding': '15px',
                            'backgroundColor': OSP_COLORS['gray'],
                            'borderRadius': '8px'
                        })
                    ], style={'display': 'flex', 'marginBottom': '20px'}),
                    
                    # Gaps table
                    self.create_gap_table(gaps) if gaps else html.Div()
                ])
                
            except Exception as e:
                return html.Div(f"Fehler beim Laden: {str(e)}", style={
                    'color': 'red',
                    'padding': '20px'
                })

    def create_gap_table(self, gaps):
        """Create a table showing gap details"""
        if not gaps:
            return html.Div()
        
        gap_data = []
        for g in gaps:
            zone = ''
            if g.get('zone_6_1'):
                zone = '🔴 6-1m (kritisch)'
            elif g.get('zone_11_6'):
                zone = '🟡 11-6m'
            else:
                zone = '-'
            
            gap_data.append({
                'Index': g['index'],
                'Start': f"{g['start_value']:.0f} mm",
                'Ende': f"{g['end_value']:.0f} mm",
                'Größe': f"{g['difference']:.0f} mm",
                'Zone': zone
            })
        
        return html.Div([
            html.H4('Lücken-Details', style={'color': OSP_COLORS['red'], 'marginTop': '20px', 'marginBottom': '10px'}),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in ['Index', 'Start', 'Ende', 'Größe', 'Zone']],
                data=gap_data,
                style_table={'borderRadius': '8px'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontSize': '0.95rem'
                },
                style_header={
                    'backgroundColor': OSP_COLORS['gold'],
                    'color': OSP_COLORS['black'],
                    'fontWeight': 'bold'
                },
                page_size=10
            )
        ])

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        print(f"\n🚀 Dashboard startet auf http://localhost:{port}")
        print(f"📊 {len(self.file_list)} Dateien geladen")
        print(f"🎯 Optimiert für FHD (1920x1080)\n")
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    from analyze_movement_data import MovementDataAnalyzer
    
    # Initialize analyzer
    folders = [
        os.path.join("Input files/Drei M"),
        os.path.join("Input files/Drei W"),
        os.path.join("Input files/Weit M"),
        os.path.join("Input files/Weit W")
    ]
    analyzer = MovementDataAnalyzer(folders)
    
    # Create and run dashboard
    dashboard = MovementAnalysisDashboard(analyzer)
    dashboard.run_server()



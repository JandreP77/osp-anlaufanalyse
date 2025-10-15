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
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import json
from pathlib import Path
import base64
import io
import dash_daq as daq

OSP_COLORS = {
    'red': '#e30613',
    'black': '#000000',
    'white': '#ffffff',
    'gold': '#ffd700',
    'gray': '#f5f5f5',
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
        self.file_options = self.get_file_options()
        self.uploaded_files = {}
        self.app = dash.Dash(__name__)
        self.setup_dashboard_layout()
        
    def get_file_options(self):
        """Scan input folders for .dat files and return dropdown options."""
        options = []
        for folder in self.analyzer.folders:
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith('.dat'):
                    label = f"{os.path.basename(folder)} / {fname}"
                    value = os.path.join(folder, fname)
                    options.append({'label': label, 'value': value})
        return sorted(options, key=lambda x: x['label'])

    def calculate_interpolation_quality(self, original_data: List[float], 
                                     interpolated_data: List[float], 
                                     gap_ranges: List[Tuple[int, int]]) -> float:
        """
        Calculate the quality of interpolation based on smoothness and continuity
        
        Args:
            original_data: Original movement data
            interpolated_data: Interpolated movement data
            gap_ranges: List of tuples containing gap ranges
            
        Returns:
            float: Quality score between 0 and 1
        """
        if not gap_ranges:
            return 1.0
            
        # Calculate velocity continuity
        original_velocities = np.diff(original_data)
        interpolated_velocities = np.diff(interpolated_data)
        
        # Calculate smoothness score
        smoothness_score = 1 - np.mean(np.abs(np.diff(interpolated_velocities)))
        
        # Calculate continuity score
        continuity_score = 1 - np.mean([
            abs(interpolated_data[end] - original_data[end+1]) 
            for start, end in gap_ranges
        ]) / np.max(original_data)
        
        return (smoothness_score + continuity_score) / 2

    def calculate_data_quality(self, data: List[float], 
                             takeoff_point: float) -> float:
        """
        Calculate overall data quality score
        
        Args:
            data: Movement data
            takeoff_point: Takeoff point distance
            
        Returns:
            float: Quality score between 0 and 1
        """
        # Calculate noise level
        noise_level = np.std(np.diff(data))
        noise_score = 1 - min(noise_level / 1000, 1)  # Normalize to 0-1
        
        # Calculate gap score
        gaps = self.analyzer.check_for_gaps(data, takeoff_point)
        gap_score = 1 - min(len(gaps) / 5, 1)  # Normalize to 0-1
        
        # Calculate velocity continuity score
        velocities = self.analyzer.calculate_velocity(data)
        velocity_score = 1 - min(np.std(velocities) / 5, 1)  # Normalize to 0-1
        
        return (noise_score + gap_score + velocity_score) / 3

    def calculate_technical_stability(self, data: List[float]) -> float:
        """
        Calculate technical stability score
        
        Args:
            data: Movement data
            
        Returns:
            float: Stability score between 0 and 1
        """
        # Calculate step pattern stability
        step_pattern = self.analyzer.analyze_step_pattern(data)
        step_stability = 1 - min(step_pattern['std_step_size'] / 100, 1)
        
        # Calculate velocity stability
        velocities = self.analyzer.calculate_velocity(data)
        velocity_stability = 1 - min(np.std(velocities) / 5, 1)
        
        # Calculate acceleration stability
        accelerations = np.diff(velocities)
        acceleration_stability = 1 - min(np.std(accelerations) / 2, 1)
        
        return (step_stability + velocity_stability + acceleration_stability) / 3

    def analyze_zone(self, data: List[float], 
                    takeoff_point: float,
                    zone_start: float,
                    zone_end: float) -> ZoneAnalysis:
        """
        Analyze a specific zone in the movement data
        
        Args:
            data: Movement data
            takeoff_point: Takeoff point distance
            zone_start: Start of zone (distance from takeoff)
            zone_end: End of zone (distance from takeoff)
            
        Returns:
            ZoneAnalysis object containing zone metrics
        """
        # Convert distances to indices
        zone_start_idx = None
        zone_end_idx = None
        
        for i, d in enumerate(data):
            if d >= takeoff_point - zone_start and zone_start_idx is None:
                zone_start_idx = i
            if d >= takeoff_point - zone_end:
                zone_end_idx = i
                break
                
        if zone_start_idx is None or zone_end_idx is None:
            return None
            
        # Extract zone data
        zone_data = data[zone_start_idx:zone_end_idx]
        
        # Calculate metrics
        velocities = self.analyzer.calculate_velocity(zone_data)
        accelerations = np.diff(velocities)
        
        # Calculate step lengths
        step_lengths = np.diff(zone_data)
        
        # Find gaps in zone
        gaps = []
        for gap in self.analyzer.check_for_gaps(data, takeoff_point):
            if (zone_start_idx <= gap[0] <= zone_end_idx):
                gaps.append({
                    'index': gap[0],
                    'start_value': gap[1],
                    'end_value': gap[2],
                    'difference': gap[3]
                })
        
        return ZoneAnalysis(
            mean_velocity=np.mean(velocities),
            velocity_std=np.std(velocities),
            step_length_mean=np.mean(step_lengths),
            step_length_std=np.std(step_lengths),
            acceleration_mean=np.mean(accelerations),
            acceleration_std=np.std(accelerations),
            data_points=len(zone_data),
            gaps=gaps
        )

    def generate_interactive_plots(self, data, takeoff_point, interpolated_data=None, gap_ranges=None, ssa_ranges=None, show_ssa=False):
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Movement Profile', 'Velocity Profile'))
        # Original data
        fig.add_trace(go.Scatter(y=data, name='Originaldaten', line=dict(color='blue')), row=1, col=1)
        # Interpolationsbereiche als lila Balken
        if show_ssa and ssa_ranges:
            for (start, end) in ssa_ranges:
                fig.add_vrect(x0=start, x1=end, fillcolor='#a259d9', opacity=0.18, line_width=0, row=1, col=1)
            # Dummy-Trace für Legende
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                     line=dict(color='#a259d9', width=10),
                                     name='Interpolationsbereich (SSA)',
                                     showlegend=True), row=1, col=1)
        # Takeoff line
        fig.add_shape(type='line', y0=takeoff_point, y1=takeoff_point, x0=0, x1=len(data), line=dict(color=OSP_COLORS['red'], dash='dash'), row=1, col=1)
        # Critical zones
        zone_11_6_start = takeoff_point - 11000
        zone_11_6_end = takeoff_point - 6000
        zone_6_1_start = takeoff_point - 6000
        zone_6_1_end = takeoff_point - 1000
        fig.add_shape(type='rect', y0=zone_11_6_start, y1=zone_11_6_end, x0=0, x1=len(data), fillcolor=OSP_COLORS['gold'], opacity=0.12, row=1, col=1)
        fig.add_shape(type='rect', y0=zone_6_1_start, y1=zone_6_1_end, x0=0, x1=len(data), fillcolor=OSP_COLORS['red'], opacity=0.08, row=1, col=1)
        # Velocity profile (continuous)
        velocities = self.analyzer.calculate_velocity(data)
        v_plot = np.array(velocities)
        nans = np.isnan(v_plot)
        if np.any(nans):
            v_plot[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), v_plot[~nans])
        fig.add_trace(go.Scatter(y=v_plot, name='Geschwindigkeit', line=dict(color='green')), row=2, col=1)
        fig.update_layout(
            title='',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=15)),
            height=700,
            hovermode='x unified',
            plot_bgcolor=OSP_COLORS['gray'],
            paper_bgcolor=OSP_COLORS['gray'],
            font=dict(family='Inter, Arial, sans-serif', size=16)
        )
        return fig

    def parse_uploaded_file(self, contents, filename):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            s = decoded.decode('utf-8')
        except UnicodeDecodeError:
            s = decoded.decode('latin1')
        lines = s.splitlines()
        # Save to temp file for compatibility
        temp_path = f"uploaded_{filename}"
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(s)
        return temp_path

    def get_all_attempts_summary(self):
        rows = []
        for opt in self.file_options:
            fpath = opt['value']
            try:
                takeoff_point, data, athlete_name, attempt_num = self.analyzer.read_data_file(fpath)
                gaps = self.analyzer.check_for_gaps(data, takeoff_point)
                zone_11_6 = self.analyze_zone(data, takeoff_point, 11000, 6000)
                zone_6_1 = self.analyze_zone(data, takeoff_point, 6000, 1000)
                velocities = self.analyzer.calculate_velocity(data)
                rows.append({
                    'Datei': os.path.basename(fpath),
                    'Athlet': athlete_name,
                    'Versuch': attempt_num,
                    'Lücken': len(gaps),
                    'Lücken 11-6m': len(zone_11_6.gaps) if zone_11_6 else 0,
                    'Lücken 6-1m': len(zone_6_1.gaps) if zone_6_1 else 0,
                    'Durchschnittsgeschw.': np.nanmean(velocities),
                    'Max. Geschw.': np.nanmax(velocities),
                    'Qualität': 'gut' if len(gaps)==0 else ('kritisch' if len(zone_6_1.gaps)>0 else 'ok'),
                })
            except Exception as e:
                continue
        return rows

    def gap_table(self, gaps, takeoff_point):
        if not gaps:
            return html.Div([
                html.Span('Keine Lücken vorhanden.', style={'color': 'white', 'background': '#28a745', 'padding': '8px 18px', 'borderRadius': '20px', 'fontWeight': 'bold', 'fontSize': '1.1rem', 'marginTop': '10px', 'display': 'inline-block'})
            ])
        rows = []
        for g in gaps:
            zone = '6-1m' if g.get('zone_6_1') else ('11-6m' if g.get('zone_11_6') else '-')
            rows.append({
                'Start': g['start_value'],
                'Ende': g['end_value'],
                'Index': g['index'],
                'Größe (mm)': g['difference'],
                'Zone': zone
            })
        return html.Div([
            html.H5('Lücken & Interpolationen', style={'color': OSP_COLORS['red'], 'marginBottom': '8px'}),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in ['Index', 'Start', 'Ende', 'Größe (mm)', 'Zone']],
                data=rows,
                style_table={'overflowX': 'auto', 'background': OSP_COLORS['white'], 'borderRadius': '10px'},
                style_cell={'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '1.1rem'},
                style_header={'backgroundColor': OSP_COLORS['gold'], 'color': OSP_COLORS['black'], 'fontWeight': 'bold'},
                page_size=10
            )
        ])

    def status_badge(self, gaps, zone_6_1, zone_11_6):
        if not gaps:
            color = '#28a745'
            text = 'Keine Lücken'
        elif zone_6_1 and len(zone_6_1.gaps) > 0:
            color = OSP_COLORS['red']
            text = 'Kritisch: Lücke in 6-1m'
        elif zone_11_6 and len(zone_11_6.gaps) > 0:
            color = OSP_COLORS['gold']
            text = 'Achtung: Lücke in 11-6m'
        else:
            color = '#ffc107'
            text = 'Lücken interpoliert'
        return html.Span(text, style={'background': color, 'color': 'white', 'padding': '8px 18px', 'borderRadius': '20px', 'fontWeight': 'bold', 'fontSize': '1.1rem', 'marginRight': '10px', 'boxShadow': '0 2px 8px #00000010'})

    def get_overview_statistics(self, rows):
        if not rows:
            return {}
        num_versuche = len(rows)
        num_luecken = sum(r['Lücken'] for r in rows)
        num_luecken_11_6 = sum(r['Lücken 11-6m'] for r in rows)
        num_luecken_6_1 = sum(r['Lücken 6-1m'] for r in rows)
        max_speed = max(r['Max. Geschw.'] for r in rows)
        avg_speed = sum(r['Durchschnittsgeschw.'] for r in rows) / num_versuche
        return {
            'Versuche': num_versuche,
            'Gesamt-Lücken': num_luecken,
            'Lücken 11-6m': num_luecken_11_6,
            'Lücken 6-1m': num_luecken_6_1,
            'Ø Geschwindigkeit': f"{avg_speed:.2f} m/s",
            'Max. Geschwindigkeit': f"{max_speed:.2f} m/s"
        }

    def setup_dashboard_layout(self):
        self.app.layout = html.Div([
            dcc.Tabs(id='tabs', value='tab-einzel', children=[
                dcc.Tab(label='Einzelanalyse', value='tab-einzel', style={'fontWeight': 'bold', 'color': OSP_COLORS['red']}),
                dcc.Tab(label='Übersicht', value='tab-uebersicht', style={'fontWeight': 'bold', 'color': OSP_COLORS['black']}),
            ], style={'marginBottom': '0'}),
            html.Div(id='tab-content')
        ], style={'fontFamily': 'Inter, Arial, sans-serif', 'background': OSP_COLORS['gray'], 'minHeight': '100vh', 'padding': '0 0 40px 0'})

        @self.app.callback(Output('tab-content', 'children'), [Input('tabs', 'value')])
        def render_content(tab):
            if tab == 'tab-einzel':
                return html.Div([
                    html.Div([
                        html.Img(src='https://osp-hessen.de/wp-content/uploads/2021/03/OSP_Logo_2021_RGB.png', style={'height': '70px', 'marginRight': '24px', 'borderRadius': '10px', 'boxShadow': '0 2px 8px #00000010'}),
                        html.H1('Movement Analysis Dashboard', style={'color': OSP_COLORS['red'], 'fontWeight': 'bold', 'fontSize': '2.2rem', 'display': 'inline-block', 'verticalAlign': 'middle', 'margin': 0}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'background': OSP_COLORS['white'], 'padding': '18px 0 8px 0', 'borderBottom': f'4px solid {OSP_COLORS["gold"]}', 'borderRadius': '0 0 18px 18px', 'boxShadow': '0 2px 8px #00000010'}),
                    html.Div([
                        html.H3('Datei auswählen', style={'color': OSP_COLORS['black'], 'marginBottom': '8px'}),
                        dcc.Dropdown(
                            id='file-dropdown',
                            options=self.file_options,
                            value=self.file_options[0]['value'] if self.file_options else None,
                            placeholder='Wähle eine Messdatei...',
                            style={'backgroundColor': OSP_COLORS['gray'], 'color': OSP_COLORS['black'], 'fontWeight': 'bold', 'fontSize': '1.1rem', 'borderRadius': '10px', 'boxShadow': '0 2px 8px #00000010'}
                        ),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(['Datei hochladen']),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '2px', 'borderStyle': 'dashed',
                                'borderRadius': '10px', 'textAlign': 'center', 'margin': '20px 0', 'background': OSP_COLORS['white'], 'color': OSP_COLORS['red'], 'fontWeight': 'bold', 'fontSize': '1.1rem', 'boxShadow': '0 2px 8px #00000010'
                            },
                            multiple=False
                        )
                    ], style={'margin': '30px 0 20px 0', 'padding': '20px', 'background': OSP_COLORS['gray'], 'borderRadius': '18px', 'boxShadow': '0 2px 8px #00000010'}),
                    html.Div(id='info-box'),
                    # Badge für Lücken interpoliert
                    html.Div(id='ssa-badge', style={'margin': '18px 0 0 0'}),
                    html.Div([
                        dcc.Graph(id='movement-plot', style={'backgroundColor': OSP_COLORS['white'], 'borderRadius': '18px', 'boxShadow': '0 2px 8px #00000010'}),
                        html.Div(id='gap-table', style={'marginTop': '18px'}),
                        html.Div(id='quality-metrics', style={'marginTop': '18px', 'background': OSP_COLORS['white'], 'padding': '18px', 'borderRadius': '14px', 'boxShadow': '0 2px 8px #00000010'}),
                        html.Div(id='zone-analysis', style={'marginTop': '18px', 'background': OSP_COLORS['white'], 'padding': '18px', 'borderRadius': '14px', 'boxShadow': '0 2px 8px #00000010'})
                    ], style={'margin': '20px 0'}),
                    html.Div([
                        html.H3('Export-Optionen', style={'color': OSP_COLORS['black']}),
                        html.Button('Export PDF Report', id='export-pdf', style={'backgroundColor': OSP_COLORS['gold'], 'color': OSP_COLORS['black'], 'fontWeight': 'bold', 'marginRight': '10px', 'border': 'none', 'borderRadius': '8px', 'padding': '12px 22px', 'fontSize': '1.1rem', 'boxShadow': '0 2px 8px #00000010'}),
                        html.Button('Export Excel', id='export-excel', style={'backgroundColor': OSP_COLORS['red'], 'color': OSP_COLORS['white'], 'fontWeight': 'bold', 'marginRight': '10px', 'border': 'none', 'borderRadius': '8px', 'padding': '12px 22px', 'fontSize': '1.1rem', 'boxShadow': '0 2px 8px #00000010'}),
                        html.Button('Export Presentation', id='export-presentation', style={'backgroundColor': OSP_COLORS['black'], 'color': OSP_COLORS['gold'], 'fontWeight': 'bold', 'border': 'none', 'borderRadius': '8px', 'padding': '12px 22px', 'fontSize': '1.1rem', 'boxShadow': '0 2px 8px #00000010'})
                    ], style={'margin': '30px 0', 'padding': '20px', 'background': OSP_COLORS['gray'], 'borderRadius': '18px', 'boxShadow': '0 2px 8px #00000010'})
                ], style={'maxWidth': '1100px', 'margin': '0 auto'})
            elif tab == 'tab-uebersicht':
                rows = self.get_all_attempts_summary()
                stats = self.get_overview_statistics(rows)
                stats_table = dash_table.DataTable(
                    columns=[{"name": k, "id": k} for k in stats.keys()],
                    data=[stats],
                    style_table={'marginTop': '30px', 'background': OSP_COLORS['white'], 'borderRadius': '14px', 'boxShadow': '0 2px 8px #00000010'},
                    style_cell={'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '1.1rem'},
                    style_header={'backgroundColor': OSP_COLORS['gold'], 'color': OSP_COLORS['black'], 'fontWeight': 'bold'},
                )
                return html.Div([
                    html.H2('Übersicht aller Versuche', style={'color': OSP_COLORS['red'], 'marginTop': '30px', 'fontWeight': 'bold', 'fontSize': '2rem'}),
                    dash_table.DataTable(
                        id='overview-table',
                        columns=[{"name": i, "id": i} for i in rows[0].keys()] if rows else [],
                        data=rows,
                        style_table={'overflowX': 'auto', 'background': OSP_COLORS['white'], 'borderRadius': '14px', 'boxShadow': '0 2px 8px #00000010'},
                        style_cell={'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '1.1rem'},
                        style_header={'backgroundColor': OSP_COLORS['gold'], 'color': OSP_COLORS['black'], 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{Qualität} = gut'}, 'backgroundColor': '#d4edda', 'color': 'black'},
                            {'if': {'filter_query': '{Qualität} = kritisch'}, 'backgroundColor': '#f8d7da', 'color': 'black'},
                            {'if': {'filter_query': '{Qualität} = ok'}, 'backgroundColor': '#fff3cd', 'color': 'black'},
                        ],
                        sort_action='native',
                        filter_action='native',
                        page_size=20
                    ),
                    html.H3('Gesamtstatistiken', style={'color': OSP_COLORS['black'], 'marginTop': '40px'}),
                    stats_table
                ], style={'padding': '30px', 'maxWidth': '1100px', 'margin': '0 auto'})

        @self.app.callback(
            [Output('movement-plot', 'figure'),
             Output('gap-table', 'children'),
             Output('info-box', 'children'),
             Output('quality-metrics', 'children'),
             Output('zone-analysis', 'children'),
             Output('ssa-badge', 'children')],
            [Input('file-dropdown', 'value')]
        )
        def update_dashboard(selected_file):
            if selected_file is None:
                return {}, '', '', '', '', ''
            gap_analysis = self.analyzer.analyze_gaps_until_takeoff(selected_file)
            takeoff_point = gap_analysis['takeoff_point']
            data = self.analyzer.read_data_file(selected_file)[1]
            athlete_name = gap_analysis['athlete']
            attempt_num = gap_analysis['attempt']
            gaps = gap_analysis['gaps']
            ssa_filled, ssa_ranges = self.analyzer.fill_gaps_with_ssa(data, gaps)
            zone_11_6 = self.analyze_zone(data, takeoff_point, 11000, 6000)
            zone_6_1 = self.analyze_zone(data, takeoff_point, 6000, 1000)
            show_ssa = bool(ssa_ranges)
            fig = self.generate_interactive_plots(data, takeoff_point, interpolated_data=ssa_filled, gap_ranges=ssa_ranges, ssa_ranges=ssa_ranges, show_ssa=show_ssa)
            # Info box
            badge = self.status_badge(gaps, zone_6_1, zone_11_6)
            if not gaps:
                info = html.Div([
                    badge,
                    html.Span("Keine Lücken, keine Interpolation notwendig.", style={'fontWeight': 'bold', 'fontSize': '1.1rem', 'marginLeft': '18px', 'color': '#28a745'})
                ], style={'background': OSP_COLORS['white'], 'padding': '16px', 'borderRadius': '14px', 'margin': '18px 0 0 0', 'boxShadow': '0 2px 8px #00000010', 'display': 'flex', 'alignItems': 'center'})
                ssa_badge = ''
            else:
                info = html.Div([
                    badge,
                    html.Span(f"Athlet: {athlete_name} | Versuch: {attempt_num}", style={'fontWeight': 'bold', 'fontSize': '1.1rem', 'marginRight': '18px'}),
                    html.Span(f"Takeoff: {takeoff_point:.0f} mm", style={'fontSize': '1.1rem', 'color': OSP_COLORS['black']})
                ], style={'background': OSP_COLORS['white'], 'padding': '16px', 'borderRadius': '14px', 'margin': '18px 0 0 0', 'boxShadow': '0 2px 8px #00000010', 'display': 'flex', 'alignItems': 'center'})
                ssa_badge = html.Span(f"{len(ssa_ranges)} Lücke(n) interpoliert", style={'background': '#7ec8e3', 'color': OSP_COLORS['black'], 'padding': '8px 18px', 'borderRadius': '20px', 'fontWeight': 'bold', 'fontSize': '1.1rem', 'marginRight': '10px', 'boxShadow': '0 2px 8px #00000010', 'display': 'inline-block'})
            # Gap table
            gap_table = self.gap_table(gaps, takeoff_point)
            # Quality metrics
            quality_metrics = QualityMetrics(
                interpolation_quality=self.calculate_interpolation_quality(data, ssa_filled, ssa_ranges) if show_ssa else 1.0,
                data_quality_score=self.calculate_data_quality(data, takeoff_point),
                technical_stability=self.calculate_technical_stability(data),
                noise_level=np.std(np.diff(data)),
                gap_count=len(gaps),
                critical_zone_gaps=len(zone_6_1.gaps) if zone_6_1 else 0
            )
            metrics_display = html.Div([
                html.H4('Qualitätsmetriken', style={'color': OSP_COLORS['black']}),
                html.P(f'Interpolationsgüte: {quality_metrics.interpolation_quality:.2f}'),
                html.P(f'Datenqualität: {quality_metrics.data_quality_score:.2f}'),
                html.P(f'Technische Stabilität: {quality_metrics.technical_stability:.2f}'),
                html.P(f'Rauschlevel: {quality_metrics.noise_level:.2f} mm'),
                html.P(f'Anzahl Lücken: {quality_metrics.gap_count}'),
                html.P(f'Lücken in kritischen Zonen: {quality_metrics.critical_zone_gaps}')
            ])
            # Zone analysis
            zone_display = html.Div([
                html.H4('Zonenanalyse', style={'color': OSP_COLORS['black']}),
                html.H5('11-6m Zone', style={'color': OSP_COLORS['gold']}),
                html.P(f'Mittlere Geschwindigkeit: {zone_11_6.mean_velocity:.2f} m/s'),
                html.P(f'Schrittlänge: {zone_11_6.step_length_mean:.2f} mm'),
                html.P(f'Beschleunigung: {zone_11_6.acceleration_mean:.2f} m/s²'),
                html.H5('6-1m Zone', style={'color': OSP_COLORS['red']}),
                html.P(f'Mittlere Geschwindigkeit: {zone_6_1.mean_velocity:.2f} m/s'),
                html.P(f'Schrittlänge: {zone_6_1.step_length_mean:.2f} mm'),
                html.P(f'Beschleunigung: {zone_6_1.acceleration_mean:.2f} m/s²')
            ])
            return fig, gap_table, info, metrics_display, zone_display, ssa_badge

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
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
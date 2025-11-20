"""
Hybrid SSA Interpolator for Movement Data
Combines global velocity patterns with individual step patterns
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pyts.decomposition import SingularSpectrumAnalysis
from collections import defaultdict
import pickle


class HybridSSAInterpolator:
    """
    Hybrid SSA Interpolation combining:
    - Global velocity model (trained on all athletes)
    - Individual step pattern models (trained per athlete)
    - Similarity groups (fallback for athletes with insufficient data)
    """
    
    def __init__(self, window_size: int = 40):
        """
        Initialize the Hybrid SSA Interpolator
        
        Args:
            window_size: SSA window size (default 40 frames)
        """
        self.window_size = window_size
        self.global_velocity_model = None
        self.athlete_step_models = {}
        self.similarity_group_models = {}
        self.athlete_groups = {}  # Maps athlete -> group
        self.min_points_for_individual_model = 500
        
    def determine_group(self, folder: str) -> str:
        """Determine athlete group from folder name"""
        folder_lower = folder.lower()
        if 'drei' in folder_lower and 'm' in folder_lower:
            return 'Drei_M'
        elif 'drei' in folder_lower and 'w' in folder_lower:
            return 'Drei_W'
        elif 'weit' in folder_lower and 'm' in folder_lower:
            return 'Weit_M'
        elif 'weit' in folder_lower and 'w' in folder_lower:
            return 'Weit_W'
        return 'Unknown'
    
    def train(self, analyzer, folders: List[str]):
        """
        Train the hybrid model on all available data
        
        Args:
            analyzer: MovementDataAnalyzer instance
            folders: List of folder paths containing .dat files
        """
        print("\n" + "="*70)
        print("🚀 TRAINING HYBRID SSA INTERPOLATOR")
        print("="*70)
        
        # Collect data
        all_velocities = []
        athlete_data = defaultdict(lambda: {'velocities': [], 'distances': [], 'steps': [], 'files': 0})
        group_data = defaultdict(lambda: {'velocities': [], 'distances': [], 'steps': []})
        
        print("\n📊 Collecting training data...")
        total_files = 0
        
        for folder in folders:
            if not os.path.exists(folder):
                continue
                
            group = self.determine_group(folder)
            
            for fname in os.listdir(folder):
                if fname.lower().endswith('.dat'):
                    fpath = os.path.join(folder, fname)
                    try:
                        takeoff_point, data, athlete_name, attempt_num = analyzer.read_data_file(fpath)
                        gap_analysis = analyzer.analyze_gaps_until_takeoff(fpath)
                        
                        # Only use clean data (no gaps) for training
                        if gap_analysis['number_of_gaps'] == 0:
                            total_files += 1
                            
                            # Calculate velocity
                            velocities = analyzer.calculate_velocity(data)
                            valid_velocities = [v for v in velocities if not np.isnan(v)]
                            
                            # Calculate step pattern
                            steps = np.diff(data)
                            valid_steps = [s for s in steps if abs(s) < 500]  # Filter outliers
                            
                            if len(valid_velocities) > 0 and len(valid_steps) > 0:
                                # Global data
                                all_velocities.extend(valid_velocities)
                                
                                # Per athlete
                                athlete_data[athlete_name]['velocities'].extend(valid_velocities)
                                athlete_data[athlete_name]['distances'].extend(data)
                                athlete_data[athlete_name]['steps'].extend(valid_steps)
                                athlete_data[athlete_name]['files'] += 1
                                
                                # Per group
                                group_data[group]['velocities'].extend(valid_velocities)
                                group_data[group]['distances'].extend(data)
                                group_data[group]['steps'].extend(valid_steps)
                                
                                # Map athlete to group
                                self.athlete_groups[athlete_name] = group
                                
                    except Exception as e:
                        continue
        
        print(f"✅ Collected data from {total_files} clean files")
        print(f"   Total velocity samples: {len(all_velocities):,}")
        print(f"   Unique athletes: {len(athlete_data)}")
        print(f"   Groups: {list(group_data.keys())}")
        
        # 1. Train global velocity model
        print("\n🌍 Training global velocity model...")
        if len(all_velocities) >= self.window_size * 2:
            try:
                velocity_series = np.array(all_velocities[:min(10000, len(all_velocities))])  # Limit for performance
                self.global_velocity_model = SingularSpectrumAnalysis(window_size=self.window_size, groups='auto')
                self.global_velocity_model.fit(velocity_series.reshape(1, -1))
                print(f"✅ Global velocity model trained on {len(velocity_series):,} samples")
            except Exception as e:
                print(f"⚠️  Warning: Could not train global model: {e}")
        
        # 2. Train individual athlete models
        print(f"\n👤 Training individual athlete models (min {self.min_points_for_individual_model} points)...")
        for athlete, data_dict in athlete_data.items():
            total_points = len(data_dict['distances'])
            if total_points >= self.min_points_for_individual_model:
                try:
                    # Train on distance data (preserves step patterns)
                    distances = np.array(data_dict['distances'][:min(5000, len(data_dict['distances']))])
                    model = SingularSpectrumAnalysis(window_size=self.window_size, groups='auto')
                    model.fit(distances.reshape(1, -1))
                    self.athlete_step_models[athlete] = model
                    print(f"   ✅ {athlete}: {total_points:,} points, {data_dict['files']} files")
                except Exception as e:
                    print(f"   ⚠️  {athlete}: Failed ({e})")
        
        print(f"✅ Trained {len(self.athlete_step_models)} individual models")
        
        # 3. Train similarity group models
        print(f"\n🔗 Training similarity group models...")
        for group, data_dict in group_data.items():
            if len(data_dict['distances']) >= self.window_size * 2:
                try:
                    distances = np.array(data_dict['distances'][:min(10000, len(data_dict['distances']))])
                    model = SingularSpectrumAnalysis(window_size=self.window_size, groups='auto')
                    model.fit(distances.reshape(1, -1))
                    self.similarity_group_models[group] = model
                    print(f"   ✅ {group}: {len(data_dict['distances']):,} points")
                except Exception as e:
                    print(f"   ⚠️  {group}: Failed ({e})")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
    
    def interpolate_gap(self, data: List[float], athlete_name: str, gap_start: int, gap_end: int, 
                       gap_size: float) -> Tuple[List[float], float]:
        """
        Interpolate a gap using hybrid approach
        
        Args:
            data: Full distance data
            athlete_name: Name of the athlete
            gap_start: Index before gap
            gap_end: Index after gap
            gap_size: Size of gap in mm
            
        Returns:
            Tuple of (interpolated_values, confidence_score)
        """
        # Calculate how many frames to interpolate
        expected_step_size = 150  # mm (conservative estimate)
        num_frames = max(1, int(gap_size / expected_step_size))
        
        # Confidence score based on gap size
        if gap_size < 1000:  # <1m
            base_confidence = 0.95
        elif gap_size < 3000:  # 1-3m
            base_confidence = 0.80
        elif gap_size < 5000:  # 3-5m
            base_confidence = 0.60
        elif gap_size < 10000:  # 5-10m
            base_confidence = 0.40
        else:  # >10m
            base_confidence = 0.20
        
        # Adjust confidence if we have individual model
        if athlete_name in self.athlete_step_models:
            confidence = base_confidence * 1.1  # Boost by 10%
            model = self.athlete_step_models[athlete_name]
        elif athlete_name in self.athlete_groups and self.athlete_groups[athlete_name] in self.similarity_group_models:
            confidence = base_confidence * 0.9  # Slight penalty
            model = self.similarity_group_models[self.athlete_groups[athlete_name]]
        else:
            # Fallback to simple linear interpolation
            confidence = base_confidence * 0.5
            start_val = data[gap_start]
            end_val = data[gap_end]
            interpolated = np.linspace(start_val, end_val, num_frames + 2)[1:-1]
            return interpolated.tolist(), min(1.0, confidence)
        
        # Use SSA model
        try:
            # Get context around gap
            context_size = self.window_size
            left_idx = max(0, gap_start - context_size)
            right_idx = min(len(data), gap_end + context_size)
            
            segment = np.array(data[left_idx:right_idx])
            
            # Create mask for gap
            gap_relative_start = gap_start - left_idx
            gap_relative_end = gap_end - left_idx
            
            # Simple approach: linear interpolation first, then SSA smoothing
            segment_filled = segment.copy()
            if gap_relative_start >= 0 and gap_relative_end < len(segment_filled):
                # Linear fill first
                gap_length = gap_relative_end - gap_relative_start
                if gap_length > 0:
                    linear_fill = np.linspace(
                        segment[gap_relative_start], 
                        segment[gap_relative_end], 
                        gap_length + 1
                    )[1:]
                    segment_filled[gap_relative_start+1:gap_relative_end] = linear_fill[:gap_length-1]
            
            # Apply SSA reconstruction
            segment_ssa = model.transform(segment_filled.reshape(1, -1))[0]
            
            # Extract interpolated values
            if gap_relative_start >= 0 and gap_relative_end < len(segment_ssa):
                interpolated = segment_ssa[gap_relative_start+1:gap_relative_end]
            else:
                # Fallback
                interpolated = linear_fill[:num_frames]
            
            return interpolated.tolist(), min(1.0, confidence)
            
        except Exception as e:
            # Fallback to linear
            start_val = data[gap_start]
            end_val = data[gap_end]
            interpolated = np.linspace(start_val, end_val, num_frames + 2)[1:-1]
            return interpolated.tolist(), min(0.3, confidence)
    
    def fill_all_gaps(self, data: List[float], athlete_name: str, gaps: List[Dict[str, Any]], 
                     min_confidence: float = 0.3) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Fill all gaps in the data
        
        Args:
            data: Original distance data
            athlete_name: Name of the athlete
            gaps: List of gap dictionaries
            min_confidence: Minimum confidence to perform interpolation (default 0.3)
            
        Returns:
            Tuple of (interpolated_data, interpolation_info)
        """
        data_filled = data.copy()
        interpolation_info = []
        
        for gap in gaps:
            gap_idx = gap['index']
            gap_size = gap['difference']
            gap_size_m = gap_size / 1000
            
            # Interpolate
            interpolated, confidence = self.interpolate_gap(
                data_filled, athlete_name, gap_idx, gap_idx + 1, gap_size
            )
            
            # Only apply if confidence is high enough
            if confidence >= min_confidence:
                # Note: This doesn't actually insert frames, just smooths between existing ones
                # For true gap filling, you'd need to insert into the array
                interpolation_info.append({
                    'index': gap_idx,
                    'size': gap_size_m,
                    'confidence': confidence,
                    'method': 'individual' if athlete_name in self.athlete_step_models else 'group',
                    'interpolated': True
                })
            else:
                interpolation_info.append({
                    'index': gap_idx,
                    'size': gap_size_m,
                    'confidence': confidence,
                    'method': 'none',
                    'interpolated': False
                })
        
        return data_filled, interpolation_info
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        models = {
            'global_velocity_model': self.global_velocity_model,
            'athlete_step_models': self.athlete_step_models,
            'similarity_group_models': self.similarity_group_models,
            'athlete_groups': self.athlete_groups,
            'window_size': self.window_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        self.global_velocity_model = models['global_velocity_model']
        self.athlete_step_models = models['athlete_step_models']
        self.similarity_group_models = models['similarity_group_models']
        self.athlete_groups = models['athlete_groups']
        self.window_size = models['window_size']
        print(f"✅ Models loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from analyze_movement_data import MovementDataAnalyzer
    
    folders = [
        "Input files/Drei M",
        "Input files/Drei W",
        "Input files/Weit M",
        "Input files/Weit W"
    ]
    
    print("Initializing Hybrid SSA Interpolator...")
    analyzer = MovementDataAnalyzer(folders)
    interpolator = HybridSSAInterpolator(window_size=40)
    
    # Train models
    interpolator.train(analyzer, folders)
    
    # Save models
    interpolator.save_models("hybrid_ssa_models.pkl")
    
    print("\n✅ Training complete! Models saved.")


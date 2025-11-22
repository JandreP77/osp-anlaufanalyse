"""
Kalman Filter + SSA Hybrid Interpolator
Olympic-grade gap filling for athletics run-up data

Scientific basis:
- Kalman Filter: Kalman, R. E. (1960) "A New Approach to Linear Filtering"
- SSA: Golyandina, N. et al. (2001) "Analysis of Time Series Structure: SSA"
- Hybrid approach: Combines physics (Kalman) with biomechanics (SSA)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.interpolate import CubicSpline
from pyts.decomposition import SingularSpectrumAnalysis


class KalmanFilter:
    """
    Extended Kalman Filter for 1D motion tracking
    
    State vector: [position, velocity, acceleration]
    Measurements: position only
    """
    
    def __init__(self, dt: float = 0.02):  # 50Hz = 0.02s
        """
        Initialize Kalman Filter
        
        Args:
            dt: Time step (default 0.02s for 50Hz)
        """
        self.dt = dt
        
        # State: [position, velocity, acceleration]
        self.x = np.zeros(3)
        
        # State transition matrix (constant acceleration model)
        self.F = np.array([
            [1, dt, 0.5*dt**2],
            [0, 1,  dt],
            [0, 0,  1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0]])
        
        # Process noise covariance (tuned for athletics)
        q = 100  # Process noise
        self.Q = q * np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2,   dt],
            [dt**2/2, dt,      1]
        ])
        
        # Measurement noise covariance (sensor noise)
        self.R = np.array([[200]])  # 200mm measurement noise
        
        # State covariance
        self.P = np.eye(3) * 1000
        
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]  # Return position
    
    def update(self, z: float):
        """Update with measurement"""
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
        
    def get_state(self) -> Tuple[float, float, float]:
        """Get current state [position, velocity, acceleration]"""
        return tuple(self.x)
    
    def get_uncertainty(self) -> float:
        """Get position uncertainty (standard deviation)"""
        return np.sqrt(self.P[0, 0])


class KalmanSSAInterpolator:
    """
    Hybrid interpolator combining Kalman Filter and SSA
    
    - Small gaps (<1m): Cubic Spline (mathematically optimal)
    - Medium gaps (1-5m): Kalman Filter (physics-based)
    - Large gaps (>5m): Kalman + SSA (physics + biomechanics)
    """
    
    def __init__(self, sampling_rate: int = 50, ssa_window: int = 40):
        """
        Initialize the hybrid interpolator
        
        Args:
            sampling_rate: Sampling frequency in Hz (default 50)
            ssa_window: SSA window size (default 40)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.ssa_window = ssa_window
        self.ssa_model = SingularSpectrumAnalysis(window_size=ssa_window, groups='auto')
        
    def interpolate_gap(self, data: np.ndarray, gap_start: int, gap_end: int, 
                       gap_size_mm: float) -> Tuple[np.ndarray, float]:
        """
        Interpolate a single gap
        
        Args:
            data: Full distance array (in mm)
            gap_start: Index before gap
            gap_end: Index after gap  
            gap_size_mm: Size of gap in mm
            
        Returns:
            Tuple of (interpolated_values, confidence_score)
        """
        gap_size_m = gap_size_mm / 1000
        
        # Determine number of points to insert
        expected_step = 180  # mm per frame (conservative estimate)
        num_points = max(1, int(gap_size_mm / expected_step))
        
        # Select method based on gap size
        if gap_size_m < 1.0:
            # Small gap: Cubic Spline
            return self._cubic_spline_interpolate(data, gap_start, gap_end, num_points)
        
        elif gap_size_m < 5.0:
            # Medium gap: Kalman Filter
            return self._kalman_interpolate(data, gap_start, gap_end, num_points)
        
        else:
            # Large gap: Kalman + SSA Hybrid
            return self._hybrid_interpolate(data, gap_start, gap_end, num_points)
    
    def _cubic_spline_interpolate(self, data: np.ndarray, gap_start: int, 
                                  gap_end: int, num_points: int) -> Tuple[np.ndarray, float]:
        """Cubic spline interpolation for small gaps"""
        # Use points around gap
        context_size = min(20, gap_start, len(data) - gap_end - 1)
        x_known = np.concatenate([
            np.arange(gap_start - context_size, gap_start + 1),
            np.arange(gap_end, gap_end + context_size + 1)
        ])
        y_known = np.concatenate([
            data[gap_start - context_size:gap_start + 1],
            data[gap_end:gap_end + context_size + 1]
        ])
        
        # Fit cubic spline
        cs = CubicSpline(x_known, y_known)
        
        # Interpolate
        x_new = np.linspace(gap_start + 1, gap_end, num_points, endpoint=False)
        y_new = cs(x_new)
        
        # High confidence for small gaps
        confidence = 0.95
        
        return y_new, confidence
    
    def _kalman_interpolate(self, data: np.ndarray, gap_start: int, 
                           gap_end: int, num_points: int) -> Tuple[np.ndarray, float]:
        """Kalman filter interpolation for medium gaps"""
        # Initialize Kalman filter with state before gap
        kf = KalmanFilter(dt=self.dt)
        
        # Estimate initial state from data before gap
        context_size = min(10, gap_start)
        if context_size >= 2:
            # Estimate velocity
            v = (data[gap_start] - data[gap_start - context_size]) / (context_size * self.dt) / 1000  # m/s
            
            # Estimate acceleration
            if context_size >= 4:
                v1 = (data[gap_start] - data[gap_start - context_size//2]) / ((context_size//2) * self.dt) / 1000
                v2 = (data[gap_start - context_size//2] - data[gap_start - context_size]) / ((context_size//2) * self.dt) / 1000
                a = (v1 - v2) / ((context_size//2) * self.dt)
            else:
                a = 0.0
            
            kf.x = np.array([data[gap_start], v * 1000, a * 1000])  # Convert to mm, mm/s, mm/s²
        
        # Predict through gap
        interpolated = []
        uncertainties = []
        
        for i in range(num_points):
            pos = kf.predict()
            interpolated.append(pos)
            uncertainties.append(kf.get_uncertainty())
        
        # Calculate confidence based on uncertainty
        avg_uncertainty = np.mean(uncertainties)
        confidence = max(0.3, 1.0 - (avg_uncertainty / 1000))  # Normalize
        
        return np.array(interpolated), confidence
    
    def _hybrid_interpolate(self, data: np.ndarray, gap_start: int, 
                           gap_end: int, num_points: int) -> Tuple[np.ndarray, float]:
        """Hybrid Kalman + SSA for large gaps"""
        # Step 1: Kalman prediction (physics-based)
        kalman_pred, kalman_conf = self._kalman_interpolate(data, gap_start, gap_end, num_points)
        
        # Step 2: SSA pattern extraction (biomechanics-based)
        try:
            # Get context for SSA
            context_size = min(self.ssa_window * 2, gap_start, len(data) - gap_end - 1)
            
            if context_size >= self.ssa_window:
                # Extract pattern from before gap
                context_before = data[max(0, gap_start - context_size):gap_start + 1]
                
                if len(context_before) >= self.ssa_window:
                    # Fit SSA
                    self.ssa_model.fit(context_before.reshape(1, -1))
                    reconstructed = self.ssa_model.transform(context_before.reshape(1, -1))[0]
                    
                    # Extract step pattern (periodicity)
                    step_pattern = np.diff(reconstructed)
                    avg_step = np.mean(step_pattern[-10:])  # Last 10 steps
                    
                    # Generate SSA-based prediction
                    ssa_pred = []
                    current_pos = data[gap_start]
                    for i in range(num_points):
                        current_pos += avg_step
                        ssa_pred.append(current_pos)
                    ssa_pred = np.array(ssa_pred)
                    
                    # Fusion: Weighted average (Kalman for trend, SSA for pattern)
                    weight_kalman = 0.6
                    weight_ssa = 0.4
                    fused = weight_kalman * kalman_pred + weight_ssa * ssa_pred
                    
                    # Confidence is lower for large gaps
                    confidence = kalman_conf * 0.7  # Penalize large gaps
                    
                    return fused, confidence
        except:
            pass
        
        # Fallback to Kalman only
        return kalman_pred, kalman_conf * 0.5
    
    def fill_all_gaps(self, data: List[float], gaps: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Fill all gaps in the dataset
        
        Args:
            data: Original distance data (list)
            gaps: List of gap dictionaries with 'index', 'difference', etc.
            
        Returns:
            Tuple of (filled_data, interpolation_info)
        """
        data_array = np.array(data)
        interpolation_info = []
        
        # Sort gaps by index (process in order)
        sorted_gaps = sorted(gaps, key=lambda g: g['index'])
        
        # Build new data array with interpolated points
        segments = []
        last_idx = 0
        
        for gap in sorted_gaps:
            gap_idx = gap['index']
            gap_size = gap['difference']
            
            # Add data before gap
            segments.append(data_array[last_idx:gap_idx + 1])
            
            # Interpolate gap
            interpolated, confidence = self.interpolate_gap(
                data_array, gap_idx, gap_idx + 1, gap_size
            )
            
            segments.append(interpolated)
            
            interpolation_info.append({
                'index': gap_idx,
                'size_mm': gap_size,
                'size_m': gap_size / 1000,
                'num_points': len(interpolated),
                'confidence': confidence,
                'method': self._get_method_name(gap_size / 1000),
                'start_idx': sum(len(s) for s in segments) - len(interpolated),
                'end_idx': sum(len(s) for s in segments)
            })
            
            last_idx = gap_idx + 1
        
        # Add remaining data
        if last_idx < len(data_array):
            segments.append(data_array[last_idx:])
        
        filled_data = np.concatenate(segments)
        
        return filled_data, interpolation_info
    
    def _get_method_name(self, gap_size_m: float) -> str:
        """Get method name based on gap size"""
        if gap_size_m < 1.0:
            return 'Cubic Spline'
        elif gap_size_m < 5.0:
            return 'Kalman Filter'
        else:
            return 'Kalman+SSA Hybrid'


if __name__ == "__main__":
    # Test the interpolator
    print("="*70)
    print("KALMAN + SSA HYBRID INTERPOLATOR TEST")
    print("="*70)
    print()
    
    # Create test data with gap
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    true_data = 1000 * t + 50 * np.sin(2 * np.pi * t)  # Trend + periodicity
    
    # Create gap
    gap_start = 200
    gap_end = 250
    data_with_gap = true_data.copy()
    true_gap_data = data_with_gap[gap_start+1:gap_end].copy()
    data_with_gap[gap_start+1:gap_end] = np.nan
    
    # Interpolate
    interpolator = KalmanSSAInterpolator(sampling_rate=50)
    
    gaps = [{
        'index': gap_start,
        'difference': float(true_data[gap_end] - true_data[gap_start])
    }]
    
    filled_data, interp_info = interpolator.fill_all_gaps(data_with_gap[:gap_end+1].tolist(), gaps)
    
    print(f"Original data length: {len(true_data[:gap_end+1])}")
    print(f"Filled data length: {len(filled_data)}")
    print(f"Gap size: {gaps[0]['difference']:.1f}mm")
    print()
    
    if interp_info:
        info = interp_info[0]
        print(f"Interpolation method: {info['method']}")
        print(f"Points inserted: {info['num_points']}")
        print(f"Confidence: {info['confidence']:.1%}")
        print()
        
        # Calculate error
        interpolated_segment = filled_data[info['start_idx']:info['end_idx']]
        true_segment = true_gap_data
        
        if len(interpolated_segment) == len(true_segment):
            rmse = np.sqrt(np.mean((interpolated_segment - true_segment)**2))
            print(f"RMSE: {rmse:.2f}mm")
            print(f"Max error: {np.max(np.abs(interpolated_segment - true_segment)):.2f}mm")
    
    print()
    print("✅ Test complete!")


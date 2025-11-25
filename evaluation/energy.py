"""
Energy measurement wrapper using CodeCarbon.

Provides batch-level energy tracking for RAG systems.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from functools import wraps
from dataclasses import dataclass, asdict
import time

try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from base import BaseRAG

logger = logging.getLogger(__name__)


@dataclass
class EnergyMetrics:
    """Container for energy measurement results."""
    energy_kwh: float  # Energy consumed in kilowatt-hours
    emissions_kg_co2: float  # CO2 emissions in kilograms
    duration_sec: float  # Duration in seconds
    power_avg_watts: float  # Average power draw in watts
    cpu_energy_kwh: Optional[float] = None
    gpu_energy_kwh: Optional[float] = None
    ram_energy_kwh: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __repr__(self) -> str:
        return (
            f"EnergyMetrics(energy={self.energy_kwh*1000:.4f} Wh, "
            f"emissions={self.emissions_kg_co2*1000:.4f} g CO2, "
            f"duration={self.duration_sec:.2f}s, "
            f"power={self.power_avg_watts:.2f}W)"
        )


class EnergyTracker:
    """
    Energy measurement wrapper for RAG evaluation.
    
    Wraps CodeCarbon EmissionsTracker for batch-level energy measurement.
    """
    
    def __init__(
        self,
        project_name: str = "rag_eval",
        output_dir: str = "results/energy",
        measure_power: bool = True,
        save_to_file: bool = True,
        offline_mode: bool = True,
        country_iso_code: str = "USA"
    ):
        """
        Initialize energy tracker.
        
        Args:
            project_name: Name for the tracking project
            output_dir: Directory to save emissions data
            measure_power: Whether to measure power consumption
            save_to_file: Whether to save detailed logs to CSV
            offline_mode: Use offline mode (no network calls)
            country_iso_code: ISO code for country (for emissions calculation)
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.measure_power = measure_power
        self.save_to_file = save_to_file
        self.offline_mode = offline_mode
        self.country_iso_code = country_iso_code
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not CODECARBON_AVAILABLE:
            logger.warning(
                "CodeCarbon not installed. Energy tracking will be disabled. "
                "Install with: pip install codecarbon"
            )
        else:
            logger.info(f"EnergyTracker initialized (offline_mode={offline_mode})")
    
    def _create_tracker(self, experiment_name: str) -> Optional[Any]:
        """Create a new emissions tracker."""
        if not CODECARBON_AVAILABLE:
            return None
        
        try:
            if self.offline_mode:
                tracker = OfflineEmissionsTracker(
                    project_name=self.project_name,
                    experiment_id=experiment_name,
                    output_dir=str(self.output_dir),
                    measure_power_secs=1,  # Measure every second
                    save_to_file=self.save_to_file,
                    country_iso_code=self.country_iso_code,
                    log_level="warning"  # Reduce verbosity
                )
            else:
                tracker = EmissionsTracker(
                    project_name=self.project_name,
                    experiment_id=experiment_name,
                    output_dir=str(self.output_dir),
                    measure_power_secs=1,
                    save_to_file=self.save_to_file,
                    log_level="warning"
                )
            return tracker
        except Exception as e:
            logger.warning(f"Failed to create emissions tracker: {e}")
            return None
    
    def measure_batch(
        self,
        rag_system: BaseRAG,
        queries: List[str],
        experiment_name: Optional[str] = None,
        return_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Measure energy consumption for a batch of queries.
        
        Args:
            rag_system: RAG system implementing BaseRAG
            queries: List of queries to process
            experiment_name: Name for this experiment
            return_trace: Whether to return trace info from RAG
            
        Returns:
            Dictionary with:
                - 'answers': List of answers
                - 'energy': EnergyMetrics object
                - 'per_query_energy': Estimated energy per query
        """
        if experiment_name is None:
            experiment_name = f"{rag_system.name}_batch"
        
        tracker = self._create_tracker(experiment_name)
        
        # Start tracking
        start_time = time.time()
        if tracker:
            tracker.start()
        
        try:
            # Run the batch
            answers = rag_system.batch_answer(queries, return_trace=return_trace)
        finally:
            # Stop tracking
            duration = time.time() - start_time
            
            if tracker:
                emissions = tracker.stop()
                
                # Extract detailed metrics
                energy_metrics = EnergyMetrics(
                    energy_kwh=tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else emissions,
                    emissions_kg_co2=emissions if isinstance(emissions, float) else 0.0,
                    duration_sec=duration,
                    power_avg_watts=(tracker._total_energy.kWh * 1000 * 3600 / duration) if hasattr(tracker, '_total_energy') and duration > 0 else 0.0,
                    cpu_energy_kwh=getattr(tracker._total_cpu_energy, 'kWh', None) if hasattr(tracker, '_total_cpu_energy') else None,
                    gpu_energy_kwh=getattr(tracker._total_gpu_energy, 'kWh', None) if hasattr(tracker, '_total_gpu_energy') else None,
                    ram_energy_kwh=getattr(tracker._total_ram_energy, 'kWh', None) if hasattr(tracker, '_total_ram_energy') else None
                )
            else:
                # Fallback when CodeCarbon not available
                energy_metrics = EnergyMetrics(
                    energy_kwh=0.0,
                    emissions_kg_co2=0.0,
                    duration_sec=duration,
                    power_avg_watts=0.0
                )
        
        # Calculate per-query estimate
        per_query_energy = energy_metrics.energy_kwh / len(queries) if queries else 0.0
        
        return {
            "system_name": rag_system.name,
            "num_queries": len(queries),
            "answers": answers,
            "energy": energy_metrics,
            "per_query_energy_kwh": per_query_energy,
            "experiment_name": experiment_name
        }


def with_energy_tracking(
    tracker: EnergyTracker,
    experiment_name: Optional[str] = None
):
    """
    Decorator to add energy tracking to a function.
    
    Usage:
        tracker = EnergyTracker()
        
        @with_energy_tracking(tracker, "my_experiment")
        def my_rag_function(queries):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tracker = tracker._create_tracker(experiment_name or func.__name__)
            
            start_time = time.time()
            if _tracker:
                _tracker.start()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if _tracker:
                    emissions = _tracker.stop()
                    logger.info(
                        f"[{experiment_name or func.__name__}] "
                        f"Duration: {duration:.2f}s, "
                        f"Emissions: {emissions*1000:.4f}g CO2"
                    )
        
        return wrapper
    return decorator


# Convenience function
def create_energy_tracker(**kwargs) -> EnergyTracker:
    """Factory function to create EnergyTracker."""
    return EnergyTracker(**kwargs)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print(f"CodeCarbon available: {CODECARBON_AVAILABLE}")
    
    tracker = EnergyTracker()
    print(f"Tracker created: {tracker.project_name}")
    
    # Test with a mock RAG system
    class MockRAG(BaseRAG):
        @property
        def name(self) -> str:
            return "mock_rag"
        
        def answer(self, query: str, return_trace: bool = False):
            time.sleep(0.1)  # Simulate work
            return "Mock answer"
        
        def batch_answer(self, queries: List[str], return_trace: bool = False):
            return [self.answer(q, return_trace) for q in queries]
    
    mock = MockRAG()
    result = tracker.measure_batch(mock, ["test query 1", "test query 2"])
    
    print(f"\nResults:")
    print(f"  System: {result['system_name']}")
    print(f"  Queries: {result['num_queries']}")
    print(f"  Energy: {result['energy']}")

"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure matplotlib for testing
import matplotlib
matplotlib.use('Agg')


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root


@pytest.fixture(scope="session")
def sample_car_data():
    """Create sample car data for testing."""
    return {
        'year': 2018,
        'km_driven': 50000,
        'fuel': 'Petrol',
        'seller_type': 'Individual',
        'transmission': 'Manual',
        'owner': 'First Owner',
        'mileage': 18.5,
        'engine': 1200,
        'max_power': 85,
        'seats': 5
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
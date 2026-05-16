import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.config.settings import create_required_directories
from src.data.extract_data import run_data_extraction
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """
    Execute data extraction script.

    Returns
    -------
    None
    """
    create_required_directories()

    monthly_prices = run_data_extraction()

    logger.info(
        "Extraction completed. Final monthly prices shape: %s",
        monthly_prices.shape,
    )


if __name__ == "__main__":
    main()
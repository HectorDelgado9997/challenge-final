import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.config.settings import create_required_directories
from src.data.build_dataset import run_dataset_building
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """
    Execute model dataset construction script.

    Returns
    -------
    None
    """
    create_required_directories()

    model_dataset = run_dataset_building()

    logger.info(
        "Dataset building completed. Final model dataset shape: %s",
        model_dataset.shape,
    )


if __name__ == "__main__":
    main()
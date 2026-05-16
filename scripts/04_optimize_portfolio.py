import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import pandas as pd

from src.config.settings import (
    DEFAULT_MONTHLY_INVESTMENT_AMOUNT,
    MODEL_DATASET_PATH,
    create_required_directories,
)
from src.data.validate_data import (
    validate_file_exists,
    validate_model_dataset,
    validate_monthly_investment_amount,
)
from src.portfolio.optimize_portfolio import run_portfolio_optimization
from src.utils.logger import get_logger


logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run monthly portfolio optimization."
    )

    parser.add_argument(
        "--amount",
        type=float,
        default=DEFAULT_MONTHLY_INVESTMENT_AMOUNT,
        help="Monthly investment amount.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Execute portfolio optimization script.

    Returns
    -------
    None
    """
    args = parse_arguments()

    validate_monthly_investment_amount(args.amount)

    create_required_directories()

    validate_file_exists(MODEL_DATASET_PATH)

    dataframe = pd.read_csv(MODEL_DATASET_PATH)
    validate_model_dataset(dataframe)

    allocation, summary = run_portfolio_optimization(
        dataframe=dataframe,
        monthly_investment_amount=args.amount,
    )

    logger.info("Recommended allocation:")
    logger.info("\n%s", allocation.to_string(index=False))

    logger.info("Portfolio summary:")
    logger.info("%s", summary)


if __name__ == "__main__":
    main()
    
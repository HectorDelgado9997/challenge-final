class DataValidationError(Exception):
    """Raised when the input dataset does not meet the expected structure."""
    pass


class DataExtractionError(Exception):
    """Raised when financial data extraction fails."""
    pass


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization fails."""
    pass
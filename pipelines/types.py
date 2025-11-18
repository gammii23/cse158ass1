"""Type definitions for pipeline results."""

from typing import TypedDict

import pandas as pd


class SubmissionResult(TypedDict, total=False):
    """Result type for pipeline execution.

    Attributes:
        success: Whether pipeline executed successfully.
        data: DataFrame with predictions (required if success=True).
        error: Error message (required if success=False).
    """

    success: bool
    data: pd.DataFrame
    error: str



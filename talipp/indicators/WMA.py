from collections import deque

import numpy as np
from typing import List, Any

from talipp.indicator_util import has_valid_values
from talipp.indicators.Indicator import Indicator, InputModifierType
from talipp.input import SamplingPeriodType


class WMA(Indicator):
    """Weighted Moving Average.

    Input type: `float`

    Output type: `float`

    Args:
        period: Period.
        input_values: List of input values.
        input_indicator: Input indicator.
        input_modifier: Input modifier.
        input_sampling: Input sampling type.
    """

    def __init__(self, period: int,
                 input_values: List[float] = None,
                 input_indicator: Indicator = None,
                 input_modifier: InputModifierType = None,
                 input_sampling: SamplingPeriodType = None):
        super().__init__(input_modifier=input_modifier,
                         input_sampling=input_sampling)

        self.period = period

        self.denom_sum = period * (period + 1) / 2.0
        self.weights = np.arange(1, self.period + 1)
        self.index = 0  # Start index at 0

        self.input_values_np = np.zeros(period)
        if input_values:
            # Populate with initial values if provided, handling wrap-around
            init_len = min(len(input_values), period)
            self.input_values_np[-init_len:] = input_values[-init_len:]
            self.index = init_len % period

        self.initialize(input_values, input_indicator)

    def _calculate_new_value(self) -> Any:
        self.add_value(self.input_values[-1])
        if not has_valid_values(self.input_values, self.period):
            return None

        # Compute indices for a virtual rotation to align the oldest to the newest
        indices = (self.index + np.arange(self.period)) % self.period

        # Calculate the weighted moving average with correct alignment
        wma = np.dot(self.input_values_np[indices], self.weights) / self.denom_sum

        return wma

    def add_value(self, value: float):
        """
        Add a new value to the WMA calculation.

        :param value: The new value to add to the calculation.
        """
        # Insert the new value at the current index
        self.input_values_np[self.index] = value
        # Update the index wrapping around using modulo operation
        self.index = (self.index + 1) % self.period

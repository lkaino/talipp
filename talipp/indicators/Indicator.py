from dataclasses import is_dataclass
from abc import ABCMeta, abstractmethod
from typing import List, Any, Callable, Dict, Union
from collections import defaultdict
from collections.abc import MutableSequence, Sequence
from itertools import chain
from operator import methodcaller, attrgetter

ListAny = List[Any]
ManagedSequenceType = Union['Indicator', MutableSequence]
ValueExtractorType = Callable[..., Any]


class Indicator(Sequence):
    __metaclass__ = ABCMeta

    def __init__(self, value_extractor: ValueExtractorType = None):
        self.value_extractor = value_extractor

        self.input_values: ListAny = []
        self.output_values: ListAny = []
        self.managed_sequences: List[ManagedSequenceType] = []
        self.sub_indicators: List[Indicator] = []

    @abstractmethod
    def _calculate_new_value(self) -> Any:
        pass

    def __getitem__(self, index):
        return self.output_values[index]

    def __len__(self):
        return len(self.output_values)

    def __str__(self):
        return str(self.output_values)

    def add_sub_indicator(self, indicator: 'Indicator') -> None:
        self.sub_indicators.append(indicator)

    def add_managed_sequence(self, lst: ManagedSequenceType) -> None:
        self.managed_sequences.append(lst)

    def initialize(self, input_values: ListAny) -> None:
        self.remove_all()

        if input_values is not None:
            for value in input_values:
                self.add_input_value(value)

    def add_input_value(self, value: Any) -> None:
        for sub_indicator in self.sub_indicators:
            sub_indicator.add_input_value(value)

        if self.value_extractor is not None:
            value = self.value_extractor(value)
        self.input_values.append(value)

        new_value = self._calculate_new_value()
        if new_value is not None:
            self.output_values.append(new_value)

    def update_input_value(self, value: Any) -> None:
        self.remove_input_value()
        self.add_input_value(value)

    def remove_input_value(self) -> None:
        for sub_indicator in self.sub_indicators:
            sub_indicator.remove_input_value()

        self.input_values.pop(-1)

        self._remove_output_value()

        for lst in self.managed_sequences:
            if isinstance(lst, Indicator):
                lst.remove_input_value()
            else:
                if len(lst) > 0:
                    lst.pop(-1)

        self._remove_input_value_custom()

    def _remove_output_value(self) -> None:
        if len(self.output_values) > 0:
            self.output_values.pop(-1)

    def _remove_input_value_custom(self) -> None:
        pass

    def remove_all(self) -> None:
        for sub_indicator in self.sub_indicators:
            sub_indicator.remove_all()

        self.input_values = []
        self.output_values = []

        for lst in self.managed_sequences:
            if isinstance(lst, Indicator):
                lst.remove_all()
            else:
                lst.clear()

        self._remove_all_custom()

    def _remove_all_custom(self) -> None:
        pass

    def set_input_values(self, input_values: ListAny, initialize: bool = True) -> None:
        if initialize:
            self.initialize(input_values)
        else:
            self.input_values = input_values

    def has_output_value(self) -> bool:
        if len(self.output_values) > 0 and self.output_values[-1] is not None:
            return True
        else:
            return False

    def to_lists(self, align_with_input: bool = False) -> Dict[str, List[float]]:
        if len(self.output_values) == 0:
            return {}
        else:
            if is_dataclass(self.output_values[0]):
                result = defaultdict(list)
                for key, value in chain.from_iterable(map(methodcaller('items'), map(attrgetter('__dict__'), self.output_values))):
                    result[key].extend([value])
                return dict(result)
            else:
                raise Exception("to_lists() method can be used only with indicators returning multiple values as their result.")
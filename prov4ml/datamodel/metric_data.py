
import os
import numpy as np
from typing import Any, Dict, List
from typing import Optional
import zarr

from prov4ml.datamodel.attribute_type import LoggingItemKind

class MetricInfo:
    """
    A class to store information about a specific metric.

    Attributes:
    -----------
    name : str
        The name of the metric.
    context : Any
        The context in which the metric is recorded.
    source : LoggingItemKind
        The source of the logging item.
    total_metric_values : int
        The total number of metric values recorded.
    epochDataList : dict
        A dictionary mapping epoch numbers to lists of metric values recorded in those epochs.

    Methods:
    --------
    __init__(name: str, context: Any, source=LoggingItemKind) -> None
        Initializes the MetricInfo class with the given name, context, and source.
    add_metric(value: Any, epoch: int, timestamp : int) -> None
        Adds a metric value for a specific epoch to the MetricInfo object.
    save_to_file(path : str, process : Optional[int] = None) -> None
        Saves the metric information to a file.
    """
    def __init__(self, name: str, context: Any, source=LoggingItemKind) -> None:
        """
        Initializes the MetricInfo class with the given name, context, and source.

        Parameters:
        -----------
        name : str
            The name of the metric.
        context : Any
            The context in which the metric is recorded.
        source : LoggingItemKind
            The source of the logging item.

        Returns:
        --------
        None
        """
        self.name = name
        self.context = context
        self.source = source
        self.total_metric_values = 0
        self.epochDataList: Dict[int, List[Any]] = {}

    def add_metric(self, value: Any, epoch: int, timestamp : int) -> None:
        """
        Adds a metric value for a specific epoch to the MetricInfo object.

        Parameters:
        -----------
        value : Any
            The value of the metric to be added.
        epoch : int
            The epoch number in which the metric value is recorded.
        timestamp : int
            The timestamp when the metric value was recorded.

        Returns:
        --------
        None
        """
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []

        self.epochDataList[epoch].append((value, timestamp))
        self.total_metric_values += 1

    def save_to_file(
            self, 
            path : str, 
            process : Optional[int] = None
        ) -> None:
        """
        Saves the metric information to a file.

        Parameters:
        -----------
        path : str
            The directory path where the file will be saved.
        process : Optional[int], optional
            The process identifier to be included in the filename. If not provided, 
            the filename will not include a process identifier.

        Returns:
        --------
        None
        """
        # if process is not None:
        #     file = os.path.join(path, f"{self.name}_{self.context}_GR{process}.txt")
        # else:
        #     file = os.path.join(path, f"{self.name}_{self.context}.txt")
        # file_exists = os.path.exists(file)

        # with open(file, "a") as f:
        #     if not file_exists:
        #         f.write(f"{self.name}, {self.context}, {self.source}\n")
        #     for epoch, values in self.epochDataList.items():
        #         for value, timestamp in values:
        #             f.write(f"{epoch}, {value}, {timestamp}\n")

        # self.epochDataList = {}

        if process is not None:
            zarr_file = os.path.join(path, f"{self.name}_{self.context}_GR{process}.zarr")
        else:
            zarr_file = os.path.join(path, f"{self.name}_{self.context}.zarr")

        if os.path.exists(zarr_file):
            dataset = zarr.open(zarr_file, mode='a')
        else:
            dataset = zarr.open(zarr_file, mode='w')

            # Metadata
            dataset.attrs['name'] = self.name
            dataset.attrs['context'] = str(self.context)
            dataset.attrs['source'] = str(self.source)

        epochs = []
        values = []
        timestamps = []

        for epoch, items in self.epochDataList.items():
            for value, timestamp in items:
                epochs.append(epoch)
                values.append(value)
                timestamps.append(timestamp)

        epochs = np.array(epochs, dtype='i4')
        values = np.array(values, dtype='f4')
        timestamps = np.array(timestamps, dtype='i8')

        if 'epochs' in dataset:
            dataset['epochs'].append(epochs)
            dataset['values'].append(values)
            dataset['timestamps'].append(timestamps)
        else:
            dataset.create_dataset('epochs', data=epochs, chunks=(1000,), dtype=epochs.dtype)
            dataset.create_dataset('values', data=values, chunks=(1000,), dtype=values.dtype)
            dataset.create_dataset('timestamps', data=timestamps, chunks=(1000,), dtype=timestamps.dtype)

        self.epochDataList = {}



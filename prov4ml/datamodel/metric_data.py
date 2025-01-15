
import os
import numpy as np
from typing import Any, Dict, List
from typing import Optional
import zarr

from prov4ml.datamodel.attribute_type import LoggingItemKind
from prov4ml.provenance.metrics_type import MetricsType

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
            path: str, 
            file_type: MetricsType,
            use_compression: bool, # Spostarlo quando viene creata la metrica
            process: Optional[int] = None
        ) -> None:
        """
        Saves the metric information to a file.

        Parameters:
        -----------
        path : str
            The directory path where the file will be saved.
        file_type : str
            The type of file to be saved.
        process : Optional[int], optional
            The process identifier to be included in the filename. If not provided, 
            the filename will not include a process identifier.

        Returns:
        --------
        None
        """
        if process is not None:
            file = os.path.join(path, f"{self.name}_{self.context}_GR{process}.{file_type.value}")
        else:
            file = os.path.join(path, f"{self.name}_{self.context}.{file_type.value}")

        if file_type == MetricsType.ZARR:
            self.save_to_zarr(file, use_compression)
        elif file_type == MetricsType.TXT:
            self.save_to_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        self.epochDataList = {}

    def save_to_zarr(
            self,
            zarr_file: str,
            use_compression: bool
        ) -> None:
        """
        Saves the metric information in a zarr file.

        Parameters:
        -----------
        zarr_file : str
            The path to the zarr file where the metric information will be saved.

        Returns:
        --------
        None
        """
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

        if 'epochs' in dataset:
            dataset['epochs'].append(epochs)
            dataset['values'].append(values)
            dataset['timestamps'].append(timestamps)
        else:
            if use_compression:
                dataset.create_dataset('epochs', data=epochs, chunks=(1000,), dtype='i4')
                dataset.create_dataset('values', data=values, chunks=(1000,), dtype='f4')
                dataset.create_dataset('timestamps', data=timestamps, chunks=(1000,), dtype='i8')
            else:
                dataset.create_dataset('epochs', data=epochs, chunks=(1000,), dtype='i4', compressor=None)
                dataset.create_dataset('values', data=values, chunks=(1000,), dtype='f4', compressor=None)
                dataset.create_dataset('timestamps', data=timestamps, chunks=(1000,), dtype='i8', compressor=None)

    def save_to_txt(
            self,
            txt_file: str
        ) -> None:
        """
        Saves the metric information in a text file.

        Parameters:
        -----------
        txt_file : str
            The path to the text file where the metric information will be saved.

        Returns:
        --------
        None
        """
        file_exists = os.path.exists(txt_file)

        with open(txt_file, "a") as f:
            if not file_exists:
                f.write(f"{self.name}, {self.context}, {self.source}\n")
            for epoch, values in self.epochDataList.items():
                for value, timestamp in values:
                    f.write(f"{epoch}, {value}, {timestamp}\n")

    def copy_to_zarr(
            self,
            path: str,
            file_type: MetricsType,
            use_compression: bool = True,
            process: Optional[int] = None
        ) -> None:
        """
        Copies the metric to a zarr file.

        Parameters:
        -----------
        path : str
            The directory path where the file will be saved.
        file_type : MetricsType
            The type of file to be read.
        use_compression : bool
            Whether to use compression when saving the zarr file. Defaults to True.
        process : Optional[int], optional
            The process identifier to be included in the filename. If not provided, 
            the filename will not include a process identifier.

        Returns:
        --------
        None
        """
        if process is not None:
            file = os.path.join(path, f"{self.name}_{self.context}_GR{process}.{file_type.value}")
        else:
            file = os.path.join(path, f"{self.name}_{self.context}.{file_type.value}")

        output_path = os.path.join(path, f"copy_{self.name}_{self.context}_GR{process}.{file_type.value}")
        output_file = zarr.open(output_path, mode='w')

        if file_type == MetricsType.ZARR:
            dataset = zarr.open(file, mode='r')

            for name in dataset.array_keys():
                if use_compression:
                    output_file.create_dataset(name, data=dataset[name], chunks=dataset[name].chunks, dtype=dataset[name].dtype, shape=dataset[name].shape)
                else:
                    output_file.create_dataset(name, data=dataset[name], chunks=dataset[name].chunks, dtype=dataset[name].dtype, shape=dataset[name].shape, compressor=None)

            for key, value in dataset.attrs.items():
                output_file.attrs[key] = value

        elif file_type == MetricsType.TXT:
            with open(file, "r") as f:
                lines = f.readlines()

            epochs = []
            values = []
            timestamps = []
            for line in lines[1:]:
                epoch, value, timestamp = line.split(',')
                epochs.append(int(epoch))
                values.append(float(value))
                timestamps.append(int(timestamp))

            epochs = np.array(epochs, dtype='i4')
            values = np.array(values, dtype='f4')
            timestamps = np.array(timestamps, dtype='i8')

            if use_compression:
                output_file.create_dataset('epochs', data=epochs, chunks=(100,), dtype='i4')
                output_file.create_dataset('values', data=values, chunks=(100,), dtype='f4')
                output_file.create_dataset('timestamps', data=timestamps, chunks=(100,), dtype='i8')
            else:
                output_file.create_dataset('epochs', data=epochs, chunks=(100,), dtype='i4', compressor=None)
                output_file.create_dataset('values', data=values, chunks=(100,), dtype='f4', compressor=None)
                output_file.create_dataset('timestamps', data=timestamps, chunks=(100,), dtype='i8', compressor=None)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

import os
from typing import Any, Dict, List
from typing import Optional
import zarr
import netCDF4 as nc
import zarr.codecs

from yprov4ml.datamodel.attribute_type import LoggingItemKind
from yprov4ml.datamodel.compressor_type import CompressorType, compressor_to_type
from yprov4ml.datamodel.metric_type import MetricsType, get_file_type

ZARR_CHUNK_SIZE = 1000

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
    def __init__(self, 
                 name: str, 
                 context: Any, 
                 source=LoggingItemKind, 
                 use_compressor : Optional[CompressorType] = None
                 ) -> None:
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
        self.use_compressor = use_compressor
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
            csv_separator = ",", 
            process: Optional[int] = None, 
        ) -> None:
        """
        Saves the metric information to a file.

        Parameters
        ----------
        path : str
            The directory path where the file will be saved.
        file_type : str
            The type of file to be saved.
        process : Optional[int], optional
            The process identifier to be included in the filename. If not provided, 
            the filename will not include a process identifier.

        Returns
        -------
        None
        """
        process = process if process is not None else 0
        file = os.path.join(path, f"{self.name}_{self.context}_GR{process}")

        ft = file + get_file_type(file_type)
        if file_type == MetricsType.ZARR:
            self.save_to_zarr(ft)
        elif file_type == MetricsType.CSV:
            self.save_to_txt(ft, csv_separator)
        elif file_type == MetricsType.NETCDF:
            self.save_to_netCDF(ft)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        self.epochDataList = {}


    def save_to_netCDF(
            self,
            netcdf_file: str
        ) -> None:
        """
        Saves the metric information in a netCDF file.

        Parameters
        ----------
        netcdf_file : str
            The path to the netCDF file where the metric information will be saved.

        Returns
        -------
        None
        """
        if os.path.exists(netcdf_file):
            dataset = nc.Dataset(netcdf_file, mode='a', format='NETCDF4')
        else:
            dataset = nc.Dataset(netcdf_file, mode='w', format='NETCDF4')
            dataset._name = self.name
            dataset._context = str(self.context)
            dataset._source = str(self.source)

            compression = 'zlib' if self.use_compressor else None
            dataset.createDimension('time', None)
            dataset.createVariable('epochs', 'i4', ('time',), compression)
            dataset.createVariable('values', 'f4', ('time',), compression)
            dataset.createVariable('timestamps', 'i8', ('time',), compression)

        epochs = []
        values = []
        timestamps = []

        for epoch, items in self.epochDataList.items():
            for value, timestamp in items:
                epochs.append(epoch)
                values.append(value)
                timestamps.append(timestamp)

        current_size = dataset.dimensions['time'].size
        new_size = current_size + len(epochs)

        dataset.variables['epochs'][current_size:new_size] = epochs
        dataset.variables['values'][current_size:new_size] = values
        dataset.variables['timestamps'][current_size:new_size] = timestamps

        dataset.close()

    def save_to_zarr(
            self,
            zarr_file: str
        ) -> None:
        """
        Saves the metric information in a zarr file.

        Parameters
        ----------
        zarr_file : str
            The path to the zarr file where the metric information will be saved.

        Returns
        -------
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

        if 'epochs' not in dataset:
            dataset.create_array('epochs', shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype='i4', compressors=compressor_to_type(self.use_compressor))
            dataset.create_array('values', shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype='f4', compressors=compressor_to_type(self.use_compressor))
            dataset.create_array('timestamps', shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype='i8', compressors=compressor_to_type(self.use_compressor))

        dataset['epochs'].append(epochs)
        dataset['values'].append(values)
        dataset['timestamps'].append(timestamps)

        dataset.store.close()

    def save_to_txt(
            self,
            txt_file: str, 
            csv_separator : str = ",", 
        ) -> None:
        """
        Saves the metric information in a text file.

        Parameters
        ----------
        txt_file : str
            The path to the text file where the metric information will be saved.

        Returns
        -------
        None
        """
        file_exists = os.path.exists(txt_file)

        with open(txt_file, "a") as f:
            if not file_exists:
                f.write(f"{self.name}{csv_separator}{self.context}{csv_separator}{self.source}\n")
            for epoch, values in self.epochDataList.items():
                for value, timestamp in values:
                    f.write(f"{epoch}{csv_separator}{value}{csv_separator}{timestamp}\n")

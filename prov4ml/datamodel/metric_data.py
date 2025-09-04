import os
from typing import Any, Dict, List, Optional

import netCDF4 as nc
import zarr

from prov4ml.datamodel.attribute_type import LoggingItemKind
from prov4ml.datamodel.compressor_type import CompressorType, compressor_to_type
from prov4ml.datamodel.metric_type import MetricsType, get_file_type

ZARR_CHUNK_SIZE = 1024


class MetricInfo:
    """
    Holds a single metric's buffered values and persists to CSV/NETCDF/ZARR.

    If `unify_experiments=True` the NETCDF/ZARR representations use a leading
    `exp` dimension and append the current run at `exp=experiment_index`.
    """

    def __init__(
        self,
        name: str,
        context: Any,
        source: LoggingItemKind = LoggingItemKind.METRIC,
        use_compressor: Optional[CompressorType] = None,
        unify_experiments: bool = False,
        experiment_index: int = 0,
    ) -> None:
        self.name = name
        self.context = context
        self.source = source
        self.total_metric_values = 0
        self.use_compressor = use_compressor
        self.unify_experiments = bool(unify_experiments)
        self.experiment_index = int(experiment_index) if unify_experiments else 0
        self.epochDataList: Dict[int, List[Any]] = {}

    # ------------------------ buffer API ------------------------
    def add_metric(self, value: Any, epoch: int, timestamp: int) -> None:
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []
        self.epochDataList[epoch].append((value, timestamp))
        self.total_metric_values += 1

    # ------------------------ persist API ------------------------
    def save_to_file(
        self,
        path: str,
        file_type: MetricsType,
        csv_separator: str = ",",
        process: Optional[int] = None,
    ) -> None:
        process = process if process is not None else 0
        file = os.path.join(path, f"{self.name}_{self.context}_GR{process}")
        ft = file + get_file_type(file_type)

        if file_type == MetricsType.ZARR:
            self.save_to_zarr(ft)
        elif file_type == MetricsType.CSV:
            self.save_to_csv(ft, csv_separator)
        elif file_type == MetricsType.NETCDF:
            self.save_to_netcdf(ft)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # clear buffer once persisted
        self.epochDataList = {}

    # ------------------------ backends ------------------------
    def collect_lists(self):
        epochs, values, timestamps = [], [], []
        for epoch, items in self.epochDataList.items():
            for value, ts in items:
                epochs.append(epoch)
                values.append(value)
                timestamps.append(ts)
        return epochs, values, timestamps

    def save_to_netcdf(self, netcdf_file: str) -> None:
        epochs, values, timestamps = self.collect_lists()

        # If file exists and is legacy 1-D, migrate it to ('exp','time') in place.
        if os.path.exists(netcdf_file):
            ds_probe = nc.Dataset(netcdf_file, mode="r")
            try:
                legacy = "exp" not in ds_probe.dimensions
                if legacy and self.unify_experiments:
                    # Read old arrays
                    old_epochs = ds_probe.variables["epochs"][:]
                    old_values = ds_probe.variables["values"][:]
                    old_ts = ds_probe.variables["timestamps"][:]
                    name = getattr(ds_probe, "_name", self.name)
                    ctx = getattr(ds_probe, "_context", str(self.context))
                    src = getattr(ds_probe, "_source", str(self.source))
                else:
                    legacy = False
            finally:
                ds_probe.close()

            if legacy:
                # Re-write file as unified
                ds_new = nc.Dataset(netcdf_file, mode="w", format="NETCDF4")
                try:
                    ds_new._name = name
                    ds_new._context = ctx
                    ds_new._source = src
                    ds_new.createDimension("exp", None)
                    ds_new.createDimension("time", None)
                    v_epochs = ds_new.createVariable("epochs", "i4", ("exp", "time"), zlib=True)
                    v_values = ds_new.createVariable("values", "f4", ("exp", "time"), zlib=True, fill_value=float("nan"))
                    v_ts     = ds_new.createVariable("timestamps", "i8", ("exp", "time"), zlib=True, fill_value=-1)
                    v_len    = ds_new.createVariable("lengths", "i4", ("exp",), zlib=True)

                    n = len(old_values)
                    # pad for potential exps
                    v_values[:, 0:n] = float("nan")
                    v_epochs[:, 0:n] = 0
                    v_ts[:, 0:n] = -1

                    # write legacy data into exp=0
                    v_epochs[0, 0:n] = old_epochs
                    v_values[0, 0:n] = old_values
                    v_ts[0, 0:n] = old_ts
                    v_len[0] = n
                finally:
                    ds_new.close()

        # Now open for create/append with unified/legacy branches
        create = not os.path.exists(netcdf_file)
        ds = nc.Dataset(netcdf_file, mode="w" if create else "a", format="NETCDF4")
        try:
            if create:
                ds._name = self.name
                ds._context = str(self.context)
                ds._source = str(self.source)

                if self.unify_experiments:
                    ds.createDimension("exp", None)
                    ds.createDimension("time", None)
                    ds.createVariable("epochs", "i4", ("exp", "time"), zlib=True)
                    ds.createVariable("values", "f4", ("exp", "time"), zlib=True, fill_value=float("nan"))
                    ds.createVariable("timestamps", "i8", ("exp", "time"), zlib=True, fill_value=-1)
                    ds.createVariable("lengths", "i4", ("exp",), zlib=True)
                else:
                    ds.createDimension("time", None)
                    ds.createVariable("epochs", "i4", ("time",), zlib=True)
                    ds.createVariable("values", "f4", ("time",), zlib=True)
                    ds.createVariable("timestamps", "i8", ("time",), zlib=True)

            if self.unify_experiments:
                # Pick exp index automatically if requested
                auto_idx = ds.dimensions["exp"].size
                exp_idx = self.experiment_index if self.experiment_index >= 0 else auto_idx

                need = len(epochs)
                time_size = ds.dimensions["time"].size
                if need > time_size:
                    ds.variables["values"][:, time_size:need] = float("nan")
                    ds.variables["epochs"][:, time_size:need] = 0
                    ds.variables["timestamps"][:, time_size:need] = -1

                ds.variables["epochs"][exp_idx, :need] = epochs
                ds.variables["values"][exp_idx, :need] = values
                ds.variables["timestamps"][exp_idx, :need] = timestamps
                ds.variables["lengths"][exp_idx] = need
            else:
                cur = ds.dimensions["time"].size
                new = cur + len(epochs)
                ds.variables["epochs"][cur:new] = epochs
                ds.variables["values"][cur:new] = values
                ds.variables["timestamps"][cur:new] = timestamps
        finally:
            ds.close()


    def save_to_zarr(self, zarr_path: str) -> None:
        epochs, values, timestamps = self.collect_lists()

        g = zarr.open(zarr_path, mode="a")
        try:
            if "name" not in g.attrs:
                g.attrs["name"] = self.name
                g.attrs["context"] = str(self.context)
                g.attrs["source"] = str(self.source)

            comp = compressor_to_type(self.use_compressor)

            if self.unify_experiments:
                if "values" not in g:
                    g.create_dataset("epochs", shape=(0, 0), chunks=(1, ZARR_CHUNK_SIZE), dtype="i4",
                                     compressor=comp, fill_value=0, maxshape=(None, None))
                    g.create_dataset("values", shape=(0, 0), chunks=(1, ZARR_CHUNK_SIZE), dtype="f4",
                                     compressor=comp, fill_value=float("nan"), maxshape=(None, None))
                    g.create_dataset("timestamps", shape=(0, 0), chunks=(1, ZARR_CHUNK_SIZE), dtype="i8",
                                     compressor=comp, fill_value=-1, maxshape=(None, None))
                    g.create_dataset("lengths", shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype="i4",
                                     compressor=comp, fill_value=0, maxshape=(None,))

                E, T = g["values"].shape
                auto_idx = E
                exp_idx = self.experiment_index if self.experiment_index >= 0 else auto_idx
                needT = len(epochs)
                newE = max(E, exp_idx + 1)
                newT = max(T, needT)
                if (newE, newT) != (E, T):
                    g["epochs"].resize((newE, newT))
                    g["values"].resize((newE, newT))
                    g["timestamps"].resize((newE, newT))
                    if g["lengths"].shape[0] < newE:
                        g["lengths"].resize((newE,))

                g["epochs"][exp_idx, :needT] = epochs
                g["values"][exp_idx, :needT] = values
                g["timestamps"][exp_idx, :needT] = timestamps
                g["lengths"][exp_idx] = needT
            else:
                if "values" not in g:
                    g.create_dataset("epochs", shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype="i4",
                                     compressor=comp, maxshape=(None,))
                    g.create_dataset("values", shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype="f4",
                                     compressor=comp, maxshape=(None,))
                    g.create_dataset("timestamps", shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype="i8",
                                     compressor=comp, maxshape=(None,))

                cur = g["values"].shape[0]
                need = len(epochs)
                if need > 0:
                    g["values"].resize((cur + need,))
                    g["epochs"].resize((cur + need,))
                    g["timestamps"].resize((cur + need,))
                    g["values"][cur: cur + need] = values
                    g["epochs"][cur: cur + need] = epochs
                    g["timestamps"][cur: cur + need] = timestamps
        finally:
            if hasattr(g, "store") and hasattr(g.store, "close"):
                g.store.close()

    def save_to_csv(self, txt_file: str, csv_separator: str = ",") -> None:
        file_exists = os.path.exists(txt_file)
        with open(txt_file, "a") as f:
            if not file_exists:
                f.write(f"{self.name}{csv_separator}{self.context}{csv_separator}{self.source}\n")
            for epoch, items in self.epochDataList.items():
                for value, ts in items:
                    f.write(f"{epoch}{csv_separator}{value}{csv_separator}{ts}\n")

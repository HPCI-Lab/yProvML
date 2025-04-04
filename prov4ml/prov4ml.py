import os
from typing import Optional
from contextlib import contextmanager
import numcodecs.abc

from prov4ml.constants import PROV4ML_DATA
from prov4ml.utils import energy_utils
from prov4ml.utils import flops_utils
from prov4ml.provenance.metrics_type import MetricsType
from prov4ml.logging_aux import log_execution_start_time, log_execution_end_time, log_artifact, get_context_from_metric_file
from prov4ml.provenance.provenance_graph import create_prov_document
from prov4ml.utils.file_utils import save_prov_file

@contextmanager
def start_run_ctx(
        prov_user_namespace: str,
        experiment_name: Optional[str] = None,
        provenance_save_dir: Optional[str] = None,
        collect_all_processes: Optional[bool] = False,
        save_after_n_logs: Optional[int] = 100,
        rank : Optional[int] = None, 
        metrics_file_type: MetricsType = MetricsType.ZARR,
        use_compression: bool = True,
        chunk_size: Optional[int] = 1000,
        zarr_compressor: Optional[numcodecs.abc.Codec] = None,
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False,
        convert_metrics_to_zarr: Optional[bool] = False,
        convert_metrics_to_netcdf: Optional[bool] = False,
        convert_use_compression: Optional[bool] = True,
        convert_chunk_size: Optional[int] = 1000,
        delete_old_metrics: Optional[bool] = True,
        convert_zarr_compressor: Optional[numcodecs.abc.Codec] = None,
    ): 
    """
    Context manager for starting and ending a run, initializing provenance data collection and optionally creating visualizations.

    Parameters
    ----------
    prov_user_namespace : str
        The user namespace for organizing provenance data.
    experiment_name : Optional[str], optional
        The name of the experiment. If not provided, defaults to None.
    provenance_save_dir : Optional[str], optional
        Directory path for saving provenance data. If not provided, defaults to None.
    collect_all_processes : Optional[bool], optional
        Whether to collect data from all processes. Default is False.
    save_after_n_logs : Optional[int], optional
        Number of logs after which to save metrics. Default is 100.
    rank : Optional[int], optional
        Rank of the current process in a distributed setting. Defaults to None.
    metrics_file_type : MetricsType
        The type of file to save metrics. Defaults to MetricsType.ZARR.
    use_compression : bool, optional
        Whether to use compression when saving metrics. Default is True.
        Available only when `metrics_file_type` is Zarr or NetCDF.
    chunk_size : Optional[int], optional
        The size of chunks to use when saving metrics. Default is 1000.
        Available only when using to Zarr format.
    zarr_compressor : Optional[numcodecs.abc.Codec], optional
        The compressor to use for Zarr format. If not provided, defaults to `Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0)`.
        See https://numcodecs.readthedocs.io/en/latest/compression/index.html for all available compressors.
    create_graph : Optional[bool], optional
        Whether to create a graph representation of the provenance data. Default is False.
    create_svg : Optional[bool], optional
        Whether to create an SVG file for the graph visualization. Default is False. 
        Must be True only if `create_graph` is also True.
    convert_metrics_to_zarr : Optional[bool], optional
        Whether to convert metrics to Zarr format at the end of the run. Default is False.
    convert_metrics_to_netcdf : Optional[bool], optional
        Whether to convert metrics to NetCDF format at the end of the run. Default is False.
    convert_use_compression : Optional[bool], optional
        Whether to use compression when saving metrics during conversion. Default is True.
    convert_chunk_size : Optional[int], optional
        The size of chunks to use when saving metrics during conversion. Default is 1000.
        Available only when converting to Zarr format.
    delete_old_metrics : Optional[bool], optional
        Whether to delete old metrics after conversion. Default is True.
        if False, a new folder will be created with the converted metrics in the root of the experiment.
        Available only if `convert_metrics_to_zarr` or `convert_metrics_to_netcdf` is True.
    convert_zarr_compressor : Optional[numcodecs.abc.Codec], optional
        The compressor to use for Zarr format during conversion.
        If not provided, defaults to `Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0)`.
        See https://numcodecs.readthedocs.io/en/latest/compression/index.html for all available compressors.

    Raises
    ------
    ValueError
        If `create_svg` is True but `create_graph` is False.

    Yields
    ------
    None
        The context manager yields control to the block of code within the `with` statement.

    Notes
    -----
    - The context manager initializes provenance data collection, sets up necessary utilities, and starts tracking.
    - After the block of code within the `with` statement completes, it finalizes the provenance data collection, 
      saves metrics, and optionally generates visualizations and a collection of provenance data.
    """
    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")

    PROV4ML_DATA.init(
        experiment_name=experiment_name, 
        prov_save_path=provenance_save_dir, 
        user_namespace=prov_user_namespace, 
        collect_all_processes=collect_all_processes, 
        save_after_n_logs=save_after_n_logs, 
        rank=rank, 
        metrics_file_type=metrics_file_type,
        use_compression=use_compression,
        chunk_size=chunk_size,
        zarr_compressor=zarr_compressor
    )
   
    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

    yield None#current_run #return the mlflow context manager, same one as mlflow.start_run()

    log_execution_end_time()

    # save remaining metrics
    PROV4ML_DATA.save_all_metrics()

    if convert_metrics_to_zarr:
        PROV4ML_DATA.convert_all_metrics_to_zarr(convert_use_compression=convert_use_compression, chunk_size=convert_chunk_size, delete_old_metrics=delete_old_metrics, convert_zarr_compressor=convert_zarr_compressor)

    if convert_metrics_to_netcdf:
        PROV4ML_DATA.convert_all_metrics_to_netcdf(use_compression=convert_use_compression, delete_old_metrics=delete_old_metrics)

    # add all metrics as artifacts
    for metric in os.listdir(PROV4ML_DATA.METRICS_DIR):
        log_artifact(os.path.join(PROV4ML_DATA.METRICS_DIR, metric), get_context_from_metric_file(metric))

    doc = create_prov_document()

    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'

    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)

    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)
    save_prov_file(doc, path_graph, create_graph, create_svg)

def start_run(
        prov_user_namespace: str,
        experiment_name: Optional[str] = None,
        provenance_save_dir: Optional[str] = None,
        collect_all_processes: Optional[bool] = False,
        save_after_n_logs: Optional[int] = 100,
        rank : Optional[int] = None, 
        metrics_file_type: MetricsType = MetricsType.ZARR,
        use_compression: bool = True,
        chunk_size: Optional[int] = 1000,
        zarr_compressor: Optional[numcodecs.abc.Codec] = None,
    ) -> None:
    """
    Initializes the provenance data collection and sets up various utilities for tracking.

    Parameters
    ----------
    prov_user_namespace : str
        The user namespace to be used for organizing provenance data.
    experiment_name : Optional[str], optional
        The name of the experiment. If not provided, defaults to None.
    provenance_save_dir : Optional[str], optional
        The directory path where provenance data will be saved. If not provided, defaults to None.
    collect_all_processes : Optional[bool], optional
        Whether to collect data from all processes. Default is False.
    save_after_n_logs : Optional[int], optional
        The number of logs after which to save metrics. Default is 100.
    rank : Optional[int], optional
        The rank of the current process in a distributed setting. If not provided, defaults to None.
    metrics_file_type : MetricsType
        The type of file to save metrics. Defaults to MetricsType.ZARR.
    use_compression : bool, optional
        Whether to use compression when saving metrics. Default is True.
        Available only when `metrics_file_type` is Zarr or NetCDF.
    chunk_size : Optional[int], optional
        The size of chunks to use when saving metrics. Default is 1000.
    zarr_compressor : Optional[numcodecs.abc.Codec], optional
        The compressor to use for Zarr format. If not provided, defaults to `Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0)`.
        See https://numcodecs.readthedocs.io/en/latest/compression/index.html for all available compressors.

    Returns
    -------
    None
    """
    PROV4ML_DATA.init(
        experiment_name=experiment_name, 
        prov_save_path=provenance_save_dir, 
        user_namespace=prov_user_namespace, 
        collect_all_processes=collect_all_processes, 
        save_after_n_logs=save_after_n_logs, 
        rank=rank,
        metrics_file_type=metrics_file_type,
        use_compression=use_compression,
        chunk_size=chunk_size,
        zarr_compressor=zarr_compressor
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False,
        convert_metrics_to_zarr: Optional[bool] = False,
        convert_metrics_to_netcdf: Optional[bool] = False,
        convert_use_compression: Optional[bool] = True,
        convert_chunk_size: Optional[int] = 1000,
        delete_old_metrics: Optional[bool] = True,
        convert_zarr_compressor: Optional[numcodecs.abc.Codec] = None,
    ):  
    """
    Finalizes the provenance data collection and optionally creates visualization and provenance collection files.
    
    Parameters
    ----------
    create_graph : Optional[bool], optional
        Whether to create a graph representation of the provenance data. Default is False.
    create_svg : Optional[bool], optional
        Whether to create an SVG file for the graph visualization. Default is False. 
        Must be set to True only if `create_graph` is also True.
    convert_metrics_to_zarr : Optional[bool], optional
        Whether to convert metrics to Zarr format. Default is False.
    convert_metrics_to_netcdf : Optional[bool], optional
        Whether to convert metrics to NetCDF format. Default is False.
    convert_use_compression : Optional[bool], optional
        Whether to use compression when saving metrics. Default is True.
    convert_chunk_size : Optional[int], optional
        The size of chunks to use when saving metrics. Default is 1000.
        Available only when converting to Zarr format.
    delete_old_metrics : Optional[bool], optional
        Whether to delete old metrics after conversion. Default is True.
        if False, a new folder will be created with the converted metrics in the root of the experiment.
        Available only if `convert_metrics_to_zarr` or `convert_metrics_to_netcdf` is True.
    convert_zarr_compressor : Optional[numcodecs.abc.Codec], optional
        The compressor to use for Zarr format during conversion.
        If not provided, defaults to `Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0)`.
        See https://numcodecs.readthedocs.io/en/latest/compression/index.html for all available compressors.

    Raises
    ------
    ValueError
        If `create_svg` is True but `create_graph` is False.

    Returns
    -------
    None
    """
    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")

    if not PROV4ML_DATA.is_collecting: return
    
    log_execution_end_time()

    # save remaining metrics
    PROV4ML_DATA.save_all_metrics()

    if convert_metrics_to_zarr:
        PROV4ML_DATA.convert_all_metrics_to_zarr(convert_use_compression=convert_use_compression, chunk_size=convert_chunk_size, delete_old_metrics=delete_old_metrics, convert_zarr_compressor=convert_zarr_compressor)

    if convert_metrics_to_netcdf:
        PROV4ML_DATA.convert_all_metrics_to_netcdf(convert_use_compression=convert_use_compression, delete_old_metrics=delete_old_metrics)

    # add all metrics as artifacts
    for metric in os.listdir(PROV4ML_DATA.METRICS_DIR):
        log_artifact(os.path.join(PROV4ML_DATA.METRICS_DIR, metric), get_context_from_metric_file(metric))

    doc = create_prov_document()
   
    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'
    
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)
    
    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)
    save_prov_file(doc, path_graph, create_graph, create_svg)


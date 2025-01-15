import os
from typing import Optional
from contextlib import contextmanager

from prov4ml.constants import PROV4ML_DATA
from prov4ml.utils import energy_utils
from prov4ml.utils import flops_utils
from prov4ml.provenance.metrics_type import MetricsType
from prov4ml.logging_aux import log_execution_start_time, log_execution_end_time
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
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False, 
        metrics_file_type: MetricsType = MetricsType.ZARR,
        use_compression: bool = True
    ): 
    """
    Context manager for starting and ending a run, initializing provenance data collection and optionally creating visualizations.

    Parameters:
    -----------
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
    create_graph : Optional[bool], optional
        Whether to create a graph representation of the provenance data. Default is False.
    create_svg : Optional[bool], optional
        Whether to create an SVG file for the graph visualization. Default is False. 
        Must be True only if `create_graph` is also True.
    create_provenance_collection : Optional[bool], optional
        Whether to create a collection of provenance data from all runs. Default is False.
    metrics_file_type : MetricsType
        The type of file to save metrics. Defaults to MetricsType.ZARR.

    Raises:
    -------
    ValueError
        If `create_svg` is True but `create_graph` is False.

    Yields:
    -------
    None
        The context manager yields control to the block of code within the `with` statement.

    Notes:
    ------
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
        use_compression=use_compression
    )
   
    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

    yield None#current_run #return the mlflow context manager, same one as mlflow.start_run()

    log_execution_end_time()

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
        use_compression: bool = True
    ) -> None:
    """
    Initializes the provenance data collection and sets up various utilities for tracking.

    Parameters:
    -----------
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

    Returns:
    --------
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
        use_compression=use_compression
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False, 
    ):  
    """
    Finalizes the provenance data collection and optionally creates visualization and provenance collection files.

    Parameters:
    -----------
    create_graph : Optional[bool], optional
        Whether to create a graph representation of the provenance data. Default is False.
    create_svg : Optional[bool], optional
        Whether to create an SVG file for the graph visualization. Default is False. 
        Must be set to True only if `create_graph` is also True.
    create_provenance_collection : Optional[bool], optional
        Whether to create a collection of provenance data from all runs. Default is False.

    Raises:
    -------
    ValueError
        If `create_svg` is True but `create_graph` is False.

    Returns:
    --------
    None
    """

    if not PROV4ML_DATA.is_collecting: return
    
    log_execution_end_time()

    # save remaining metrics
    PROV4ML_DATA.save_all_metrics()

    doc = create_prov_document()
   
    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'
    
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)
    
    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)
    save_prov_file(doc, path_graph, create_graph, create_svg)


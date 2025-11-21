import os
from typing import Optional, Union

from yprov4ml.constants import PROV4ML_DATA
from yprov4ml.utils import energy_utils
from yprov4ml.utils import flops_utils
from yprov4ml.logging_aux import log_execution_start_time, log_execution_end_time
from yprov4ml.provenance.provenance_graph import create_prov_document, create_rocrate_in_dir
from yprov4ml.utils.file_utils import save_prov_file
from yprov4ml.datamodel.metric_type import MetricsType
from yprov4ml.datamodel.compressor_type import CompressorType

def start_run(
        experiment_name: str,
        prov_user_namespace: Optional[str] = None,
        provenance_save_dir: Optional[str] = None,
        collect_all_processes: Optional[bool] = False,
        save_after_n_logs: Optional[int] = 100,
        rank : Optional[int] = None, 
        disable_codecarbon : Optional[bool] = False,
        metrics_file_type: MetricsType = MetricsType.CSV,
        csv_separator : str = ",", 
        use_compressor: Optional[Union[CompressorType, bool]] = None,
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

    Returns:
    --------
    None
    """
    PROV4ML_DATA.start_run(
        experiment_name=experiment_name, 
        prov_save_path=provenance_save_dir, 
        user_namespace=prov_user_namespace, 
        collect_all_processes=collect_all_processes, 
        save_after_n_logs=save_after_n_logs, 
        rank=rank, 
        metrics_file_type=metrics_file_type,
        csv_separator=csv_separator,
        use_compressor=use_compressor,
    )

    if not disable_codecarbon: 
        energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False, 
        crate_ro_crate: Optional[bool]=False,
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

    found = False
    for root, _, filenames in os.walk('.'):
        for filename in filenames:
            if filename == "requirements.txt": 
                PROV4ML_DATA.add_artifact("requirements", os.path.join(root, filename), step=0, context=None, is_input=True)
                found = True
            if found: break
        if found: break

    PROV4ML_DATA.save_all_metrics()

    doc = create_prov_document()
   
    graph_filename = f'prov_{PROV4ML_DATA.PROV_JSON_NAME}.json'
    
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)
    
    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)
    save_prov_file(doc, path_graph, create_graph, create_svg)

    if crate_ro_crate: 
        create_rocrate_in_dir(PROV4ML_DATA.EXPERIMENT_DIR)

import os
from typing import Optional, Union

from prov4ml.constants import PROV4ML_DATA
from prov4ml.utils import energy_utils
from prov4ml.utils import flops_utils
from prov4ml.logging_aux import log_execution_start_time, log_execution_end_time
from prov4ml.provenance.provenance_graph import create_prov_document, create_rocrate_in_dir
from prov4ml.utils.file_utils import save_prov_file
from prov4ml.datamodel.metric_type import MetricsType
from prov4ml.datamodel.compressor_type import CompressorType

def start_run(
        prov_user_namespace: str,
        experiment_name: str,
        provenance_save_dir: Optional[str] = None,
        collect_all_processes: Optional[bool] = False,
        save_after_n_logs: Optional[int] = 100,
        rank: Optional[int] = None,
        disable_codecarbon: Optional[bool] = False,
        metrics_file_type: MetricsType = MetricsType.CSV,
        csv_separator: str = ",",
        use_compressor: Optional[Union[CompressorType, bool]] = None,
        unify_experiments: bool = False,                     # NEW/CHANGE
    ) -> None:
    """
    Initializes provenance collection and runtime counters.

    unify_experiments:
      - CSV: no change (still per-run files).
      - NetCDF/Zarr: appends rows into a shared metric file and writes an
        additional 'experiment' dimension/column (run index) so multiple runs
        can live in one file.                                      # NEW/CHANGE
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
        unify_experiments=unify_experiments,                 # NEW/CHANGE
    )

    if not disable_codecarbon:
        energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()


def end_run(
        create_graph: Optional[bool] = False,
        create_svg: Optional[bool] = False,
        crate_ro_crate: Optional[bool] = False,              # keeping your flag name as-is
    ):
    """
    Finalizes provenance collection, flushes metrics, and writes the PROV doc.
    """
    if not PROV4ML_DATA.is_collecting:
        return

    log_execution_end_time()

    # Try to capture requirements.txt once
    found = False
    for root, _, filenames in os.walk('./'):
        for filename in filenames:
            if filename == "requirements.txt":
                PROV4ML_DATA.add_artifact(
                    "requirements",
                    os.path.join(root, filename),
                    step=0,
                    context=None,
                    is_input=True
                )
                found = True
                break
        if found:
            break

    # Flush any buffered metrics to disk (handles experiment dimension if enabled)
    PROV4ML_DATA.save_all_metrics()

    # Build the provenance document
    doc = create_prov_document()

    graph_filename = f'prov_{PROV4ML_DATA.PROV_JSON_NAME}.json'

    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)

    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)
    save_prov_file(doc, path_graph, create_graph, create_svg)

    if crate_ro_crate:
        create_rocrate_in_dir(PROV4ML_DATA.EXPERIMENT_DIR)

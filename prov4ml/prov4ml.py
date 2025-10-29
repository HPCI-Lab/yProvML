import os, glob, re, time, uuid
from typing import Optional, Union

from prov.model import ProvDocument

from prov4ml.constants import PROV4ML_DATA
from prov4ml.utils import energy_utils
from prov4ml.utils import flops_utils
from prov4ml.logging_aux import log_execution_start_time, log_execution_end_time
from prov4ml.provenance.provenance_graph import create_prov_document, create_rocrate_in_dir
from prov4ml.utils.file_utils import save_prov_file
from prov4ml.datamodel.metric_type import MetricsType
from prov4ml.datamodel.compressor_type import CompressorType


def _next_unified_suffix(experiment_dir: str, experiment_name: str) -> str:
    """
    Generate a unique suffix for per-run filenames when unify_experiments=True.
    Looks for files like: prov_<experiment_name>_uNNN.json and picks NNN+1.
    Falls back to timestamp/uuid if a clash occurs.
    """
    pattern = os.path.join(experiment_dir, f"prov_{experiment_name}_u*.json")
    max_idx = -1
    for p in glob.glob(pattern):
        m = re.match(
            rf"prov_{re.escape(experiment_name)}_u(\d+)\.json$",
            os.path.basename(p),
        )
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass
    candidate = f"u{max_idx+1:03d}"
    test_path = os.path.join(experiment_dir, f"prov_{experiment_name}_{candidate}.json")
    if os.path.exists(test_path):
        candidate = f"u{int(time.time())}_{str(uuid.uuid4())[:8]}"
    return candidate


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
        unify_experiments: bool = False,  # NEW/CHANGE
    ) -> None:
    """
    Initializes provenance collection and runtime counters.

    unify_experiments:
      - CSV: no change (still per-run files).
      - NetCDF/Zarr: appends rows into a shared metric file and writes an
        additional 'experiment' dimension/column (run index) so multiple runs
        can live in one file.
      - PROV-JSON: in addition to per-run JSONs, maintain a cumulative merged file.
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
        unify_experiments=unify_experiments,  # NEW/CHANGE
    )

    # --- prepare safe filenames/paths for PROV when unifying experiments ---
    if PROV4ML_DATA.is_collecting:
        # Unique per-run suffix for file names (keeps RUN_ID semantics intact)
        if PROV4ML_DATA.unify_experiments:
            PROV4ML_DATA.PER_RUN_SUFFIX = _next_unified_suffix(
                PROV4ML_DATA.EXPERIMENT_DIR,
                PROV4ML_DATA.PROV_JSON_NAME,
            )
        else:
            PROV4ML_DATA.PER_RUN_SUFFIX = str(PROV4ML_DATA.RUN_ID)

        # Dynamic merged filename (env-overridable) and directory selection
        base_dir = getattr(PROV4ML_DATA, "UNIFY_BASE_EXPERIMENT_DIR", None) or PROV4ML_DATA.EXPERIMENT_DIR
        prefix = os.getenv("PROV4ML_MERGED_PREFIX", "prov")
        suffix = os.getenv("PROV4ML_MERGED_SUFFIX", "merged")
        include_rank = os.getenv("PROV4ML_MERGED_INCLUDE_RANK", "0") == "1"

        name_parts = [prefix, PROV4ML_DATA.CLEAN_EXPERIMENT_NAME]
        if include_rank and PROV4ML_DATA.global_rank is not None:
            name_parts.append(f"GR{PROV4ML_DATA.global_rank}")
        name_parts.append(suffix)
        merged_filename = "_".join(name_parts) + ".json"

        PROV4ML_DATA.PROV_MERGED_PATH = os.path.join(base_dir, merged_filename)
    # -----------------------------------------------------------------------

    if not disable_codecarbon:
        energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()


def end_run(
        create_graph: Optional[bool] = False,
        create_svg: Optional[bool] = False,
        crate_ro_crate: Optional[bool] = False,  # keeping your flag name as-is
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

    # Build the provenance document for THIS run
    doc = create_prov_document()

    # Ensure experiment dir exists
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)

    # --- per-run PROV filename (unique when unifying) ---
    # Example: prov_<PROV_JSON_NAME>_u000.json
    per_run_filename = f"prov_{PROV4ML_DATA.PROV_JSON_NAME}_{PROV4ML_DATA.PER_RUN_SUFFIX}.json"
    per_run_path = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, per_run_filename)

    # Save per-run PROV (and optionally DOT/SVG) to the unique path
    save_prov_file(doc, per_run_path, create_graph, create_svg)

    # --- cumulative merge only when unify_experiments=True ---
    if PROV4ML_DATA.unify_experiments:
        merged_doc = ProvDocument()
        merged_path = PROV4ML_DATA.PROV_MERGED_PATH or os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, "prov_merged.json")
        if os.path.exists(merged_path):
            with open(merged_path, "r", encoding="utf-8") as f:
                merged_doc = ProvDocument.deserialize(content=f.read(), format="json")

        # union this run into the cumulative doc
        merged_doc.update(doc)

        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_doc.serialize(indent=2))

    if crate_ro_crate:
        create_rocrate_in_dir(PROV4ML_DATA.EXPERIMENT_DIR)

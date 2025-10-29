import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import prov.model as prov
import pwd
import warnings
from aenum import extend_enum
import subprocess

from prov4ml.datamodel.artifact_data import ArtifactInfo
from prov4ml.datamodel.attribute_type import LoggingItemKind
from prov4ml.datamodel.metric_data import MetricInfo
from prov4ml.datamodel.context import Context
from prov4ml.datamodel.metric_type import MetricsType
from prov4ml.datamodel.compressor_type import (
    CompressorType,
    COMPRESSORS_FOR_ZARR,
)
from prov4ml.utils import funcs
from prov4ml.utils.prov_utils import get_activity, create_activity
from prov4ml.utils.funcs import get_global_rank, get_runtime_type


class Prov4MLData:
    def __init__(self) -> None:
        self.metrics: Dict[(str, Context), MetricInfo] = {}
        self.artifacts: Dict[(str, Context), ArtifactInfo] = {}

        self.PROV_SAVE_PATH = "prov_save_path"
        self.PROV_JSON_NAME = "test_experiment"
        self.EXPERIMENT_DIR = "test_experiment_dir"
        self.ARTIFACTS_DIR = "artifact_dir"
        self.METRIC_DIR = "metric_dir"
        self.USER_NAMESPACE = "user_namespace"
        self.PROV_PREFIX = "yProv4ML"
        self.RUN_ID = 0

        self.global_rank = None
        self.is_collecting = False

        self.save_metrics_after_n_logs = 100
        self.csv_separator = ","

        # unified controls
        self.unify_experiments = False
        self.UNIFY_BASE_EXPERIMENT_DIR: Optional[str] = None
        self.EXPERIMENT_VERSION: int = 0 
        self.PER_RUN_SUFFIX: str = "0"          
        self.PROV_MERGED_PATH: Optional[str] = None

    # ------------------------ lifecycle ------------------------
    def start_run(
        self,
        experiment_name: str,
        prov_save_path: Optional[str] = None,
        user_namespace: Optional[str] = None,
        collect_all_processes: bool = False,
        save_after_n_logs: int = 100,
        rank: Optional[int] = None,
        metrics_file_type: MetricsType = MetricsType.ZARR,
        csv_separator: str = ",",
        use_compressor: Optional[CompressorType] = None,
        unify_experiments: bool = False,
    ) -> None:
        self.global_rank = funcs.get_global_rank() if rank is None else rank
        self.is_collecting = (
            self.global_rank is None or int(self.global_rank) == 0 or collect_all_processes
        )
        if not self.is_collecting:
            return

        self.save_metrics_after_n_logs = save_after_n_logs
        if prov_save_path:
            self.PROV_SAVE_PATH = prov_save_path
        if user_namespace:
            self.USER_NAMESPACE = user_namespace

        if use_compressor in COMPRESSORS_FOR_ZARR and metrics_file_type != MetricsType.ZARR:
            warnings.warn(
                f">start_run(): use_compressor chosen is only compatible with MetricsType.ZARR, "
                f"but saving type is {metrics_file_type}, the compressor chosen will have no effect"
            )
        if metrics_file_type == MetricsType.ZARR and use_compressor not in (None, False) and use_compressor not in COMPRESSORS_FOR_ZARR:
            raise AttributeError(
                f">start_run(): use_compressor chosen is only compatible with MetricsType.ZARR"
            )

        if metrics_file_type == MetricsType.ZARR and use_compressor:
            use_compressor = CompressorType.BLOSC_ZSTD
        elif metrics_file_type in [MetricsType.NETCDF, MetricsType.CSV] and use_compressor:
            use_compressor = CompressorType.ZIP
        if not use_compressor:
            use_compressor = CompressorType.NONE

        # Discover next run id for the experiment name
        if not os.path.exists(self.PROV_SAVE_PATH):
            os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
        if unify_experiments and metrics_file_type in (MetricsType.ZARR, MetricsType.NETCDF):
            self.RUN_ID = 0
        else:
            prev_exps = os.listdir(self.PROV_SAVE_PATH)
            matching_files = [
                int(exp.split("_")[-1].split(".")[0])
                for exp in prev_exps
                if funcs.prov4ml_experiment_matches(experiment_name, exp)
            ]
            self.RUN_ID = max(matching_files) + 1 if len(matching_files) > 0 else 0

        self.CLEAN_EXPERIMENT_NAME = experiment_name
        self.PROV_JSON_NAME = (
            self.CLEAN_EXPERIMENT_NAME + f"_GR{self.global_rank}"
            if self.global_rank is not None
            else experiment_name + "_GR0"
        )
        self.PROV_JSON_NAME = f"{self.PROV_JSON_NAME}_{self.RUN_ID}"

        self.EXPERIMENT_DIR = os.path.join(
            self.PROV_SAVE_PATH, f"{self.CLEAN_EXPERIMENT_NAME}_{self.RUN_ID}"
        )
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, "artifacts")
        self.METRIC_DIR = os.path.join(self.EXPERIMENT_DIR, "metrics")
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.METRIC_DIR, exist_ok=True)

        self.metrics_file_type = metrics_file_type
        self.use_compressor = use_compressor
        self.csv_separator = csv_separator
        self.unify_experiments = bool(unify_experiments)
        self.PER_RUN_SUFFIX = str(self.RUN_ID)
        self.PROV_MERGED_PATH = os.path.join(self.EXPERIMENT_DIR, "prov_merged.json")

        # CSV policy: never unify
        if self.metrics_file_type == MetricsType.CSV and self.unify_experiments:
            raise ValueError(
                "unify_experiments=True is not supported for CSV; use NETCDF or ZARR."
            )

        # For unified ZARR/NETCDF, map all metric IO to the base (run 0) directory
        if self.unify_experiments and self.metrics_file_type in (
            MetricsType.ZARR,
            MetricsType.NETCDF,
        ):
            self.UNIFY_BASE_EXPERIMENT_DIR = self.EXPERIMENT_DIR
            os.makedirs(os.path.join(self.UNIFY_BASE_EXPERIMENT_DIR, "metrics"), exist_ok=True)
            self.EXPERIMENT_VERSION = -1
        else:
            self.UNIFY_BASE_EXPERIMENT_DIR = None
            self.EXPERIMENT_VERSION = 0

        self.init_root_context()

    # ------------------------ PROV scaffolding ------------------------
    def add_ctx(self, rootContext, ctx):
        c = self.root_provenance_doc.activity("context:" + str(ctx))
        c.wasInformedBy(rootContext)
        c.add_attributes({f"{self.PROV_PREFIX}:level": 1})

    def init_root_context(self):
        self.root_provenance_doc = prov.ProvDocument()
        self.root_provenance_doc.add_namespace("context", "context")
        self.root_provenance_doc.add_namespace(self.PROV_PREFIX, self.PROV_PREFIX)
        self.root_provenance_doc.set_default_namespace(self.PROV_JSON_NAME)
        self.root_provenance_doc.add_namespace("prov", "http://www.w3.org/ns/prov#")
        self.root_provenance_doc.add_namespace("xsd", "http://www.w3.org/2000/10/XMLSchema#")
        self.root_provenance_doc.add_namespace("prov-ml", "prov-ml")

        user_ag = self.root_provenance_doc.agent(f"{pwd.getpwuid(os.getuid())[0]}")
        rootContext = self.root_provenance_doc.activity("context:" + self.PROV_JSON_NAME)
        rootContext.add_attributes(
            {
                f"{self.PROV_PREFIX}:level": 0,
                f"{self.PROV_PREFIX}:provenance_path": self.PROV_SAVE_PATH,
                f"{self.PROV_PREFIX}:artifact_uri": self.ARTIFACTS_DIR,
                f"{self.PROV_PREFIX}:experiment_dir": self.EXPERIMENT_DIR,
                f"{self.PROV_PREFIX}:experiment_name": self.PROV_JSON_NAME,
                f"{self.PROV_PREFIX}:run_id": self.RUN_ID,
                f"{self.PROV_PREFIX}:python_version": str(sys.version),
            }
        )
        rootContext.wasAssociatedWith(user_ag)
        self.add_ctx(rootContext, Context.TRAINING)
        self.add_ctx(rootContext, Context.VALIDATION)
        self.add_ctx(rootContext, Context.TESTING)
        self.add_ctx(rootContext, Context.DATASETS)
        self.add_ctx(rootContext, Context.MODELS)

        global_rank = get_global_rank()
        runtime_type = get_runtime_type()
        if runtime_type == "slurm":
            node_rank = os.getenv("SLURM_NODEID", None)
            local_rank = os.getenv("SLURM_LOCALID", None)
            rootContext.add_attributes(
                {
                    f"{self.PROV_PREFIX}:global_rank": str(global_rank),
                    f"{self.PROV_PREFIX}:local_rank": str(local_rank),
                    f"{self.PROV_PREFIX}:node_rank": str(node_rank),
                }
            )
        elif runtime_type == "single_core":
            rootContext.add_attributes({f"{self.PROV_PREFIX}:global_rank": str(global_rank)})

    # ------------------------ provenance ops ------------------------
    def log_input(self, path: str, context: Context, attributes: dict = {}):
        entity = self.root_provenance_doc.entity(path, attributes)
        activity = get_activity(self.root_provenance_doc, "context:" + str(context))
        activity.used(entity)
        return entity

    def log_output(self, path: str, context: Context, attributes: dict = {}):
        entity = self.root_provenance_doc.entity(path, attributes)
        activity = get_activity(self.root_provenance_doc, "context:" + str(context))
        entity.wasGeneratedBy(activity)
        return entity

    def get_git_revision_hash() -> str:  # type: ignore[misc]
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    def get_git_remote_url() -> Optional[str]:  # type: ignore[misc]
        try:
            remote_url = (
                subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .decode()
            )
            return remote_url
        except subprocess.CalledProcessError:
            print("> get_git_remote_url() Repository not found")
            return None

    def add_source_code(self, path: Optional[str]):
        if path is None:
            repo = self.get_git_remote_url()  # type: ignore[attr-defined]
            if repo is not None:
                commit_hash = self.get_git_revision_hash()  # type: ignore[attr-defined]
                self.add_parameter(f"{self.PROV_PREFIX}:source_code", f"{repo}/{commit_hash}")

        else:
            p = Path(path)
            if p.is_file():
                self.add_artifact(
                    p.name.replace(".py", ""),
                    str(p),
                    log_copy_in_prov_directory=True,
                    is_model=False,
                    is_input=True,
                )
                self.add_parameter(
                    f"{self.PROV_PREFIX}:source_code",
                    os.path.join(self.ARTIFACTS_DIR, p.name),
                )
            else:
                self.add_artifact(
                    "source_code",
                    str(p),
                    log_copy_in_prov_directory=True,
                    is_model=False,
                    is_input=True,
                )
                self.add_parameter(
                    f"{self.PROV_PREFIX}:source_code",
                    os.path.join(self.ARTIFACTS_DIR, "source_code"),
                )

    def add_context(self, context: str, is_subcontext_of: Optional[Context] = None):
        is_subcontext_of_name = str(is_subcontext_of).split(".")[-1] if is_subcontext_of else None
        extend_enum(Context, context, str(context))
        new_context = create_activity(self.root_provenance_doc, "context:" + str(getattr(Context, context)))

        if is_subcontext_of_name is not None:
            if not hasattr(Context, is_subcontext_of_name):
                raise Exception(f"{is_subcontext_of_name} not found as a valid Context")
            parent_context = get_activity(
                self.root_provenance_doc, "context:" + str(getattr(Context, is_subcontext_of_name))
            )
        else:
            parent_context = get_activity(self.root_provenance_doc, "context:" + str(Context.EXPERIMENT))

        level = list(parent_context.get_attribute(f"{self.PROV_PREFIX}:level"))[0]
        new_context.wasInformedBy(parent_context)
        new_context.add_attributes({f"{self.PROV_PREFIX}:level": level + 1})

    # ------------------------ data ops ------------------------
    def add_metric(
        self,
        metric: str,
        value: Any,
        step: int,
        context: Optional[Any] = None,
        source: Optional[LoggingItemKind] = None,
    ) -> None:
        if not self.is_collecting:
            return

        if context is None:
            context = self.PROV_JSON_NAME

        if (metric, context) not in self.metrics:
            self.metrics[(metric, context)] = MetricInfo(
                metric,
                context,
                source=source or LoggingItemKind.METRIC,
                use_compressor=self.use_compressor,
                unify_experiments=self.unify_experiments,
                experiment_index=self.EXPERIMENT_VERSION,
            )

        self.metrics[(metric, context)].add_metric(value, step, funcs.get_current_time_millis())

        total = self.metrics[(metric, context)].total_metric_values
        if total % self.save_metrics_after_n_logs == 0:
            self.save_metric_to_file(self.metrics[(metric, context)])

    def add_parameter(self, parameter_name: str, parameter_value: Any, context: Optional[Context] = None) -> None:
        if not self.is_collecting:
            return
        if context is None:
            context = self.PROV_JSON_NAME
        current_activity = get_activity(self.root_provenance_doc, "context:" + str(context))
        current_activity.add_attributes({parameter_name: str(parameter_value)})

    def add_artifact(
        self,
        artifact_name: str,
        artifact_path: str,
        step: Optional[int] = None,
        context: Optional[Any] = None,
        is_input: bool = False,
        log_copy_in_prov_directory: bool = True,
        is_model: bool = False,
    ):
        if not self.is_collecting:
            return

        if context is None:
            context = self.PROV_JSON_NAME

        if not isinstance(artifact_path, str):
            raise AttributeError(
                f">add_artifact({artifact_path}): the parameter \"artifact_path\" has to be a string"
            )

        if log_copy_in_prov_directory:
            try:
                path = Path(artifact_path)
                # create original artefact
                original = self.add_artifact(
                    "Original " + path.name,
                    str(path),
                    log_copy_in_prov_directory=False,
                    is_model=is_model,
                    is_input=is_input,
                )
                copied = self.add_artifact(
                    path.name,
                    os.path.join(self.ARTIFACTS_DIR, path.name),
                    log_copy_in_prov_directory=False,
                    is_model=is_model,
                    is_input=True,
                )
                if original is not None and copied is not None:
                    copied.wasDerivedFrom(original)

                newart_path = os.path.join(self.ARTIFACTS_DIR, path.name)
                if path.is_file():
                    shutil.copy(path, newart_path)
                else:
                    shutil.copytree(path, newart_path)
                artifact_path = newart_path
                return copied
            except Exception as e:
                raise Exception(
                    f">add_artifact: log_copy_in_prov_directory True but invalid path: {artifact_path}. {e}"
                )

        self.artifacts[(artifact_name, context)] = ArtifactInfo(
            artifact_name, artifact_path, step, context=context, is_model=is_model
        )

        attributes = {
            f"{self.PROV_PREFIX}:label": artifact_name,
            f"{self.PROV_PREFIX}:path": artifact_path,
        }
        if is_input:
            attributes.setdefault(f"{self.PROV_PREFIX}:role", "input")
            return self.log_input(artifact_name, context, attributes)
        else:
            attributes.setdefault(f"{self.PROV_PREFIX}:role", "output")
            return self.log_output(artifact_name, context, attributes)

    # ------------------------ queries ------------------------
    def get_artifacts(self) -> List[ArtifactInfo]:
        if not self.is_collecting:
            return []
        return list(self.artifacts.values())

    def get_model_versions(self) -> List[ArtifactInfo]:
        if not self.is_collecting:
            return []
        return [a for a in self.artifacts.values() if a.is_model_version]

    def get_final_model(self) -> Optional[ArtifactInfo]:
        if not self.is_collecting:
            return None
        versions = self.get_model_versions()
        return versions[-1] if versions else None

    # ------------------------ persistence ------------------------
    def save_metric_to_file(self, metric: MetricInfo) -> None:
        if not self.is_collecting:
            return
        target_dir = self.METRIC_DIR
        if (
            self.unify_experiments
            and self.metrics_file_type in (MetricsType.ZARR, MetricsType.NETCDF)
            and self.UNIFY_BASE_EXPERIMENT_DIR
        ):
            target_dir = os.path.join(self.UNIFY_BASE_EXPERIMENT_DIR, "metrics")
        metric.save_to_file(
            target_dir,
            file_type=self.metrics_file_type,
            process=self.global_rank,
            csv_separator=self.csv_separator,
        )

    def save_all_metrics(self) -> None:
        if not self.is_collecting:
            return
        for metric in list(self.metrics.values()):
            self.save_metric_to_file(metric)

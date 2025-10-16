
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import prov.model as prov
import pwd
import warnings
from aenum import extend_enum
import subprocess
import uuid

from yprov4ml.datamodel.artifact_data import ArtifactInfo
from yprov4ml.datamodel.metric_data import MetricInfo
from yprov4ml.datamodel.context import Context
from yprov4ml.datamodel.metric_type import MetricsType
from yprov4ml.datamodel.compressor_type import CompressorType, COMPRESSORS_FOR_ZARR
from yprov4ml.utils import funcs
from yprov4ml.utils.prov_utils import get_or_create_activity
from yprov4ml.utils.funcs import get_global_rank, get_runtime_type

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

    def start_run(
            self, 
            experiment_name: str, 
            prov_save_path: Optional[str] = None, 
            user_namespace: Optional[str] = None, 
            collect_all_processes: bool = False, 
            save_after_n_logs: int = 100, 
            rank: Optional[int] = None, 
            disable_codecarbon : bool = False,
            metrics_file_type: MetricsType = MetricsType.ZARR,
            csv_separator:str = ",", 
            use_compressor: Optional[CompressorType] = None,
        ) -> None:

        self.global_rank = funcs.get_global_rank() if rank is None else rank
        self.is_collecting = self.global_rank is None or int(self.global_rank) == 0 or collect_all_processes
        
        if not self.is_collecting: return

        self.save_metrics_after_n_logs = save_after_n_logs
        if prov_save_path: self.PROV_SAVE_PATH = prov_save_path
        if user_namespace: self.USER_NAMESPACE = user_namespace

        if use_compressor in COMPRESSORS_FOR_ZARR and metrics_file_type != MetricsType.ZARR: 
            warnings.warn(f">start_run(): use_compressor chosen is only compatible with MetricsType.ZARR, but saving type is {metrics_file_type}, the compressor chosen will have no effect")
        if metrics_file_type == MetricsType.ZARR and use_compressor != False and use_compressor not in COMPRESSORS_FOR_ZARR: 
            raise AttributeError(f">start_run(): use_compressor chosen is only compatible with MetricsType.ZARR")

        if metrics_file_type == MetricsType.ZARR and use_compressor:
            use_compressor = CompressorType.BLOSC_ZSTD
        elif metrics_file_type in [MetricsType.NETCDF, MetricsType.CSV] and use_compressor:
            use_compressor = CompressorType.ZIP
        if not use_compressor: 
            use_compressor = CompressorType.NONE

        # look at PROV dir how many experiments are there with the same name
        if not os.path.exists(self.PROV_SAVE_PATH):
            os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
            self.RUN_ID = 0
        else: 
            prev_exps = os.listdir(self.PROV_SAVE_PATH) 
            matching_files = [int(exp.split("_")[-1].split(".")[0]) for exp in prev_exps if funcs.prov4ml_experiment_matches(experiment_name, exp)]
            self.RUN_ID = max(matching_files)+1  if len(matching_files) > 0 else 0

        self.CLEAN_EXPERIMENT_NAME = experiment_name
        self.PROV_JSON_NAME = self.CLEAN_EXPERIMENT_NAME + f"_GR{self.global_rank}" if self.global_rank else experiment_name + f"_GR0"
        self.PROV_JSON_NAME = f"{self.PROV_JSON_NAME}_{self.RUN_ID}"

        self.EXPERIMENT_DIR = os.path.join(self.PROV_SAVE_PATH, f"{self.CLEAN_EXPERIMENT_NAME}_{self.RUN_ID}")
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, f"artifacts_GR{self.global_rank}")
        self.METRIC_DIR = os.path.join(self.EXPERIMENT_DIR, f"metrics_GR{self.global_rank}")

        self.metrics_file_type = metrics_file_type
        self.use_compressor = use_compressor
        self.csv_separator = csv_separator
        self.codecarbon_is_disabled = disable_codecarbon

        self._init_root_context()

        # necessary when spawning threads, 
        # otherwise they get counted as different runs
        # TODO: find better approach
        time.sleep(1)
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.METRIC_DIR, exist_ok=True)

    def _add_ctx(self, rootContext, ctx, source):
        if source is not None: 
            src_context_name = self._format_activity_name(context=ctx, source=None)
            maybe_src_context, created = get_or_create_activity(self.root_provenance_doc, src_context_name)
            if created: 
                maybe_src_context.wasInformedBy(rootContext)

        context_name = self._format_activity_name(context=ctx, source=source)
        c, created = get_or_create_activity(self.root_provenance_doc, context_name)
        if created:         
            if source is not None: 
                c.wasInformedBy(maybe_src_context)
            else: 
                c.wasInformedBy(rootContext)
            c.add_attributes({f'{self.PROV_PREFIX}:level':1})
        return c

    def _init_root_context(self): 
        self.root_provenance_doc = prov.ProvDocument()
        self.root_provenance_doc.add_namespace('context', 'context')
        self.root_provenance_doc.add_namespace(self.PROV_PREFIX, self.PROV_PREFIX)
        self.root_provenance_doc.set_default_namespace(self.PROV_JSON_NAME)
        # self.root_provenance_doc.set_default_namespace(self.USER_NAMESPACE)
        self.root_provenance_doc.add_namespace('prov','http://www.w3.org/ns/prov#')
        self.root_provenance_doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
        self.root_provenance_doc.add_namespace('prov-ml', 'prov-ml')
        # self.provDoc.add_namespace(name,name)

        user_ag = self.root_provenance_doc.agent(f'{pwd.getpwuid(os.getuid())[0]}')
        rootContext, _ = get_or_create_activity(self.root_provenance_doc, "context:"+ self.PROV_JSON_NAME)
        rootContext.add_attributes({
            f'{self.PROV_PREFIX}:level':0, 
            f"{self.PROV_PREFIX}:provenance_path":self.PROV_SAVE_PATH,
            f"{self.PROV_PREFIX}:artifact_uri":self.ARTIFACTS_DIR,
            f"{self.PROV_PREFIX}:experiment_dir":self.EXPERIMENT_DIR,
            f"{self.PROV_PREFIX}:experiment_name":self.PROV_JSON_NAME,
            f"{self.PROV_PREFIX}:run_id":self.RUN_ID,
            f"{self.PROV_PREFIX}:python_version":str(sys.version), 
            f"{self.PROV_PREFIX}:PID":str(uuid.uuid4()), 
        })
        rootContext.wasAssociatedWith(user_ag)

        global_rank = get_global_rank()
        runtime_type = get_runtime_type()
        if runtime_type == "slurm":
            node_rank = os.getenv("SLURM_NODEID", None)
            local_rank = os.getenv("SLURM_LOCALID", None) 
            rootContext.add_attributes({
                f"{self.PROV_PREFIX}:global_rank": str(global_rank),
                f"{self.PROV_PREFIX}:local_rank":str(local_rank),
                f"{self.PROV_PREFIX}:node_rank":str(node_rank),
            })
        elif runtime_type == "single_core":
            rootContext.add_attributes({
                f"{self.PROV_PREFIX}:global_rank":str(global_rank)
            })

        self._add_ctx(f"context:"+self.PROV_JSON_NAME, self.PROV_JSON_NAME, 'std.time')

    def _format_activity_name(self, context : Optional[Context] = None, source: Optional[str]=None): 
        if context is None: context = self.PROV_JSON_NAME
        return f"context:{context}" + (f"-source:{source}" if source else "")

    def _format_artifact_name(self, label : str, context : Optional[Context] = None, source: Optional[str]=None): 
        if context is None: context = self.PROV_JSON_NAME
        return f"{self.PROV_PREFIX}:{label}-context:{context}" + (f"-source:{source}" if source else "")

    def _log_input(self, path : str, context : Context, source: Optional[str]=None, attributes : dict={}) -> prov.ProvEntity:
        entity = self.root_provenance_doc.entity(path, attributes)
        root_ctx = self._format_activity_name(self.PROV_JSON_NAME, None)
        activity = self._add_ctx(root_ctx, context, source)
        activity.used(entity)
        return entity
    
    def _log_output(self, path : str, context : Context, source: Optional[str]=None, attributes : dict={}) -> prov.ProvEntity:
        entity= self.root_provenance_doc.entity(path, attributes)
        root_ctx = self._format_activity_name(self.PROV_JSON_NAME, None)
        activity = self._add_ctx(root_ctx, context, source)
        entity.wasGeneratedBy(activity)
        return entity
    
    def _get_git_revision_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    def _get_git_remote_url() -> Optional[str]:
        try:
            remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.DEVNULL).strip().decode()
            return remote_url
        except subprocess.CalledProcessError:
            print("> get_git_remote_url() Repository not found")
            return None  # No remote found

    def add_source_code(self, path: str): 
        if path is None:
            repo = _get_git_remote_url()
            if repo is not None:
                commit_hash = _get_git_revision_hash()
                log_param(f"{PROV4ML_DATA.PROV_PREFIX}:source_code", f"{repo}/{commit_hash}")
        else:
            p = Path(path)
            if p.is_file():
                self.add_artifact(p.name.replace(".py", ""), str(p), log_copy_in_prov_directory=True, is_model=False, is_input=True)
                self.add_parameter(f"{self.PROV_PREFIX}:source_code", os.path.join(self.ARTIFACTS_DIR, p.name))
            else:
                self.add_artifact("source_code", str(p), log_copy_in_prov_directory=True, is_model=False, is_input=True)
                self.add_parameter(f"{self.PROV_PREFIX}:source_code", os.path.join(self.ARTIFACTS_DIR, "source_code"))

    def add_metric(
        self, 
        metric: str, 
        value: Any, 
        step: Optional[int] = None, 
        context: Optional[Any] = None, 
        source: Optional[str] = None, 
    ) -> None:
        if not self.is_collecting: return

        if context is None: context = self.PROV_JSON_NAME
        if step is None: step = 0

        if (metric, context) not in self.metrics:
            self.metrics[(metric, context)] = MetricInfo(metric, context, source=source, use_compressor=self.use_compressor)
        
        self.metrics[(metric, context)].add_metric(value, step, funcs.get_current_time_millis())

        total_metrics_values = self.metrics[(metric, context)].total_metric_values
        if total_metrics_values % self.save_metrics_after_n_logs == 0:
            self.save_metric_to_file(self.metrics[(metric, context)])

    def add_parameter(
            self, 
            parameter_name: str, 
            parameter_value: Any, 
            context : Optional[Context] = None, 
            source : Optional[str] = None, 
        ) -> None:
        """
        Adds a parameter to the provenance data.

        Parameters:
        -----------
        parameter : str
            The name of the parameter to add.
        value : Any
            The value of the parameter to add.

        Returns:
        --------
        None
        """
        if not self.is_collecting: return

        if context is None: context = self.PROV_JSON_NAME

        root_ctx = self._format_activity_name(self.PROV_JSON_NAME, None)
        current_activity = self._add_ctx(root_ctx, context, source)
        current_activity.add_attributes({parameter_name:str(parameter_value)})

    def add_artifact(
        self, 
        artifact_name: str, 
        artifact_path: str, 
        step: Optional[int] = None, 
        context: Optional[Any] = None,
        source: Optional[str] = None,
        is_input : bool = False, 
        log_copy_in_prov_directory : bool = True, 
        is_model = False, 
    ) -> prov.ProvEntity:
        if not self.is_collecting: return

        if context is None: context = self.PROV_JSON_NAME

        if not isinstance(artifact_path, str): 
            print(artifact_path)
            raise AttributeError(f">add_artifact({artifact_path}): the parameter \"artifact_path\" has to be a string")

        if log_copy_in_prov_directory: 
            try: 
                path = Path(artifact_path)
                original = self.add_artifact("Original_" + path.name, str(path), log_copy_in_prov_directory=False, is_model=is_model, is_input=is_input, source=source, context=context)
                copied = self.add_artifact(path.name, os.path.join(self.ARTIFACTS_DIR, path.name), log_copy_in_prov_directory=False, is_model=is_model, is_input=True, source=source, context=context)
                copied.wasDerivedFrom(original)

                newart_path = os.path.join(self.ARTIFACTS_DIR, path.name)
                if path.is_file():
                    shutil.copy(path, newart_path)
                else:  
                    shutil.copytree(path, newart_path)
                artifact_path = newart_path

                return copied
            except: 
                Exception(f">add_artifact: log_copy_in_prov_directory was True but value is not a valid Path: {artifact_path}")

        artifact_name = self._format_artifact_name(artifact_name, context, source)
        self.artifacts[(artifact_name, context)] = ArtifactInfo(artifact_name, artifact_path, step, context=context, source=source, is_model=is_model)

        attributes = {
            f'{self.PROV_PREFIX}:label': artifact_name, 
            f'{self.PROV_PREFIX}:path': artifact_path,
        }

        if is_input: 
            attributes.setdefault(f'{self.PROV_PREFIX}:role','input')
            return self._log_input(artifact_name, context, source, attributes)
        else: 
            attributes.setdefault(f'{self.PROV_PREFIX}:role', 'output')
            return self._log_output(artifact_name, context, source, attributes)

    def save_metric_to_file(self, metric: MetricInfo) -> None:
        if not self.is_collecting: return
        metric.save_to_file(self.METRIC_DIR, file_type=self.metrics_file_type, process=self.global_rank, csv_separator=self.csv_separator)

    def save_all_metrics(self) -> None:
        if not self.is_collecting: return

        for metric in self.metrics.values():
            self.save_metric_to_file(metric)



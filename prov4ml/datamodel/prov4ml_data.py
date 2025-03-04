
import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import prov.model as prov
import pwd
from aenum import Enum, extend_enum

from prov4ml.datamodel.artifact_data import ArtifactInfo
from prov4ml.datamodel.attribute_type import LoggingItemKind
from prov4ml.datamodel.cumulative_metrics import CumulativeMetric, FoldOperation
from prov4ml.datamodel.metric_data import MetricInfo
from prov4ml.datamodel.context import Contexts
from prov4ml.utils import funcs
from prov4ml.utils.prov_utils import get_activity, create_activity
from prov4ml.utils.funcs import get_global_rank, get_runtime_type

class Prov4MLData:
    def __init__(self) -> None:
        self.metrics: Dict[(str, Contexts), MetricInfo] = {}
        self.artifacts: Dict[(str, Contexts), ArtifactInfo] = {}
        self.cumulative_metrics: Dict[str, CumulativeMetric] = {}

        self.PROV_SAVE_PATH = "prov_save_path"
        self.EXPERIMENT_NAME = "test_experiment"
        self.EXPERIMENT_DIR = "test_experiment_dir"
        self.ARTIFACTS_DIR = "artifact_dir"
        self.USER_NAMESPACE = "user_namespace"
        self.PROV_PREFIX = "yProv4ML"
        self.RUN_ID = 0

        self.global_rank = None
        self.is_collecting = False

        self.save_metrics_after_n_logs = 100
        self.TMP_SEP = "\t"

    def start_run(
            self, 
            experiment_name: str, 
            prov_save_path: Optional[str] = None, 
            user_namespace: Optional[str] = None, 
            collect_all_processes: bool = False, 
            save_after_n_logs: int = 100, 
            rank: Optional[int] = None
        ) -> None:

        self.global_rank = funcs.get_global_rank() if rank is None else rank
        self.is_collecting = self.global_rank is None or int(self.global_rank) == 0 or collect_all_processes
        
        if not self.is_collecting: return

        self.save_metrics_after_n_logs = save_after_n_logs
        if prov_save_path: self.PROV_SAVE_PATH = prov_save_path
        if user_namespace: self.USER_NAMESPACE = user_namespace

        # look at PROV dir how many experiments are there with the same name
        if not os.path.exists(self.PROV_SAVE_PATH):
            os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
        prev_exps = os.listdir(self.PROV_SAVE_PATH) if self.PROV_SAVE_PATH else []

        self.EXPERIMENT_NAME = experiment_name + f"_GR{self.global_rank}" if self.global_rank else experiment_name
        self.RUN_ID = len([exp for exp in prev_exps if funcs.prov4ml_experiment_matches(experiment_name, exp)]) 
        self.EXPERIMENT_DIR = os.path.join(self.PROV_SAVE_PATH, experiment_name + f"_{self.RUN_ID}")
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, "artifacts")
        self.EXPERIMENT_NAME = f"{self.EXPERIMENT_NAME}_{self.RUN_ID}"

        self._init_root_context()

    def _add_ctx(self, rootContext, ctx):
        c = self.root_provenance_doc.activity("context:"+ str(ctx))
        c.wasInformedBy(rootContext)
        c.add_attributes({f'{self.PROV_PREFIX}:level':1})

    def _init_root_context(self): 
        self.root_provenance_doc = prov.ProvDocument()
        self.root_provenance_doc.add_namespace('context', 'context')
        self.root_provenance_doc.add_namespace(self.PROV_PREFIX, self.PROV_PREFIX)
        self.root_provenance_doc.set_default_namespace(self.EXPERIMENT_NAME)
        # self.root_provenance_doc.set_default_namespace(self.USER_NAMESPACE)
        self.root_provenance_doc.add_namespace('prov','http://www.w3.org/ns/prov#')
        self.root_provenance_doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
        self.root_provenance_doc.add_namespace('prov-ml', 'prov-ml')
        # self.provDoc.add_namespace(name,name)

        user_ag = self.root_provenance_doc.agent(f'{pwd.getpwuid(os.getuid())[0]}')
        rootContext = self.root_provenance_doc.activity("context:"+ self.EXPERIMENT_NAME)
        rootContext.add_attributes({
            f'{self.PROV_PREFIX}:level':0, 
            f"{self.PROV_PREFIX}:provenance_path":self.PROV_SAVE_PATH,
            f"{self.PROV_PREFIX}:artifact_uri":self.ARTIFACTS_DIR,
            f"{self.PROV_PREFIX}:run_id":self.RUN_ID,
            f"{self.PROV_PREFIX}:python_version":str(sys.version), 
        })
        rootContext.wasAssociatedWith(user_ag)
        self._add_ctx(rootContext, Contexts.TRAINING)
        self._add_ctx(rootContext, Contexts.VALIDATION)
        self._add_ctx(rootContext, Contexts.TESTING)
        self._add_ctx(rootContext, Contexts.DATASETS)
        self._add_ctx(rootContext, Contexts.MODELS)

        global_rank = get_global_rank()
        runtime_type = get_runtime_type()
        if runtime_type == "slurm":
            node_rank = os.getenv("SLURM_NODEID", None)
            local_rank = os.getenv("SLURM_LOCALID", None) 
            rootContext.add_attributes({
                "prov-ml:global_rank": str(global_rank),
                "prov-ml:local_rank":str(local_rank),
                "prov-ml:node_rank":str(node_rank),
            })
        elif runtime_type == "single_core":
            rootContext.add_attributes({
                "prov-ml:global_rank":str(global_rank)
            })


        
    def _log_input(self, path : str, context : Contexts, attributes : dict={}) -> prov.ProvEntity:
        entity = self.root_provenance_doc.entity(path, attributes)
        activity = get_activity(self.root_provenance_doc,"context:"+str(context))
        activity.used(entity)
        return entity
    
    def _log_output(self, path : str, context : Contexts, attributes : dict={}) -> prov.ProvEntity:
        entity= self.root_provenance_doc.entity(path, attributes)
        activity = get_activity(self.root_provenance_doc,"context:"+str(context))
        entity.wasGeneratedBy(activity)
        return entity

    def add_context(self, context : str, is_subcontext_of: Optional[Contexts] = None):         
        extend_enum(Contexts, context, str(context))
        new_context = create_activity(self.root_provenance_doc,'context:'+ str(getattr(Contexts, context)))

        if is_subcontext_of is not None:
            if not hasattr(Contexts, is_subcontext_of): 
                raise Exception(f"{is_subcontext_of} not found as a valid Context")
            parent_context = get_activity(self.root_provenance_doc,'context:' + str(getattr(Contexts, is_subcontext_of)))
        else: 
            parent_context = get_activity(self.root_provenance_doc,'context:'+ str(Contexts.EXPERIMENT))

        level=list(parent_context.get_attribute('yProv4ML:level'))[0]
        new_context.wasInformedBy(parent_context)
        new_context.add_attributes({'yProv4ML:level':level+1})

    def add_metric(
        self, 
        metric: str, 
        value: Any, 
        step: int, 
        context: Optional[Any] = None, 
        source: Optional[LoggingItemKind] = None, 
    ) -> None:
        if not self.is_collecting: return

        if context is None: 
            context = self.EXPERIMENT_NAME

        if (metric, context) not in self.metrics:
            self.metrics[(metric, context)] = MetricInfo(metric, context, source=source)
        
        self.metrics[(metric, context)].add_metric(value, step, funcs.get_current_time_millis())

        if metric in self.cumulative_metrics:
            self.cumulative_metrics[metric].update(value)

        total_metrics_values = self.metrics[(metric, context)].total_metric_values
        if total_metrics_values % self.save_metrics_after_n_logs == 0:
            self.save_metric_to_file(self.metrics[(metric, context)])

    def add_cumulative_metric(self, label: str, value: Any, fold_operation: FoldOperation) -> None:
        """
        Adds a cumulative metric to the provenance data.

        Parameters:
        -----------
        label : str
            The label of the cumulative metric.
        value : Any
            The initial value of the cumulative metric.
        fold_operation : FoldOperation
            The operation used to fold new values into the cumulative metric.

        Returns:
        --------
        None
        """
        if not self.is_collecting: return

        self.cumulative_metrics[label] = CumulativeMetric(label, value, fold_operation)

    def add_parameter(
            self, 
            parameter_name: str, 
            parameter_value: Any, 
            context : Optional[Contexts] = None, 
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

        if context is None: 
            context = self.EXPERIMENT_NAME

        current_activity = get_activity(self.root_provenance_doc,"context:"+ str(context))
        current_activity.add_attributes({parameter_name:str(parameter_value)})

    def add_artifact(
        self, 
        artifact_name: str, 
        artifact_path: str, 
        step: Optional[int] = None, 
        context: Optional[Any] = None,
        is_input : bool = False, 
        log_copy_in_prov_directory : bool = True, 
        is_model = False, 
    ) -> prov.ProvEntity:
        if not self.is_collecting: return

        if context is None: 
            context = self.EXPERIMENT_NAME

        if log_copy_in_prov_directory: 
            try: 
                path = Path(artifact_path)
            except: 
                Exception(f">add_artifact: log_copy_in_prov_directory was True but value is not a valid Path: {artifact_path}")

            newart_path = self.ARTIFACTS_DIR + "/" + path.name
            if path.is_file():
                shutil.copy(path, newart_path)
            else:  
                shutil.copytree(path, newart_path)
            artifact_path = newart_path

        self.artifacts[(artifact_name, context)] = ArtifactInfo(artifact_name, artifact_path, step, context=context, is_model=is_model)

        attributes = {
            'prov:label': artifact_name, 
            'prov:path': artifact_path,
        }
        if is_input: 
            attributes.setdefault('yProv4ML:role','input')
            return self._log_input(artifact_name, context, attributes)
        else: 
            attributes.setdefault('yProv4ML:role', 'output')
            return self._log_output(artifact_name, context, attributes)

    def get_artifacts(self) -> List[ArtifactInfo]:
        """
        Returns a list of all artifacts.

        Returns:
            List[ArtifactInfo]: A list of artifact information objects.
        """
        if not self.is_collecting: return

        return list(self.artifacts.values())
    
    def get_model_versions(self) -> List[ArtifactInfo]:
        """
        Returns a list of all model version artifacts.

        Returns:
            List[ArtifactInfo]: A list of model version artifact information objects.
        """
        if not self.is_collecting: return

        return [artifact for artifact in self.artifacts.values() if artifact.is_model_version]
    
    def get_final_model(self) -> Optional[ArtifactInfo]:
        """
        Returns the most recent model version artifact.

        Returns:
            Optional[ArtifactInfo]: The most recent model version artifact information object, or None if no model versions exist.
        """
        if not self.is_collecting: return

        model_versions = self.get_model_versions()
        if model_versions:
            return model_versions[-1]
        return None

    def save_metric_to_file(self, metric: MetricInfo) -> None:
        """
        Saves a metric to a temporary file.

        Parameters:
        --------
            metric (MetricInfo): The metric to save.
        
        Returns:
        --------
        None
        """
        if not self.is_collecting: return

        if not os.path.exists(self.ARTIFACTS_DIR):
            os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

        metric.save_to_file(self.ARTIFACTS_DIR, process=self.global_rank, sep=self.TMP_SEP)

    def save_all_metrics(self) -> None:
        """
        Saves all tracked metrics to temporary files.

        Returns:
        --------
        None
        """
        if not self.is_collecting: return

        for metric in self.metrics.values():
            self.save_metric_to_file(metric)



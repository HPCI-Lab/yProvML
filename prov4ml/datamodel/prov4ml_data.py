
import os
from typing import Any, Dict, List, Optional

from prov4ml.datamodel.artifact_data import ArtifactInfo
from prov4ml.datamodel.attribute_type import LoggingItemKind
from prov4ml.datamodel.parameter_data import ParameterInfo
from prov4ml.datamodel.cumulative_metrics import CumulativeMetric, FoldOperation
from prov4ml.datamodel.metric_data import MetricInfo
from prov4ml.provenance.context import Context
from prov4ml.provenance.metrics_type import MetricsType
from prov4ml.utils import funcs

class Prov4MLData:
    """
    A class for managing and storing provenance information for machine learning experiments.

    Attributes:
    -----------
    metrics : dict
        A dictionary mapping tuples of (str, Context) to MetricInfo objects, 
        representing the metrics tracked during the experiment.
    parameters : dict
        A dictionary mapping strings to ParameterInfo objects, representing 
        the parameters used in the experiment.
    artifacts : dict
        A dictionary mapping tuples of (str, Context) to ArtifactInfo objects, 
        representing the artifacts generated during the experiment.
    cumulative_metrics : dict
        A dictionary mapping strings to CumulativeMetric objects, representing 
        cumulative metrics aggregated over the experiment.
    PROV_SAVE_PATH : str
        The path where the provenance data will be saved.
    EXPERIMENT_NAME : str
        The name of the experiment.
    EXPERIMENT_DIR : str
        The directory where the experiment's data is stored.
    ARTIFACTS_DIR : str
        The directory where the artifacts are stored.
    METRICS_DIR : str
        The directory where the metrics are stored.
    USER_NAMESPACE : str
        The user namespace for organizing experiments.
    RUN_ID : int
        The identifier for the current run of the experiment.
    METRICS_FILE_TYPE : MetricsType
        The file type used to store metrics.
    global_rank : optional
        The global rank of the current process in a distributed setting.
    is_collecting : bool
        A flag indicating whether the provenance data collection is active.
    save_metrics_after_n_logs : int
        The number of logs after which metrics are saved.

    Methods:
    --------
    __init__() -> None
        Initializes the Prov4MLData class with default values.

    init(experiment_name: str, prov_save_path: Optional[str] = None, user_namespace: Optional[str] = None, 
         collect_all_processes: bool = False, save_after_n_logs: int = 100, rank: Optional[int] = None) -> None
        Initializes the experiment with the given parameters and sets up directories and metadata.

    add_metric(metric: str, value: Any, step: int, context: Optional[Any] = None, source: LoggingItemKind = None,
                timestamp = None) -> None
        Adds a metric to the provenance data.

    add_cumulative_metric(label: str, value: Any, fold_operation: FoldOperation) -> None
        Adds a cumulative metric to the provenance data.

    add_parameter(parameter: str, value: Any) -> None
        Adds a parameter to the provenance data.

    add_artifact(artifact_name: str, value: Any = None, step: Optional[int] = None, context: Optional[Any] = None,
                timestamp: Optional[int] = None) -> None
        Adds an artifact to the artifacts dictionary.

    get_artifacts() -> List[ArtifactInfo]
        Returns a list of all artifacts.

    get_model_versions() -> List[ArtifactInfo]
        Returns a list of all model version artifacts.

    get_final_model() -> Optional[ArtifactInfo]
        Returns the most recent model version artifact.

    save_metric_to_file(metric: MetricInfo) -> None
        Saves a metric to a temporary file.

    save_all_metrics() -> None
        Saves all tracked metrics to temporary files.
    """
    def __init__(self) -> None:
        self.metrics: Dict[(str, Context), MetricInfo] = {}
        self.parameters: Dict[str, ParameterInfo] = {}
        self.artifacts: Dict[(str, Context), ArtifactInfo] = {}
        self.cumulative_metrics: Dict[str, CumulativeMetric] = {}

        self.PROV_SAVE_PATH = "prov_save_path"
        self.EXPERIMENT_NAME = "test_experiment"
        self.EXPERIMENT_DIR = "test_experiment_dir"
        self.ARTIFACTS_DIR = "artifact_dir"
        self.METRICS_DIR = "metrics_dir"
        self.USER_NAMESPACE = "user_namespace"
        self.RUN_ID = 0
        self.METRICS_FILE_TYPE: MetricsType = MetricsType.ZARR

        self.global_rank = None
        self.is_collecting = False

        self.save_metrics_after_n_logs = 100

    def init(
            self, 
            experiment_name: str, 
            prov_save_path: Optional[str] = None, 
            user_namespace: Optional[str] = None, 
            collect_all_processes: bool = False, 
            save_after_n_logs: int = 100, 
            rank: Optional[int] = None,
            metrics_file_type: MetricsType = MetricsType.ZARR
        ) -> None:
        """
        Initializes the experiment with the given parameters and sets up directories and metadata.

        Parameters:
        -----------
        experiment_name : str
            The name of the experiment.
        prov_save_path : Optional[str], optional
            The path where the provenance data will be saved. If not provided, uses default.
        user_namespace : Optional[str], optional
            The user namespace for organizing experiments. If not provided, uses default.
        collect_all_processes : bool, optional
            A flag indicating whether to collect provenance data for all processes in a distributed setting.
        save_after_n_logs : int, optional
            The number of logs after which metrics are saved. Default is 100.
        rank : Optional[int], optional
            The rank of the current process in a distributed setting. If not provided, determines the global rank.
        metrics_file_type : MetricsType
            The file type used to store metrics. Default is MetricsType.ZARR.

        Returns:
        --------
        None
        """
        self.global_rank = funcs.get_global_rank() if rank is None else rank
        self.EXPERIMENT_NAME = experiment_name + f"_GR{self.global_rank}" if self.global_rank else experiment_name
        self.is_collecting = self.global_rank is None or int(self.global_rank) == 0 or collect_all_processes
        
        if not self.is_collecting: return

        self.save_metrics_after_n_logs = save_after_n_logs

        if prov_save_path: 
            self.PROV_SAVE_PATH = prov_save_path
        if user_namespace:
            self.USER_NAMESPACE = user_namespace

        # look at PROV dir how many experiments are there with the same name
        if not os.path.exists(self.PROV_SAVE_PATH):
            os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
        prev_exps = os.listdir(self.PROV_SAVE_PATH) if self.PROV_SAVE_PATH else []
        run_id = len([exp for exp in prev_exps if funcs.prov4ml_experiment_matches(experiment_name, exp)]) 

        self.EXPERIMENT_DIR = os.path.join(self.PROV_SAVE_PATH, experiment_name + f"_{run_id}")
        self.RUN_ID = run_id
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, "artifacts")
        self.METRICS_DIR = os.path.join(self.EXPERIMENT_DIR, "metrics")
        self.METRICS_FILE_TYPE = metrics_file_type

    def add_metric(
        self, 
        metric: str, 
        value: Any, 
        step: int, 
        context: Optional[Any] = None, 
        source: LoggingItemKind = None, 
        timestamp = None
    ) -> None:
        """
        Adds a metric to the provenance data.

        Parameters:
        -----------
        metric : str
            The name of the metric to add.
        value : Any
            The value of the metric to add.
        step : int
            The step or iteration number associated with the metric value.
        context : Optional[Any], optional
            The context in which the metric is recorded, default is None.
        source : LoggingItemKind, optional
            The source of the logging item, default is None.
        timestamp : optional
            The timestamp when the metric is recorded. If not provided, the current time in milliseconds is used.

        Returns:
        --------
        None
        """        
        if not self.is_collecting: return

        if (metric, context) not in self.metrics:
            self.metrics[(metric, context)] = MetricInfo(metric, context, source=source)
        
        self.metrics[(metric, context)].add_metric(value, step, timestamp if timestamp else funcs.get_current_time_millis())

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

    def add_parameter(self, parameter: str, value: Any) -> None:
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

        self.parameters[parameter] = ParameterInfo(parameter, value)

    def add_artifact(
        self, 
        artifact_name: str, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[Any] = None, 
        timestamp: Optional[int] = None
    ) -> None:
        """
        Adds an artifact to the artifacts dictionary.

        Parameters:
            artifact_name (str): The name of the artifact.
            value (Any): The value of the artifact. Defaults to None.
            step (Optional[int]): The step number for the artifact. Defaults to None.
            context (Optional[Any]): The context of the artifact. Defaults to None.
            timestamp (Optional[int]): The timestamp of the artifact. Defaults to None.
        """
        if not self.is_collecting: return

        self.artifacts[(artifact_name, context)] = ArtifactInfo(artifact_name, value, step, context=context, timestamp=timestamp)

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

        if not os.path.exists(self.METRICS_DIR):
            os.makedirs(self.METRICS_DIR, exist_ok=True)

        metric.save_to_file(self.METRICS_DIR, file_type=self.METRICS_FILE_TYPE, process=self.global_rank)

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
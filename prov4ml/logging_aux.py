import os
import torch
import warnings
import zarr
import netCDF4 as nc

from torch.utils.data import DataLoader, Subset, Dataset
from typing import Any, Optional, Union

from prov4ml.datamodel.attribute_type import LoggingItemKind
from prov4ml.utils import energy_utils, flops_utils, system_utils, time_utils, funcs
from prov4ml.provenance.context import Context
from prov4ml.datamodel.cumulative_metrics import FoldOperation
from prov4ml.provenance.metrics_type import MetricsType
from prov4ml.constants import PROV4ML_DATA
    
def log_metric(key: str, value: float, context:Context, step: Optional[int] = None, source: LoggingItemKind = None) -> None:
    """
    Logs a metric with the specified key, value, and context.

    Args:
        key (str): The key of the metric.
        value (float): The value of the metric.
        context (Context): The context in which the metric is recorded.
        step (Optional[int], optional): The step number for the metric. Defaults to None.
        source (LoggingItemKind, optional): The source of the logging item. Defaults to None.

    Returns:
        None
    """
    PROV4ML_DATA.add_metric(key,value,step, context=context, source=source)

def log_execution_start_time() -> None:
    """Logs the start time of the current execution. """
    return log_param("execution_start_time", time_utils.get_time())

def log_execution_end_time() -> None:
    """Logs the end time of the current execution."""
    return log_param("execution_end_time", time_utils.get_time())

def log_current_execution_time(label: str, context: Context, step: Optional[int] = None) -> None:
    """Logs the current execution time under the given label.
    
    Args:
        label (str): The label to associate with the logged execution time.
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged execution time. Defaults to None.

    Returns:
        None
    """
    return log_metric(label, time_utils.get_time(), context, step=step, source=LoggingItemKind.EXECUTION_TIME)

def log_param(key: str, value: Any) -> None:
    """Logs a single parameter key-value pair. 
    
    Args:
        key (str): The key of the parameter.
        value (Any): The value of the parameter.

    Returns:
        None
    """
    PROV4ML_DATA.add_parameter(key,value)

def log_model_memory_footprint(model: Union[torch.nn.Module, Any], model_name: str = "default") -> None:
    """Logs the memory footprint of the provided model.
    
    Args:
        model (Union[torch.nn.Module, Any]): The model whose memory footprint is to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".

    Returns:
        None
    """
    log_param("model_name", model_name)

    total_params = sum(p.numel() for p in model.parameters())
    try: 
        if hasattr(model, "trainer"): 
            precision_to_bits = {"64": 64, "32": 32, "16": 16, "bf16": 16}
            if hasattr(model.trainer, "precision"):
                precision = precision_to_bits.get(model.trainer.precision, 32)
            else: 
                precision = 32
        else: 
            precision = 32
    except RuntimeError: 
        warnings.warn("Could not determine precision, defaulting to 32 bits. Please make sure to provide a model with a trainer attached, this is often due to calling this before the trainer.fit() method")
        precision = 32
    
    precision_megabytes = precision / 8 / 1e6

    memory_per_model = total_params * precision_megabytes
    memory_per_grad = total_params * 4 * 1e-6
    memory_per_optim = total_params * 4 * 1e-6
    
    log_param("total_params", total_params)
    log_param("memory_of_model", memory_per_model)
    log_param("total_memory_load_of_model", memory_per_model + memory_per_grad + memory_per_optim)

def log_model(model: Union[torch.nn.Module, Any], model_name: str = "default", log_model_info: bool = True, log_as_artifact=True) -> None:
    """Logs the provided model as artifact and logs memory footprint of the model. 
    
    Args:
        model (Union[torch.nn.Module, Any]): The model to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".
        log_model_info (bool, optional): Whether to log model memory footprint. Defaults to True.
        log_as_artifact (bool, optional): Whether to log the model as an artifact. Defaults to True.
    """
    if log_model_info:
        log_model_memory_footprint(model, model_name)

    if log_as_artifact:
        save_model_version(model, model_name, Context.EVALUATION)
        
def log_flops_per_epoch(label: str, model: Any, dataset: Any, context: Context, step: Optional[int] = None) -> None:
    """Logs the number of FLOPs (floating point operations) per epoch for the given model and dataset.
    
    Args:
        label (str): The label to associate with the logged FLOPs per epoch.
        model (Any): The model for which FLOPs per epoch are to be logged.
        dataset (Any): The dataset used for training the model.
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged FLOPs per epoch. Defaults to None.

    Returns:
        None
    """
    return log_metric(label, flops_utils.get_flops_per_epoch(model, dataset), context, step=step, source=LoggingItemKind.FLOPS_PER_EPOCH)

def log_flops_per_batch(label: str, model: Any, batch: Any, context: Context, step: Optional[int] = None) -> None:
    """Logs the number of FLOPs (floating point operations) per batch for the given model and batch of data.
    
    Args:
        label (str): The label to associate with the logged FLOPs per batch.
        model (Any): The model for which FLOPs per batch are to be logged.
        batch (Any): A batch of data used for inference with the model.
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged FLOPs per batch. Defaults to None.

    Returns:
        None
    """
    return log_metric(label, flops_utils.get_flops_per_batch(model, batch), context, step=step, source=LoggingItemKind.FLOPS_PER_BATCH)

def log_system_metrics(
    context: Context,
    step: Optional[int] = None,
    ) -> None:
    """Logs system metrics such as CPU usage, memory usage, disk usage, and GPU metrics.

    Args:
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged metrics. Defaults to None.

    Returns:
        None
    """
    log_metric("cpu_usage", system_utils.get_cpu_usage(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)
    log_metric("memory_usage", system_utils.get_memory_usage(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)
    log_metric("disk_usage", system_utils.get_disk_usage(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)
    log_metric("gpu_memory_usage", system_utils.get_gpu_memory_usage(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)
    log_metric("gpu_usage", system_utils.get_gpu_usage(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)
    log_metric("gpu_temperature", system_utils.get_gpu_temperature(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)
    log_metric("gpu_power_usage", system_utils.get_gpu_power_usage(), context, step=step, source=LoggingItemKind.SYSTEM_METRIC)

def log_carbon_metrics(
    context: Context,
    step: Optional[int] = None,
    ):
    """Logs carbon emissions metrics such as energy consumed, emissions rate, and power consumption.
    
    Args:
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged metrics. Defaults to None.
    
    Returns:
        None
    """    
    emissions = energy_utils.stop_carbon_tracked_block()
   
    log_metric("emissions", emissions.energy_consumed, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("emissions_rate", emissions.emissions_rate, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("cpu_power", emissions.cpu_power, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("gpu_power", emissions.gpu_power, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("ram_power", emissions.ram_power, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("cpu_energy", emissions.cpu_energy, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("gpu_energy", emissions.gpu_energy, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("ram_energy", emissions.ram_energy, context, step=step, source=LoggingItemKind.CARBON_METRIC)
    log_metric("energy_consumed", emissions.energy_consumed, context, step=step, source=LoggingItemKind.CARBON_METRIC)

def log_artifact(
        artifact_path : str, 
        context: Context,
        step: Optional[int] = None, 
        timestamp: Optional[int] = None
    ) -> None:
    """
    Logs the specified artifact to the given context.

    Parameters:
        artifact_path (str): The file path of the artifact to log.
        context (Context): The context in which the artifact is logged.
        step (Optional[int]): The step or epoch number associated with the artifact. Defaults to None.
        timestamp (Optional[int]): The timestamp associated with the artifact. Defaults to None.

    Returns:
        None
    """
    timestamp = timestamp or funcs.get_current_time_millis()
    PROV4ML_DATA.add_artifact(artifact_path, step=step, context=context, timestamp=timestamp)

def save_model_version(
        model: Union[torch.nn.Module, Any], 
        model_name: str, 
        context: Context, 
        step: Optional[int] = None, 
        timestamp: Optional[int] = None
    ) -> None:
    """
    Saves the state dictionary of the provided model and logs it as an artifact.
    
    Parameters:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name under which to save the model.
        context (Context): The context in which the model is saved.
        step (Optional[int]): The step or epoch number associated with the saved model. Defaults to None.
        timestamp (Optional[int]): The timestamp associated with the saved model. Defaults to None.

    Returns:
        None
    """

    path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, model_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # count all models with the same name stored at "path"
    num_files = len([file for file in os.listdir(path) if str(file).startswith(model_name)])

    torch.save(model.state_dict(), f"{path}/{model_name}_{num_files}.pth")
    log_artifact(f"{path}/{model_name}_{num_files}.pth", context=context, step=step, timestamp=timestamp)

def log_dataset(dataset : Union[DataLoader, Subset, Dataset], label : str): 
    """
    Logs dataset statistics such as total samples and total steps.

    Args:
        dataset (Union[DataLoader, Subset, Dataset]): The dataset for which statistics are to be logged.
        label (str): The label to associate with the logged dataset statistics.

    Returns:
        None
    """
    # handle datasets from DataLoader
    if isinstance(dataset, DataLoader):
        dl = dataset
        dataset = dl.dataset

        log_param(f"{label}_dataset_stat_batch_size", dl.batch_size)
        log_param(f"{label}_dataset_stat_num_workers", dl.num_workers)
        # log_param(f"{label}_dataset_stat_shuffle", dl.shuffle)
        log_param(f"{label}_dataset_stat_total_steps", len(dl))

    elif isinstance(dataset, Subset):
        dl = dataset
        dataset = dl.dataset
        log_param(f"{label}_dataset_stat_total_steps", len(dl))

    total_samples = len(dataset)
    log_param(f"{label}_dataset_stat_total_samples", total_samples)

def register_final_metric(
        metric_name : str,
        initial_value : float,
        fold_operation : FoldOperation
    ) -> None:
    """
    Registers a final metric to be computed at the end of the experiment.

    Args:
        metric_name (str): The name of the metric.
        initial_value (float): The initial value of the metric.
        fold_operation (FoldOperation): The operation to be performed on the metric.

    Returns:
        None
    """
    PROV4ML_DATA.add_cumulative_metric(metric_name, initial_value, fold_operation)

def get_context_from_metric_file(metric_file : str,) -> Context:
    """
    Retrieves the context from a metric file.

    Args:
        metric_file (str): The path to the metric file.

    Returns:
        Context: The context associated with the metric file.
    """
    if PROV4ML_DATA.METRICS_FILE_TYPE == MetricsType.TXT:
        with open(metric_file, 'r') as f:
            line = f.readline()
        return eval(line.split(',')[1])
    elif PROV4ML_DATA.METRICS_FILE_TYPE == MetricsType.ZARR:
        dataset = zarr.open(metric_file, mode='r')
        return eval(dataset.attrs['context'])
    elif PROV4ML_DATA.METRICS_FILE_TYPE == MetricsType.NETCDF:
        dataset = nc.Dataset(metric_file, mode='r')
        return eval(dataset._context)
import os
import torch
import json
import warnings
import subprocess
from pathlib import Path
import prov.model as prov

from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler
from typing import Any, Optional, Union

from yprov4ml.datamodel.attribute_type import LoggingItemKind
from yprov4ml.utils import energy_utils, flops_utils, system_utils, time_utils, funcs
from yprov4ml.datamodel.context import Context
from yprov4ml.constants import PROV4ML_DATA, VERBOSE
    
def log_metric(
        key: str, 
        value: float, 
        context: Optional[Context] = None, 
        step: Optional[int] = 0, 
        source: Optional[LoggingItemKind] = None, 
    ) -> None:
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
    PROV4ML_DATA.add_metric(key, value, step, context=context, source=source)

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

def log_param(key: str, value: Any, context : Context = None) -> None:
    """Logs a single parameter key-value pair. 
    
    Args:
        key (str): The key of the parameter.
        value (Any): The value of the parameter.

    Returns:
        None
    """
    PROV4ML_DATA.add_parameter(key,value, context)

def _get_model_memory_footprint(model_name: str, model: Union[torch.nn.Module, Any]) -> dict:
    """Logs the memory footprint of the provided model.
    
    Args:
        model (Union[torch.nn.Module, Any]): The model whose memory footprint is to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".

    Returns:
        None
    """
    ret = {"model_name": model_name}


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
        if VERBOSE: 
            warnings.warn("Could not determine precision, defaulting to 32 bits. Please make sure to provide a model with a trainer attached, this is often due to calling this before the trainer.fit() method")
        precision = 32
    
    precision_megabytes = precision / 8 / 1e6

    memory_per_model = total_params * precision_megabytes
    memory_per_grad = total_params * 4 * 1e-6
    memory_per_optim = total_params * 4 * 1e-6
    
    ret["total_params"] = total_params
    ret["memory_of_model"] = memory_per_model
    ret["total_memory_load_of_model"] = memory_per_model + memory_per_grad + memory_per_optim

    return ret

def _get_nested_model_desc(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        node = {}
        funcs._safe_get_model_attr(node, m, "type", attr_label="layer_type")
        funcs._safe_get_model_attr(node, m, "in_features")
        funcs._safe_get_model_attr(node, m, "out_features")
        funcs._safe_get_model_attr(node, m, "in_channels")
        funcs._safe_get_model_attr(node, m, "out_channels")
        funcs._safe_get_model_attr(node, m, "kernel_size")
        funcs._safe_get_model_attr(node, m, "stride")
        funcs._safe_get_model_attr(node, m, "padding")
        funcs._safe_get_model_attr(node, m, "weight.dtype", attr_label="dtype")
        # safe_get_attr(node, m, "bias", attr_label="layer_bias")
        return node
    else:
        for name, child in children.items():
            output[name] = _get_nested_model_desc(child)

    return output

def _get_model_layers_description(model_name : str, model: Union[torch.nn.Module, Any]) -> prov.ProvEntity: 
    mo = _get_nested_model_desc(model)
    
    path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, model_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(f"{path}/{model_name}_layers_description.json", "w") as fp:
        json.dump(mo , fp) 

    return {
        "layers_description_path": f"{path}/{model_name}_layers_description.json"
    }

def log_model(
        model_name: str, 
        model: Union[torch.nn.Module, Any], 
        log_model_info: bool = True, 
        log_model_layers : bool = False,
        is_input: bool = False,
    ) -> None:
    """Logs the provided model as artifact and logs memory footprint of the model. 
    
    Args:
        model (Union[torch.nn.Module, Any]): The model to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".
        log_model_info (bool, optional): Whether to log model memory footprint. Defaults to True.
        log_model_layers (bool, optional): Whether to log model layers details. Defaults to False.
        log_as_artifact (bool, optional): Whether to log the model as an artifact. Defaults to True.
    """
    e = save_model_version(model_name, model, Context.MODELS, incremental=False, is_input=is_input)

    if log_model_info:
        d = _get_model_memory_footprint(model_name, model)
        e.add_attributes(d)

    if log_model_layers: 
        d = _get_model_layers_description(model_name, model)
        e.add_attributes(d)

     
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
        artifact_name : str, 
        artifact_path : str, 
        context: Optional[Context] = None,
        step: Optional[int] = None, 
        log_copy_in_prov_directory : bool = True, 
        is_model : bool = False, 
        is_input : bool = False, 
    ) -> prov.ProvEntity:
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
    return PROV4ML_DATA.add_artifact(
        artifact_name=artifact_name, 
        artifact_path=artifact_path, 
        step=step, 
        context=context, 
        log_copy_in_prov_directory=log_copy_in_prov_directory, 
        is_model=is_model, 
        is_input=is_input, 
    )

def save_model_version(
        model_name: str, 
        model: Union[torch.nn.Module, Any], 
        context: Optional[Context] = None, 
        step: Optional[int] = None, 
        incremental : bool = True, 
        is_input : bool =False, 
    ) -> prov.ProvEntity:
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
    if incremental: 
        num_files = len([file for file in os.listdir(path) if str(file).startswith(model_name)])
        torch.save(model.state_dict(), f"{path}/{model_name}_{num_files}.pth")
        return log_artifact(f"{model_name}_{num_files}", f"{path}/{model_name}_{num_files}.pth", context=context, step=step, log_copy_in_prov_directory=False, is_model=True, is_input=is_input)
    else: 
        torch.save(model.state_dict(), f"{path}/{model_name}.pth")
        return log_artifact(model_name, f"{path}/{model_name}.pth", context=context, step=step, log_copy_in_prov_directory=False, is_model=True, is_input=is_input)

def log_dataset(
        dataset_label : str, 
        dataset : Union[DataLoader, Subset, Dataset], 
        log_dataset_info : bool = True, 
        ): 
    """
    Logs dataset statistics such as total samples and total steps.

    Args:
        dataset (Union[DataLoader, Subset, Dataset]): The dataset for which statistics are to be logged.
        label (str): The label to associate with the logged dataset statistics.

    Returns:
        None
    """

    e = log_artifact(f"{dataset_label}", "", context=Context.DATASETS, log_copy_in_prov_directory=False, is_model=False, is_input=True)
    
    if not log_dataset_info: return

    e.add_attributes({f"{dataset_label}_stat_total_samples": len(dataset)})
    # handle datasets from DataLoader
    if isinstance(dataset, DataLoader):
        dl = dataset
        dataset = dl.dataset
        attrs = {
            f"{dataset_label}_stat_batch_size": dl.batch_size, 
            f"{dataset_label}_stat_num_workers": dl.num_workers, 
            f"{dataset_label}_stat_shuffle": isinstance(dl.sampler, RandomSampler), 
            f"{dataset_label}_stat_total_steps": len(dl), 
        }
        e.add_attributes(attrs)

    elif isinstance(dataset, Subset):
        dl = dataset
        dataset = dl.dataset
        e.add_attributes({f"{dataset_label}_stat_total_steps": len(dl)})

def log_execution_command(cmd: str, path : str) -> None:
    """
    Logs the execution command.
    
    Args:
        cmd (str): The command to be logged.
    """
    path = os.path.join("/workspace", f"{PROV4ML_DATA.CLEAN_EXPERIMENT_NAME}_{PROV4ML_DATA.RUN_ID}", "artifacts", path)
    log_param("prov-ml:execution_command", cmd + " " + path)

def log_source_code(path: Optional[str] = None) -> None:
    """
    Logs the source code location, either from a Git repository or a specified path.
    
    Args: 
        path (Optional[str]): The path to the source code. If None, attempts to retrieve from Git.
    """
    PROV4ML_DATA.add_source_code(path)

def create_context(context : str, is_subcontext_of=None): 
    PROV4ML_DATA.add_context(context, is_subcontext_of=is_subcontext_of)
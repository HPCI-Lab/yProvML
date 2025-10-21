import os
import warnings
import prov.model as prov
import tensorflow as tf
import keras
from typing import Any, Optional
import hashlib
import numpy as np

from yprov4ml.datamodel.attribute_type import LoggingItemKind
from yprov4ml.utils import energy_utils, system_utils, time_utils
from yprov4ml.datamodel.context import Context
from yprov4ml.constants import PROV4ML_DATA
    
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

def _get_model_memory_footprint(model_name: str, model: Any) -> dict:
    """Logs the memory footprint of the provided model.
    
    Args:
        model (Union[torch.nn.Module, Any]): The model whose memory footprint is to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".

    Returns:
        None
    """
    # log_param("model_name", model_name)   
    ret = {"model_name": str(model_name)}

    def is_keras_or_tf_model(obj):
        model_classes = (tf.keras.Model, keras.Model)#tf.estimator.Estimator
        return isinstance(obj, model_classes)

    if is_keras_or_tf_model(model): 
        trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
        non_trainable_params = sum([tf.size(v).numpy() for v in model.non_trainable_variables])
        # log_param("trainable_params", trainable_params)
        # log_param("non_trainable_params", non_trainable_params)
        ret["trainable_params"] = str(trainable_params)
        ret["non_trainable_params"] = str(non_trainable_params)

        total_params = trainable_params + non_trainable_params
    elif hasattr(model, "parameters"): 
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else: 
        total_params = -1
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
    
    # log_param("total_params", total_params)
    # log_param("memory_of_model", memory_per_model)
    # log_param("total_memory_load_of_model", memory_per_model + memory_per_grad + memory_per_optim)
    ret["total_params"] = str(total_params)
    ret["memory_of_model"] = str(memory_per_model)
    ret["total_memory_load_of_model"] = str(memory_per_model + memory_per_grad + memory_per_optim)
    return ret

def log_model(
        model_name: str, 
        model: Any, 
        log_model_info: bool = True, 
        # log_model_layers : bool = False,
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
        model: Any, 
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
        # torch.save(model.state_dict(), f"{path}/{model_name}_{num_files}.pth")
        model.save_weights(f"{path}/{model_name}_{num_files}.weights.h5")
        return log_artifact(f"{model_name}_{num_files}", f"{path}/{model_name}_{num_files}.pth", context=context, step=step, log_copy_in_prov_directory=False, is_model=True, is_input=is_input)
    else: 
        # torch.save(model.state_dict(), f"{path}/{model_name}.pth")
        model.save_weights(f"{path}/{model_name}.weights.h5")
        return log_artifact(model_name, f"{path}/{model_name}.pth", context=context, step=step, log_copy_in_prov_directory=False, is_model=True, is_input=is_input)

def log_dataset(
        dataset_label: str, 
        dataset: Any, 
        log_dataset_info: bool = True, 
        context : Optional[Context] = Context.DATASETS
        ):
    e = log_artifact(f"{dataset_label}", "", context=context, log_copy_in_prov_directory=False, is_model=False, is_input=True)
    if not log_dataset_info:
        return

    d = {}

    if isinstance(dataset, tf.keras.utils.Sequence):
        batch_size = dataset.batch_size
        total_steps = len(dataset)
        num_samples = total_steps * batch_size
    else:
        # fallback per tf.data.Dataset
        try:
            batch_size = dataset.element_spec[0].shape[0]
        except Exception:
            batch_size = "Unknown"

        try:
            total_steps = dataset.cardinality().numpy()
            if total_steps == tf.data.experimental.INFINITE_CARDINALITY:
                total_steps = "Infinite"
                num_samples = "Infinite"
            else:
                num_samples = total_steps * (batch_size if batch_size != "Unknown" else 1)
        except Exception:
            total_steps = "Unknown"
            num_samples = "Unknown"

    # Log parameters
    d[f"{dataset_label}_dataset_stat_batch_size"] = str(batch_size)
    d[f"{dataset_label}_dataset_stat_total_steps"] = str(total_steps)
    d[f"{dataset_label}_dataset_stat_total_samples"] = str(num_samples)
    e.add_attributes(d)

def log_proof_of_learning_step(model_label, model, loss, batch, step, context=None): 
    def hash_tensor(tensor):
        arr = tf.reshape(tensor, [-1]).numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()
        
    step_proof = {
        'step': batch,
        'loss': float(loss),
        'weights_hash': hash_tensor(model.trainable_variables[0])
    }
    log_metric(f"{model_label}_pol", step_proof, context=context, step=step)

    path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, f"{model_label}_pol_checkpoints", f"{model_label}_pol_checkpoint_{step}_{batch}.npy")
    os.makedirs(os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, f"{model_label}_pol_checkpoints"), exist_ok=True)
    np.save(path, tf.reshape(model.trainable_variables[0], [-1]).numpy())
    log_artifact(f"{model_label}_pol_checkpoint_{step}_{batch}", path, context=context, step=step, log_copy_in_prov_directory=False, is_model=True, is_input=False)


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

import os
from typing import Any, Dict, Optional, Union
from lightning.pytorch.loggers.logger import Logger
from typing_extensions import override
from argparse import Namespace
from torch import Tensor

from prov4ml.logging_aux import log_param, log_metric
from prov4ml.datamodel.context import Contexts

class ProvMLLogger(Logger):
    def __init__(
        self,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        """
        Initializes a ProvMLLogger instance.

        Parameters:
            name (Optional[str]): The name of the experiment. Defaults to "lightning_logs".
            version (Optional[Union[int, str]]): The version of the experiment. Defaults to None.
            prefix (str): The prefix for the experiment. Defaults to an empty string.
            flush_logs_every_n_steps (int): The number of steps after which logs should be flushed. Defaults to 100.
        """
        super().__init__()
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._experiment = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    @override
    def root_dir(self) -> str:
        """
        Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version".

        Returns:
            str: The root directory path.
        """
        return os.path.join(self.save_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        """
        The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        Returns:
            str: The log directory path.
        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, version)

    @property
    @override
    def name(self) -> str:
        """
        The name of the experiment.

        Returns:
            str: The name of the experiment.
        """
        return self._name
    
    @property
    @override
    def version(self) -> Optional[Union[int, str]]:
        """
        The version of the experiment.

        Returns:
            Optional[Union[int, str]]: The version of the experiment.
        """
        return self._version
    
    @override
    def log_metrics(
        self, 
        metrics: Dict[str, Union[Tensor, float]], 
        step: Optional[int] = None, 
        context : Optional[Contexts] = None
        ) -> None:
        """
        Logs the provided metrics to the MLflow tracking context.

        Parameters:
            metrics (Dict[str, Union[Tensor, float]]): A dictionary containing the metrics and their associated values.
            step (Optional[int]): The step number for the metrics. Defaults to None.
        """
        print(metrics)
        print(step)
        print(context)
        log_metric(list(metrics.keys())[0], metrics[list(metrics.keys())[0]], context=Contexts.TRAINING, step=metrics["epoch"])
    
    @override
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """
        Logs the provided hyperparameters to the MLflow tracking context.

        Parameters:
            params (Union[Dict[str, Any], Namespace]): The hyperparameters to be logged.
        """
        for key, value in params.items():
            log_param(key, value)

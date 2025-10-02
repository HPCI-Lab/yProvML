import os
from typing import Any, Dict, Optional, Union
from lightning.pytorch.loggers.logger import Logger
from typing_extensions import override
from argparse import Namespace
from torch import Tensor

from yprov4ml.logging_aux import log_param, log_metric
from yprov4ml.datamodel.context import Context

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
        return os.path.join(self.save_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, version)

    @property 
    @override
    def name(self) -> str:
        return self._name
    
    @property 
    @override
    def version(self) -> Optional[Union[int, str]]:
        return self._version
    
    @override
    def log_metrics(self, metrics: Dict[str, Union[Tensor, float]], step) -> None:

        for m, v in metrics.items(): 
            context = Context.TRAINING
            if "evaluation" in m or "test" in m: 
                context = Context.TESTING
            elif "validation" in m: 
                context = Context.VALIDATION
            log_metric(m, v, context=context)
    
    @override
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        for key, value in params.items():
            log_param(key, value)

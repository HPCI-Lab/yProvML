
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from typing import List, Tuple, Literal

from yprov4ml.logging_aux import *
from yprov4ml.loggers.itwinai_logger import Logger
from yprov4ml.yprov4ml import *

class ProvMLItwinAILogger(Logger):
    """
    Abstraction around Prov4ML logger.

    Args:
        prov_user_namespace (str, optional): location to where provenance
            files will be uploaded. Defaults to "www.example.org".
        experiment_name (str, optional): experiment name.
            Defaults to "experiment_name".
        provenance_save_dir (str, optional): path where to store provenance
            files and logs. Defaults to "prov".
        save_after_n_logs (Optional[int], optional): how often to save
            logs to disk from main memory. Defaults to 100.
        create_graph (Optional[bool], optional): whether to create a
            provenance graph. Defaults to True.
        create_svg (Optional[bool], optional): whether to create an SVG
            representation of the provenance graph. Defaults to True.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            determines whether the logger should fulfill or ignore
            calls to the `log()` method. See ``Logger.should_log`` method for
            more details. Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """

    #: Supported kinds in the ``log`` method
    supported_kinds: Tuple[str] = (
        'metric', 'flops_pb', 'flops_pe', 'system', 'carbon',
        'execution_time', 'model', 'best_model',
        'torch')

    def __init__(
        self,
        prov_user_namespace="www.example.org",
        experiment_name="experiment_name",
        provenance_save_dir="mllogs",
        save_after_n_logs: Optional[int] = 100,
        create_graph: Optional[bool] = True,
        create_svg: Optional[bool] = True,
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        super().__init__(
            savedir=provenance_save_dir,
            log_freq=log_freq,
            log_on_workers=log_on_workers
        )
        self.name = experiment_name
        self.version = None
        self.prov_user_namespace = prov_user_namespace
        self.provenance_save_dir = provenance_save_dir
        self.save_after_n_logs = save_after_n_logs
        self.create_graph = create_graph
        self.create_svg = create_svg

    @override
    def create_logger_context(self, rank: Optional[int] = None):
        """
        Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """
        self.worker_rank = rank

        if not self.should_log():
            return

        start_run(
            prov_user_namespace=self.prov_user_namespace,
            experiment_name=self.name,
            provenance_save_dir=self.provenance_save_dir,
            save_after_n_logs=self.save_after_n_logs,
            # This class will control which workers can log
            collect_all_processes=True,
            rank=rank
        )

    @override
    def destroy_logger_context(self):
        """
        Destroys the logger context.
        """
        if not self.should_log():
            return

        end_run(
            create_graph=self.create_graph,
            create_svg=self.create_svg
        )

    @override
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        if not self.should_log():
            return

        # Save hyperparams
        for param_name, val in params.items():
            log_param(param_name, val)

    @override
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        context: Optional[str] = 'training',
        **kwargs
    ) -> None:
        """Logs with Prov4ML.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be
                one among the list of ``self.supported_kinds``.
                Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
            kwargs: keyword arguments to pass to the logger.
        """

        if not self.should_log(batch_idx=batch_idx):
            return

        if kind == "metric":
            log_metric(key=identifier, value=item,
                               context=context, step=step)
        elif kind == "flops_pb":
            model, batch = item
            log_flops_per_batch(
                identifier, model=model,
                batch=batch, context=context, step=step)
        elif kind == "flops_pe":
            model, dataset = item
            log_flops_per_epoch(
                identifier, model=model,
                dataset=dataset, context=context, step=step)
        elif kind == "system":
            log_system_metrics(context=context, step=step)
        elif kind == "carbon":
            log_carbon_metrics(context=context, step=step)
        elif kind == "execution_time":
            log_current_execution_time(
                label=identifier, context=context, step=step)
        elif kind == 'model':
            save_model_version(
                model=item, model_name=identifier, context=context, step=step)
        elif kind == 'best_model':
            log_model(model=item, model_name=identifier, log_model_info=True, log_model_layers = False, is_input = False)
        elif kind == 'torch':
            from torch.utils.data import DataLoader
            if isinstance(item, DataLoader):
                log_dataset(dataset=item, dataset_label=identifier)
            else:
                log_param(key=identifier, value=item)


import os
import pickle
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Tuple
from typing_extensions import Literal

class LogMixin(metaclass=ABCMeta):
    @abstractmethod
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log ``item`` with ``identifier`` name of ``kind`` type at ``step``
        time step.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_kinds. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """


class Logger(LogMixin, metaclass=ABCMeta):
    """Base class for logger

    Args:
        savedir (str, optional): filesystem location where logs are stored.
            Defaults to 'mllogs'.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            how often should the logger fulfill calls to the `log()`
            method:

            - When set to 'epoch', the logger logs only if ``batch_idx``
              is not passed to the ``log`` method.

            - When an integer
              is given, the logger logs if ``batch_idx`` is a multiple of
              ``log_freq``.

            - When set to ``'batch'``, the logger logs always.

            Defaults to 'epoch'.
        log_on_workers (Optional[Union[int, List[int]]]): if -1, log on all
            workers; if int log on worker with rank equal to log_on_workers;
            if List[int], log on workers which rank is in the list.
            Defaults to 0 (the global rank of the main worker).
    """
    #: Location on filesystem where to store data.
    savedir: str = None
    #: Supported logging 'kind's.
    supported_kinds: Tuple[str]
    #: Current worker global rank
    worker_rank: int

    _log_freq: Union[int, Literal['epoch', 'batch']]

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch',
        log_on_workers: Union[int, List[int]] = 0
    ) -> None:
        self.savedir = savedir
        self.log_freq = log_freq
        self.log_on_workers = log_on_workers

    @property
    def log_freq(self) -> Union[int, Literal['epoch', 'batch']]:
        """Get ``log_feq``, namely how often should the logger
        fulfill or ignore calls to the `log()` method."""
        return self._log_freq

    @log_freq.setter
    def log_freq(self, val: Union[int, Literal['epoch', 'batch']]):
        """Sanitize log_freq value."""
        if val in ['epoch', 'batch'] or (isinstance(val, int) and val > 0):
            self._log_freq = val
        else:
            raise ValueError(
                "Wrong value for 'log_freq'. Supported values are: "
                f"['epoch', 'batch'] or int > 0. Received: {val}"
            )

    @contextmanager
    def start_logging(self, rank: Optional[int] = None):
        """Start logging context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.

        Example:


        >>> with my_logger.start_logging():
        >>>     my_logger.log(123, 'value', kind='metric', step=0)


        """
        try:
            self.create_logger_context(rank=rank)
            yield
        finally:
            self.destroy_logger_context()

    @abstractmethod
    def create_logger_context(self, rank: Optional[int] = None) -> Any:
        """
        Initializes the logger context.

        Args:
            rank (Optional[int]): global rank of current process,
                used in distributed environments. Defaults to None.
        """

    @abstractmethod
    def destroy_logger_context(self) -> None:
        """Destroy logger."""

    @abstractmethod
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """

    def serialize(self, obj: Any, identifier: str) -> str:
        """Serializes object to disk and returns its path.

        Args:
            obj (Any): item to save.
            identifier (str): identifier of the item to log (expected to be a
                path under ``self.savedir``).

        Returns:
            str: local path of the serialized object to be logged.
        """
        itm_path = os.path.join(self.savedir, identifier)
        with open(itm_path, 'wb') as itm_file:
            pickle.dump(obj, itm_file)

    def should_log(
        self,
        batch_idx: Optional[int] = None
    ) -> bool:
        """Determines whether the logger should fulfill or ignore calls to the
        `log()` method, depending on the ``log_freq`` property:

        - When ``log_freq`` is set to 'epoch', the logger logs only if
          ``batch_idx`` is not passed to the ``log`` method.

        - When ``log_freq`` is an integer
          is given, the logger logs if ``batch_idx`` is a multiple of
          ``log_freq``.

        - When ``log_freq`` is set to ``'batch'``, the logger logs always.

        It also takes into account whether logging on the current worker
        rank is allowed by ``self.log_on_workers``.

        Args:
            batch_idx (Optional[int]): the dataloader batch idx, if available.
                Defaults to None.

        Returns:
            bool: True if the logger should log, False otherwise.
        """
        # Check worker's global rank
        worker_ok = (
            self.worker_rank is None or
            (isinstance(self.log_on_workers, int) and (
                self.log_on_workers == -1 or
                self.log_on_workers == self.worker_rank
            )
            )
            or
            (isinstance(self.log_on_workers, list)
             and self.worker_rank in self.log_on_workers)
        )
        if not worker_ok:
            return False

        # Check batch ID
        if batch_idx is not None:
            if isinstance(self.log_freq, int):
                if batch_idx % self.log_freq == 0:
                    return True
                return False
            if self.log_freq == 'batch':
                return True
            return False
        return True
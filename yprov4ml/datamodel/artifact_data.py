
from typing import Any, Optional

from yprov4ml.utils.funcs import get_current_time_millis

class ArtifactInfo:
    """
    Holds information about an artifact, including its path, value, step, context, and timestamps.

    Attributes:
        path (str): The path of the artifact.
        value (Any): The value of the artifact.
        step (Optional[int]): The step number associated with the artifact.
        context (Optional[Any]): The context of the artifact.
        creation_timestamp (Optional[int]): The creation timestamp of the artifact.
        last_modified_timestamp (Optional[int]): The last modified timestamp of the artifact.
        is_model_version (bool): Indicates if the artifact is a PyTorch model.
    """
    def __init__(
        self, 
        name: str, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[Any] = None, 
        is_model : bool = False, 
    ) -> None:
        self.path = name
        self.value = value
        self.step = step
        self.context = context
        self.creation_timestamp = get_current_time_millis()
        self.last_modified_timestamp = get_current_time_millis()

        self.is_model_version = is_model

    def update(
        self, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[Any] = None
    ) -> None:
        """
        Updates the artifact information with the provided values.

        Parameters:
            value (Any): The new value of the artifact. Defaults to None.
            step (Optional[int]): The new step number for the artifact. Defaults to None.
            context (Optional[Any]): The new context of the artifact. Defaults to None.
        """
        self.value = value if value is not None else self.value
        self.step = step if step is not None else self.step
        self.context = context if context is not None else self.context
        self.last_modified_timestamp = get_current_time_millis()

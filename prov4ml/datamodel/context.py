
from aenum import Enum

class Contexts(Enum): 
    # EXPERIMENT = "EXPERIMENT"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    TESTING = "TESTING"

    @staticmethod
    def get_context_from_string(context: str): 
        """
        Returns the context enum from a string.

        Parameters:
            context (str): The context string.

        Returns:
            Context: The context enum.
        """
        if context == 'training' or context == 'Contexts.TRAINING':
            return Contexts.TRAINING
        elif context == 'evaluation' or context == 'Contexts.TESTING':
            return Contexts.TESTING
        elif context == 'validation' or context == 'Contexts.VALIDATION':
            return Contexts.VALIDATION
        else:
            raise ValueError(f"Invalid context: {context}")
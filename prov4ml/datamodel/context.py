
from aenum import Enum

class Contexts(Enum): 
    # EXPERIMENT = "EXPERIMENT"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    TESTING = "TESTING"
    DATASETS = "DATASETS"
    MODELS = "MODELS"

    @staticmethod
    def get_context_from_string(context: str): 
        """
        Returns the context enum from a string.

        Parameters:
            context (str): The context string.

        Returns:
            Context: The context enum.
        """
        try: 
            context = eval(context)
            if type(context) == Contexts: 
                return context
            else: 
                raise ValueError(f"Invalid context: {context}")
        except: 
            raise ValueError(f"Not a context: {context}")
            
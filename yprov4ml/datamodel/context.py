
from aenum import Enum

class Context(Enum): 
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
            if type(context) == Context: 
                return context
            else: 
                raise ValueError(f"Invalid context: {context}")
        except: 
            if context == 'training' or context == 'Context.TRAINING':
                return Context.TRAINING
            elif context == 'testing' or context == 'Context.TESTING':
                return Context.TESTING
            elif context == 'validation' or context == 'Context.VALIDATION':
                return Context.VALIDATION
            elif context == 'models' or context == 'Context.MODELS':
                return Context.MODELS
            elif context == 'datasets' or context == 'Context.DATASETS':
                return Context.DATASETS
            else:
                raise ValueError(f"Not a context: {context}")
                    
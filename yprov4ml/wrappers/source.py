

import torch

from yprov4ml.logging_aux import log_metric

class ProvenanceTrackedFunction:
    def __init__(self, func, context=None):
        self.fn = func
        self.context = context
        self.source = type(func).__name__
        self.inc = 0

    def __call__(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)

        if torch.is_tensor(result) and result.shape == torch.Size([]): 
            log_metric(self.source, result.item(), context=self.context, step=self.inc, source=self.source)
        else: 
            log_metric(self.source, result, context=self.context, step=self.inc, source=self.source)
        self.inc += 1
        return result

from torch.utils.data import Dataset

class IndexedDatasetWrapper(Dataset):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, index):
        batch = self.dataset[index]
        return index, batch
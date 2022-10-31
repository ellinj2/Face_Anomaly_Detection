
class RegionProposalNetowrk():
    """
    Class to interact with the region proposal network.
    """

    def __init__(self, load_path=None):
        pass

    def parameters(self):
        """
        Returns the model parameters.
        """
        pass

    def to(self, device):
        """
        Loads and performs computations on the model and input data to specified device.
        """
        pass

    def save(self, save_path, save_name):
        """
        Save the model to specified path.
        """
        pass

    def load(self, load_path):
        """
        Load the model from spesified path.
        """
        pass

    def fit(self, epochs, datasets, batch_size, optimizer, save_path, checkpoints=0, progress=False):
        """
        Fits the model to the training dataset and evaluates on the validation dataset
        """
        pass

    def propose(self, X):
        """
        Proposes regions on the input data. 
        """
        pass

    def evaluate(self, datasets, batch_size=1, progress=False):
        """
        Evaluates the model on a dataset.
        """
        pass

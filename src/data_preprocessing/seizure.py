class Seizure:
    """
    Class to store seizure information
    """
    
    def __init__(self, id, start, end):
        """
        Initialize the seizure information.

        Args:
            id (str): ID of the seizure.
            start (datetime): start datetime of the seizure.
            end (datetime): end datetime of the seizure.
        """
        self.id = id
        self.start = start
        self.end = end
        
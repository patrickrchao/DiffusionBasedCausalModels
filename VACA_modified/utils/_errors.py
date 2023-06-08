class IsHeterogeneousError(Exception):
    """
    Exception raised when accesing a variable not
    implemented for heterogeneous SCMs
    """

    def __init__(self, ):
        self.message = 'The SCM is Heterogeneous'
        super().__init__(self.message)

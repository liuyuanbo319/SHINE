def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class nmt():
    def __init__(self):
        super(nmt, self).__init__()
        # data parameters
        self.num_classes = 2
        self.in_channels = 19
        self.ts_len = 1280
        self.z_dim = 96
        self.alpha = 0.1
        self.pooled = False
        self.norm = False
class cgm():
    def __init__(self):
        super(cgm, self).__init__()
        # data parameters
        self.num_classes = 4
        self.in_channels = 1
        self.ts_len = 288
        self.z_dim = 12
        self.alpha = 0.1
        self.pooled = True
        self.norm = True
class tuab():
    def __init__(self):
        super(tuab, self).__init__()
        # data parameters
        self.num_classes = 2
        self.in_channels = 19
        self.ts_len = 1280
        self.z_dim = 96
        self.alpha = 0.1
        self.pooled = False
        self.norm = False

class mit():
    def __init__(self):
        super(mit, self).__init__()
        # data parameters
        self.num_classes = 5
        self.in_channels = 1
        self.ts_len = 300
        self.z_dim = 48
        self.alpha = 0.1
        self.pooled = False
        self.norm = True

class ECG200():
    def __init__(self):
        super(ECG200, self).__init__()
        # data parameters
        self.num_classes = 2
        self.in_channels = 1
        self.ts_len = 96
        self.z_dim = 12
        self.alpha = 0.1
        self.pooled = True
        self.norm = True

class PTBXL():
    def __init__(self):
        super(PTBXL, self).__init__()
        # data parameters
        self.num_classes = 5
        self.in_channels = 12
        self.ts_len = 1000
        self.z_dim = 96
        self.alpha = 0.1
        self.pooled = False
        self.norm=False
        # self.lr = 5e-3
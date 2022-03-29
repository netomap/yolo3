class model_architeture():

    def __init__(self):
        self.architecture_config = [
            [3, 128, 5, 1, 0],
            'leaky:.1',
            [128, 128, 5, 1, 0],
            'leaky:.1',
            'maxpool:2',
            [128, 128, 5, 1, 0],
            'leaky:.1',
            [128, 64, 5, 1, 0],
            [64, 64, 9, 1, 0],
            'leaky:.1',
            'maxpool:2',
            [64, 64, 5, 1, 0],
            'leaky:.1',
            'maxpool:2',
            [64, 64, 7, 1, 0],
            'leaky:.1',
            'maxpool:2',
            [64, 64, 5, 1, 0],
            'leaky:.1',
            [64, 64, 5, 1, 0],
            'leaky:.1',
        ]
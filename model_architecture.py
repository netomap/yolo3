class model_architeture():

    def __init__(self):
        CONVOLUCIONAL = [3, 32, 3, 1, 0]
        LEAKY_RELU = 'leaky:.1'
        MAX_POOL = 'maxpool:2'

        self.architecture_config = [
            CONVOLUCIONAL,
            LEAKY_RELU,
            CONVOLUCIONAL,
            LEAKY_RELU,
            MAX_POOL,
            CONVOLUCIONAL,
            LEAKY_RELU,
            CONVOLUCIONAL,
            LEAKY_RELU,
            MAX_POOL,
            CONVOLUCIONAL,
            LEAKY_RELU,
            MAX_POOL,
            CONVOLUCIONAL,
            LEAKY_RELU,
            MAX_POOL,
            CONVOLUCIONAL,
            LEAKY_RELU,
            CONVOLUCIONAL,
            LEAKY_RELU,
            'flatten:',
            [-1, 1024*2], # LINEAR a entrada é calculada pelo modelo
            'leaky:.1',
            [1024*2, 1024],
            'leaky:.1',
            [1024, -1]    # LINEAR a saída é calculada pelo modelo
        ]
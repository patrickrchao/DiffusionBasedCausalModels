class Cte:
    # equation-types
    LINEAR = "linear"
    NONLINEAR = 'non-linear'
    NONADDITIVE = 'non-additive'

    # Datasets
    SENS = 'sex'  # sensitive attribute for CF fairness

    TRIANGLE = 'triangle'  # a.k.a. Connected fork
    COLLIDER = 'collider'
    LOAN = 'loan'
    ADULT = 'adult'
    MGRAPH = 'mgraph'
    CHAIN = 'chain'
    GERMAN = 'german'

    DATASET_LIST = [COLLIDER,
                    TRIANGLE,
                    LOAN,
                    MGRAPH,
                    CHAIN,
                    ADULT,
                    GERMAN]
    DATASET_LIST_TOY = [COLLIDER,
                        TRIANGLE,
                        LOAN,
                        MGRAPH,
                        CHAIN,
                        ADULT]
    # Models
    VACA = 'vaca'
    VACA_PIWAE = 'vaca_piwae'
    MCVAE = 'mcvae'
    CAREFL = 'carefl'

    # Optimizers
    ADAM = 'adam'
    RADAM = 'radam'
    ADAGRAD = 'adag'
    ADADELTA = 'adad'
    RMS = 'rms'
    ASGD = 'asgd'

    # Scheduler
    STEP_LR = 'step_lr'
    EXP_LR = 'exp_lr'

    # Activation
    TAHN = 'tahn'
    RELU = 'relu'
    RELU6 = 'relu6'
    SOFTPLUS = 'softplus'
    RRELU = 'rrelu'
    LRELU = 'lrelu'
    ELU = 'elu'
    SELU = 'selu'
    SIGMOID = 'sigmoid'
    GLU = 'glu'
    IDENTITY = 'identity'

    # Distribution
    BETA = 'beta'
    CONTINOUS_BERN = 'cb'
    BERNOULLI = 'ber'
    GAUSSIAN = 'normal'
    CATEGORICAL = 'cat'
    EXPONENTIAL = 'exp'
    DELTA = 'delta'

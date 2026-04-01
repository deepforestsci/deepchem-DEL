import os
import random
import numpy as np
import torch
from models.belka import Encodings


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cpu'
    torch.set_default_device(device)
    torch.use_deterministic_algorithms(True)


def test_encodings():

    set_seed(42)
    max_length=5
    depth=3
    enc = Encodings(depth=depth, max_length=max_length)
    out_encodings = enc.encodings

    numpy_array_input = np.array([
        [[-0.05965029,  0.45509598,  0.7437328 ],
        [-0.1701466 ,  0.48044246, -0.70599025],
        [ 3.4038882 ,  1.524032  ,  0.864533  ],
        [-0.35505533, -2.130892  ,  1.0449005 ]],

        [[-1.7214996 ,  1.2570152 , -0.5730889 ],
        [ 0.1413599 , -0.06228027,  0.3747894 ],
        [-1.5961138 ,  1.260025  ,  0.10489947],
        [-0.19719963, -0.7793941 , -0.762407  ]]
    ], dtype=np.float32)

    example_inputs = torch.from_numpy(numpy_array_input)
    assert np.allclose(out_encodings, enc(example_inputs), atol=1e-04)

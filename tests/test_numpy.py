from tensorboardX import utils
import numpy as np
def test_scalar():
    res = utils.makenp(1.1)
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils.makenp(1000000000000000000000)
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils.makenp(np.float16(1.00000087))
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils.makenp(np.float128(1.00008 + 9))
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils.makenp(np.int64(100000000000))
    assert isinstance(res, np.ndarray) and res.shape == (1,)


def test_make_grid():
    pass
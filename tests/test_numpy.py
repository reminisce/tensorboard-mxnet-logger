from tensorboardX import utils
import numpy as np
def test_scalar():
    res = utils._makenp(1.1)
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils._makenp(1000000000000000000000)
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils._makenp(np.float16(1.00000087))
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils._makenp(np.float128(1.00008 + 9))
    assert isinstance(res, np.ndarray) and res.shape == (1,)
    res = utils._makenp(np.int64(100000000000))
    assert isinstance(res, np.ndarray) and res.shape == (1,)


def test_make_grid():
    pass
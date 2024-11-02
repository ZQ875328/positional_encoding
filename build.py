from torch.utils.cpp_extension import load as _load

import glob as _glob
import os as _os

# print('getcwd:      ', _os.getcwd())
# print('__file__:    ', __file__)
# print(_os.path.join(_os.path.dirname(__file__), 'csrc/*'))
_build = _os.path.join(_os.path.dirname(__file__), 'build')
_os.makedirs(_build, exist_ok=True)

_funcs = _load(
    name='pe',
    sources=_glob.glob(_os.path.join(_os.path.dirname(__file__), 'csrc/*')),
    extra_cflags=["-O3"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-std=c++20", "-expt-relaxed-constexpr", "-use_fast_math"],
    build_directory=_build
)

pe = _funcs.pe
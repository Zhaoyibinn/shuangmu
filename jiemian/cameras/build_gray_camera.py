"""Build the local callback adapter against an existing Shengxiang SDK."""
import argparse
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def build(sdk_root=None):
    try:
        import pybind11
        binding_include = pybind11.get_include()
    except ImportError:
        binding_include = '/usr/include'
        if not (Path(binding_include) / 'pybind11/pybind11.h').is_file():
            raise RuntimeError('缺少 pybind11 头文件，请安装 pybind11 后重新编译')
    sdk = Path(sdk_root or PROJECT_ROOT.parent / 'shenxiang_pysdk') / 'MPSizectorS_Release'
    if not (sdk / 'inc/MPSizectorS_API.h').is_file():
        raise RuntimeError(f'未找到盛相 SDK：{sdk}；请执行 python jiemian/cameras/build_gray_camera.py --sdk-root SDK目录')
    output = ROOT / ('_gray_camera' + sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['c++', '-O2', '-shared', '-std=c++14', '-fPIC', '-pthread',
        '-I' + binding_include, '-I' + sysconfig.get_path('include'), '-I' + str(sdk / 'inc'),
        str(ROOT / 'native/gray_camera.cpp'), '-L' + str(sdk / 'lib'), '-lMPSizectorS_API',
        '-Wl,-rpath,' + str(sdk / 'lib'), '-o', str(output)], check=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root')
    print(build(parser.parse_args().sdk_root))

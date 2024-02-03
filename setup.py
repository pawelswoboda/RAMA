import glob
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pkg_resources import parse_version


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = parse_version(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < parse_version('3.1.0'):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def _validate_gcc_version(self, gcc_command):
        print(f'Testing {gcc_command}...')
        out = subprocess.check_output([gcc_command, '--version']).decode()
        if 'clang' in out.lower():
            return False
        words = out.split('\n')[0].split(' ')
        for word in reversed(words):
            if "." in word:
                gcc_version = parse_version(word)
                print(f"...has version {gcc_version}")
                if gcc_version >= parse_version('8.0'):
                    return True

        return False

    def _get_all_gcc_commands(self):
        all_path_dirs = subprocess.check_output("echo -n $PATH", shell=True).decode("utf-8").rstrip().split(":")

        all_gcc_commands = ['gcc']
        for path_dir in all_path_dirs:
            if not os.path.exists(path_dir):
                continue
            local_gccs = [s for s in os.listdir(path_dir) if re.search(r'^gcc-[0-9].?.?.?', s)]
            local_gccs = [s for s in local_gccs if os.access(os.path.join(path_dir, s), os.X_OK)]
            all_gcc_commands.extend(local_gccs)
        return all_gcc_commands


    def _find_suitable_gcc_gpp(self):
        # lists all gcc version in PATH
        all_gccs = self._get_all_gcc_commands()

        for gcc in all_gccs:
            if self._validate_gcc_version(gcc):
                matching_gpp = gcc.replace("cc", "++")
                print(f'Found suitable gcc/g++ version {gcc} {matching_gpp}')
                return gcc, matching_gpp

        raise RuntimeError("gcc >= 8.0 not found on the system")


    def _prepare_environment(self):
        gcc, gpp = self._find_suitable_gcc_gpp()

        gcc_path = subprocess.check_output(f"which {gcc}", shell=True).decode("utf-8").rstrip()
        gpp_path = subprocess.check_output(f"which {gpp}", shell=True).decode("utf-8").rstrip()

        os.environ["CC"] = gcc_path
        os.environ["CXX"] = gpp_path

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j8']
        build_with_torch = os.environ.get("WITH_TORCH", "OFF")
        print(f"build_with_torch:{build_with_torch}\n\n")

        if build_with_torch.upper() == 'ON':
            print("Building with torch")
            cmake_args += ['-DWITH_TORCH=ON']
            import torch
            cmake_args += ['-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path]
            print(cmake_args)
        self._prepare_environment()
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', ext.name] + build_args, cwd=self.build_temp)

setup(
    name='RAMA',
    version='0.0.8',
    description='Bindings for RAMA: Rapid algorithm for multicut.',
    packages=find_packages('.'),
    ext_modules=[CMakeExtension(name='rama_py')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    setup_requires=['wheel']
)

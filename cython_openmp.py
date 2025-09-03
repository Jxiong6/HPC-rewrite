# ================================
# File: cython_openmp.py (v2)
# ================================
from IPython.core.magic import Magics, magics_class, cell_magic
from setuptools import Extension
from Cython.Build import cythonize
import tempfile
import os
import sys
import importlib.util
import shutil
import subprocess
import uuid
import glob
import textwrap
import argparse
from dataclasses import dataclass

BANNER = "🔧 Cython + OpenMP 并行编译器 v2 (magic: %cython_openmp)"

CACHE_DIR = ".cython_openmp_cache"


def _write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _detect_platform_flags():
    is_win = sys.platform == "win32"
    is_macos = sys.platform == "darwin"
    compile_args, link_args, include_dirs, library_dirs = [], [], [], []
    if is_win:
        compile_args += ["/O2", "/openmp"]
        link_args   += ["/openmp"]
    else:
        if is_macos:
            # Clang on macOS
            compile_args += ["-O3", "-Xpreprocessor", "-fopenmp"]
            link_args   += ["-lomp"]
            for base in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
                if os.path.isdir(base):
                    include_dirs.append(os.path.join(base, "include"))
                    lib_dir = os.path.join(base, "lib")
                    library_dirs.append(lib_dir)
                    # 关键: 运行时 rpath，避免找不到 libomp.dylib
                    link_args += [f"-Wl,-rpath,{lib_dir}"]
        else:
            # Linux / others
            compile_args += ["-O3", "-fopenmp"]
            link_args   += ["-fopenmp"]
    return compile_args, link_args, include_dirs, library_dirs


def _build_setup_code(module_name: str, schedule: str):
    compile_args, link_args, extra_includes, extra_libdirs = _detect_platform_flags()
    # 将 prange 调度策略固化在生成的 .pyx（Cython 需要字面量），这里仅用于信息打印
    setup_code = f"""
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

compile_args = {compile_args!r}
link_args    = {link_args!r}
include_dirs = [np.get_include()] + {extra_includes!r}
library_dirs = {extra_libdirs!r}

ext_modules = [
    Extension(
        name="{module_name}",
        sources=["{module_name}.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c",
    )
]

setup(
    name="{module_name}",
    ext_modules=cythonize(ext_modules, language_level=3),
)
"""
    return setup_code


def _load_generate_module():
    generate_path = os.path.join(os.getcwd(), "generate_cython_openmp_files.py")
    if not os.path.isfile(generate_path):
        raise FileNotFoundError("未找到 generate_cython_openmp_files.py（需要与当前工作目录同级）。")
    spec = importlib.util.spec_from_file_location("generate_cython_openmp_files", generate_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _move_built_artifact(build_dir: str, target_module_name: str) -> str:
    # 更严格的匹配：优先以模块名前缀匹配
    patterns = [
        os.path.join(build_dir, f"{target_module_name}.*.so"),
        os.path.join(build_dir, f"{target_module_name}.*.pyd"),
        os.path.join(build_dir, f"{target_module_name}.so"),
        os.path.join(build_dir, f"{target_module_name}.pyd"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        # 兜底
        candidates = glob.glob(os.path.join(build_dir, "*.so")) + glob.glob(os.path.join(build_dir, "*.pyd"))
    if not candidates:
        raise FileNotFoundError("❌ 未找到编译生成的扩展模块产物。")
    src = max(candidates, key=os.path.getmtime)

    os.makedirs(CACHE_DIR, exist_ok=True)
    dst = os.path.join(os.getcwd(), CACHE_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


@dataclass
class BuildOpts:
    name: str
    threads: int | None
    schedule: str


def _parse_line_to_opts(line: str) -> BuildOpts:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--name', dest='name', default='cython_func')
    parser.add_argument('-t', '--threads', dest='threads', type=int, default=None)
    parser.add_argument('--schedule', dest='schedule', choices=['static', 'dynamic', 'guided'], default='static')
    try:
        args = parser.parse_args(line.split())
    except SystemExit:
        # 避免魔法命令打断
        args = parser.parse_args([])
    return BuildOpts(name=args.name, threads=args.threads, schedule=args.schedule)


@magics_class
class CythonOpenMPMagics(Magics):

    @cell_magic
    def cython_openmp(self, line, cell):
        """
        用法:
          %%%%cython_openmp -n func_name [-t 线程数] [--schedule static|dynamic|guided]
          <写入要并行化的 Python 循环核>

        说明:
          -n 指定生成函数名（默认: cython_func）
          -t 为可选默认线程数（仅作为 omp_set_num_threads 提示；0/未给则由 OMP 自行决定）
          --schedule 选择 prange 调度策略（在生成代码里固化为字面量）
        """
        opts = _parse_line_to_opts(line)
        func_name = opts.name
        default_threads = opts.threads
        schedule = opts.schedule

        module_name = f"{func_name}_{uuid.uuid4().hex[:8]}"
        temp_dir = tempfile.mkdtemp(prefix="cython_omp_")

        pyx_path = os.path.join(temp_dir, f"{module_name}.pyx")
        setup_path = os.path.join(temp_dir, "setup.py")

        print(BANNER)
        print(f"📌 目标函数名: {func_name}")
        print(f"🧩 临时构建目录: {temp_dir}")
        print(f"📐 prange 调度: {schedule}")

        # 加载代码生成器
        try:
            gen_mod = _load_generate_module()
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        convert = getattr(gen_mod, "convert_python_to_cython_omp_v2", None)
        if convert is None:
            raise RuntimeError("generate_cython_openmp_files.py 中未找到 convert_python_to_cython_omp_v2")

        try:
            pyx_code = convert(cell, func_name=func_name, default_threads=default_threads, schedule=schedule)
        except TypeError:
            # 兼容旧签名
            pyx_code = gen_mod.convert_python_to_cython_omp(cell, func_name=func_name, default_threads=default_threads)

        print("\n📄 生成的 Cython 并行代码如下：\n")
        print(textwrap.indent(pyx_code, "    "))

        _write_text(pyx_path, pyx_code)
        _write_text(setup_path, _build_setup_code(module_name, schedule))

        # 编译
        prev_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            print("\n🔧 正在编译…（平台: {}, Python: {}）".format(sys.platform, sys.version.split()[0]))
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("⚠️  编译 stderr (摘要):\n", "\n".join(result.stderr.splitlines()[-50:]))
                print("❌ 编译失败。排查建议：\n"
                      "   • macOS: brew install libomp，并确认 rpath 生效\n"
                      "   • Windows: 安装 VS Build Tools（含 C++ 工具集）\n"
                      "   • 依赖: pip install -U cython numpy setuptools\n"
                      "   • 查看完整日志: 上方 stdout/stderr")
                return
            else:
                print("⚙️  编译成功。")
        finally:
            os.chdir(prev_cwd)

        # 移动产物并清理
        try:
            built_path = _move_built_artifact(temp_dir, module_name)
        except Exception as e:
            print(str(e))
            return
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # 动态加载模块并注入函数
        spec = importlib.util.spec_from_file_location(module_name, built_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        ipy = self.shell
        ipy.push({func_name: getattr(mod, func_name)})

        print(f"\n✅ 函数 `{func_name}` 已并行加速并可直接使用！")
        print(f"📌 用法示例: result = {func_name}(1000000, num_threads=4)")
        print(f"📦 扩展模块位于: {built_path}")


def load_ipython_extension(ipython):
    ipython.register_magics(CythonOpenMPMagics)



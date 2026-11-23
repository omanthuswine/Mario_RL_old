"""Dựng wheel nes-py 8.2.1 cho Windows mà không cần Visual Studio (dùng trình biên dịch Zig cài qua pip).

Vì sao: trên PyPI, nes-py chỉ có mã nguồn; cài trên Windows bình thường đòi Visual Studio Build Tools (vài GB).
Thư viện C++ của nes-py được nạp bằng ctypes và không dùng C-API của Python, nên một DLL dùng được cho
mọi phiên bản Python 3 trên Windows 64-bit. DLL chỉ phụ thuộc các thư viện UCRT có sẵn trong Windows 10/11.

Dựng lại (môi trường tạm, xóa đi sau khi xong):
    python -m venv %TEMP%\\nesbuild
    %TEMP%\\nesbuild\\Scripts\\python -m pip install ziglang wheel setuptools
    %TEMP%\\nesbuild\\Scripts\\python tools/build_nes_py_windows.py --out tools/wheels

Cài: pip install tools/wheels/nes_py-8.2.1-py3-none-win_amd64.whl   (trước pip install -e .)
"""
import argparse
import glob
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

VERSION = "8.2.1"


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="tools/wheels", help="thư mục ghi tệp .whl")
    out = Path(p.parse_args(argv).out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run(py, "-m", "pip", "download", f"nes-py=={VERSION}", "--no-deps", "--no-binary=:all:", "-d", tmp)
        with tarfile.open(tmp / f"nes_py-{VERSION}.tar.gz") as t:
            t.extractall(tmp, filter="data")
        src = tmp / f"nes_py-{VERSION}"
        cpp = sorted(glob.glob(str(src / "nes_py/nes/src/*.cpp")) + glob.glob(str(src / "nes_py/nes/src/mappers/*.cpp")))
        run(py, "-m", "ziglang", "c++", "-shared", "-O3", "-std=c++14", "-target", "x86_64-windows-gnu",
            f"-I{src / 'nes_py/nes/include'}", *cpp, "-o", src / "nes_py/lib_nes_env.dll")
        # Bỏ phần biên dịch của setup.py, đóng gói DLL vừa dựng như dữ liệu của gói (nes_env.py tìm lib_nes_env*).
        setup = src / "setup.py"
        s = setup.read_text()
        s = s.replace("    ext_modules=[LIB_NES_ENV],\n", "    package_data={'nes_py': ['lib_nes_env.dll']},\n")
        if "ext_modules" in s:
            raise SystemExit("setup.py của nes-py khác dự kiến, không sửa được")
        setup.write_text(s)
        run(py, "setup.py", "-q", "bdist_wheel", "--plat-name", "win_amd64", "--python-tag", "py3", "-d", out, cwd=src)
    print(*out.glob(f"nes_py-{VERSION}-*.whl"), sep="\n")


if __name__ == "__main__":
    main()

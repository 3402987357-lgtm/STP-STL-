# -*- mode: python ; coding: utf-8 -*-
"""我用 PyInstaller 把命令行打包成一个 exe。

我为什么要写 spec 而不是直接 `pyinstaller --onefile`：
OCP（OpenCASCADE 的 Python 绑定）里有大量**运行时动态加载**的模块，
PyInstaller 的静态分析看不见它们，打出来的 exe 一跑就
"No module named OCP.XXX"。spec 里我显式声明 hiddenimports，
并且把整个 OCP 作为二进制数据收进来，避免逐个猜。

我为什么**没有**用 --onefile：onefile 每次启动都要把几百 MB
解包到临时目录，实测启动慢很多，而甲方的使用场景是反复跑检测。
onedir 启动快，代价是分发一个目录而不是单个文件——
我用 build_installer.py 里的打包脚本把它压成 zip 解决这个代价。
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# OCP 的动态库（.pyd / .dll）必须显式收集，否则 exe 起不来。
ocp_binaries = collect_dynamic_libs("OCP")
ocp_hidden = collect_submodules("OCP")

block_cipher = None

a = Analysis(
    ["run_inspector.py"],
    pathex=["src"],
    binaries=ocp_binaries,
    datas=[],
    hiddenimports=[
        "cad_inspector",
        *ocp_hidden,
        "numpy",
        "scipy",
        "trimesh",
        "rtree",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # torch 体积巨大且只被可选的面级深度学习增强用到，
        # 缺它时几何基线照常可用。不排除的话 exe 会大出好几个 GB。
        "torch",
        "torchvision",
        "pytorch3d",
        # 测试与开发工具不应该进交付物。
        "pytest",
        "IPython",
    ],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cad-inspector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cad-inspector",
)

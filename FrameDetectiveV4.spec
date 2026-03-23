# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

ccvfi_cache = os.path.join(
    os.path.dirname(__import__('ccvfi').__file__), 'cache_models'
)

datas = [
    ('fd_icon.ico', '.'),
    ('FDIcon.png', '.'),
    # Bundle RIFE model weights where frozen ccvfi expects them
    (os.path.join(ccvfi_cache, 'RIFE_IFNet_v426_heavy.pkl'), 'cache_models'),
]
binaries = []
hiddenimports = ['ccvfi', 'ccvfi.model', 'ccvfi.model.rife', 'PIL', 'PIL.Image', 'PIL.ImageTk']
tmp_ret = collect_all('ccvfi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['frame_detective_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FrameDetective_Build2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\josko\\Documents\\tools\\frame-detective\\fd_icon.ico'],
)

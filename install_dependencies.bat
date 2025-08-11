@echo off
echo 正在安装 PyTorch 模型转 ONNX 所需的依赖库...
echo.

echo 方法1: 使用清华镜像安装 (推荐)
python -m pip install --no-input --disable-pip-version-check -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn torch transformers onnx
if %errorlevel% equ 0 (
    echo 安装成功！
    goto :success
)

echo.
echo 方法1失败，尝试方法2: 使用阿里云镜像
python -m pip install --no-input --disable-pip-version-check -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com torch transformers onnx
if %errorlevel% equ 0 (
    echo 安装成功！
    goto :success
)

echo.
echo 方法2失败，尝试方法3: 使用豆瓣镜像
python -m pip install --no-input --disable-pip-version-check -i https://pypi.douban.com/simple --trusted-host pypi.douban.com torch transformers onnx
if %errorlevel% equ 0 (
    echo 安装成功！
    goto :success
)

echo.
echo 方法3失败，尝试方法4: 官方源 (可能较慢)
python -m pip install --no-input --disable-pip-version-check torch transformers onnx
if %errorlevel% equ 0 (
    echo 安装成功！
    goto :success
)

echo.
echo 所有安装方法都失败了。请检查网络连接或手动安装。
echo 手动安装命令:
echo pip install torch transformers onnx
goto :end

:success
echo.
echo 依赖库安装完成！现在可以运行转换脚本了:
echo python convert_tinybert_to_onnx.py
echo.
echo 验证安装:
python -c "import torch, transformers, onnx; print('所有库安装成功！')"

:end
pause
from setuptools import setup, find_packages

setup(
    name="faceanlz",
    version="1.0.0",
    description="Face Detecting / Face Meshing in Video or Picture",
    author="MheenKyoungWhan",
    author_email="kwmheen@gmail.com",
    license="MheenKyoungWhan",
    packages = find_packages(),
    url="https://github.com/kwmheen357/FaceAnlz.git",
    py_modules=["pandas", "mediapipe", "cv2"],
    install_requires =[
        'mediapipe==0.8.10',
        'numpy==1.23.0',
        'pandas==1.4.2',],
)

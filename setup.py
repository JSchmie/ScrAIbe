import pkg_resources
import os
from setuptools import setup, find_packages

module_name = "scraibe"
github_url = "https://github.com/JSchmie/ScAIbe"

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "version.py"))
version = {"__file__": verfile}

with open(verfile, "r") as fp:
    exec(fp.read(), version)


############### setup ###############

build_version = "SCRAIBE_BUILD" in os.environ

version["ISRELEASED"] = True if "ISRELEASED" in os.environ else False

if __name__ == "__main__":

    setup(
        name=module_name,
        version=version["get_version"](build_version),
        packages=find_packages(),
        python_requires=">=3.8",
        readme="README.md",
        install_requires = [str(r) for r in pkg_resources.parse_requirements(
                open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
            )
        ],
        dependency_links=[
            'https://download.pytorch.org/whl/cu113',
            ],
        url= github_url,
        
        license='GPL-3',
        author='Jacob Schmieder',
        author_email='Jacob.Schmieder@dbfz.de',
        description='Transcription tool for audio files based on Whisper and Pyannote',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: GPU :: NVIDIA CUDA :: 11.2',
            'License :: OSI Approved :: Open Software License 3.0 (OSL-3.0)',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10'],
        keywords = ['transcription', 'speech recognition', 'whisper', 'pyannote', 'audio', 'ScrAIbe', 'scraibe',
                    'speech-to-text', 'speech-to-text transcription', 'speech-to-text recognition',
                    'voice-to-speech'],
        package_data={'scraibe.app' : ["*.html", "*.svg","*.yml"]},
        entry_points={'console_scripts':
            ['scraibe = scraibe.cli:cli']}
        
    )

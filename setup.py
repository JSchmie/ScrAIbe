import pkg_resources
import os
from setuptools import setup, find_packages

module_name = "autotranscript"
github_url = "https://github.com/JSchmie/autotranscript"

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "version.py"))
version = {"__file__": verfile}

with open(verfile, "r") as fp:
    exec(fp.read(), version)


############### setup ###############

build_version = "AUTOTRANSCRIPT_BUILD" in os.environ

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
        license='',
        author='Jacob Schmieder',
        author_email='Jacob.Schmieder@dbfz.de',
        description='Transcription tool for audio files based on Whisper and Pyannote',
        entry_points={'console_scripts':
            ['autotranscript = autotranscript.cli:cli']}
    )

"""Mirrors the pip install steps scrypted's plugin_remote.py performs for a plugin.

usage: install_plugin.py <scrypted-checkout> <plugin-dir> <pip-target>
"""
import glob
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

LOREM = "Lorem ipsum.txt"


def restore_lorem_ipsum():
    """Workaround: some portable-python zips omit setuptools' vendored 'Lorem ipsum.txt'
    (the zip step word-split filenames containing spaces), which makes setuptools fail to
    import and breaks every sdist build. Put the file back from the matching setuptools wheel.
    """
    site = sysconfig.get_paths()["purelib"]
    dest_dir = os.path.join(site, "setuptools", "_vendor", "jaraco", "text")
    dest = os.path.join(dest_dir, LOREM)
    if os.path.exists(dest):
        print("lorem ipsum workaround: not needed, file present")
        return
    dist = glob.glob(os.path.join(site, "setuptools-*.dist-info"))
    if not dist or not os.path.isdir(dest_dir):
        print("lorem ipsum workaround: setuptools not found, skipping")
        return
    version = os.path.basename(dist[0])[len("setuptools-"):-len(".dist-info")]
    print(f"lorem ipsum workaround: restoring '{LOREM}' from setuptools=={version} wheel")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "-q", f"setuptools=={version}", "-d", tmp])
        with zipfile.ZipFile(glob.glob(os.path.join(tmp, "setuptools-*.whl"))[0]) as z:
            name = next(n for n in z.namelist() if n.endswith("/" + LOREM))
            with open(dest, "wb") as f:
                f.write(z.read(name))
    print("lorem ipsum workaround: restored")


scrypted, plugin, target = sys.argv[1:4]
if os.environ.get("RESTORE_LOREM_IPSUM", "true").lower() == "true":
    restore_lorem_ipsum()
else:
    print("lorem ipsum workaround: disabled")
sys.path.insert(0, os.path.join(scrypted, "server", "python"))
from plugin_pip import install_with_pip  # noqa: E402

with open(os.path.join(plugin, "package.json")) as f:
    package_json = json.load(f)


def read(name):
    p = os.path.join(plugin, "src", name)
    return open(p).read().strip() if os.path.exists(p) else ""


# same base requirements as SCRYPTED_REQUIREMENTS in plugin_remote.py
base = "ptpython\nwheel"
install_with_pip(target, package_json, base, os.path.join(target, "requirements.scrypted"), ignore_error=True)

requirements = read("requirements.txt")
if requirements:
    install_with_pip(target, package_json, requirements, os.path.join(target, "requirements"))

optional = read("requirements.optional.txt")
if optional:
    install_with_pip(target, package_json, optional, os.path.join(target, "requirements.optional"), ignore_error=True)

# verify the interpreter can see what was installed
sys.path.insert(0, target)
print("OK", sys.version)

"""Mirrors the pip install steps scrypted's plugin_remote.py performs for a plugin.

usage: install_plugin.py <scrypted-checkout> <plugin-dir> <pip-target>
"""
import json
import os
import sys

scrypted, plugin, target = sys.argv[1:4]
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

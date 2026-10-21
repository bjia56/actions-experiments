// Diagnoses why sdist builds fail in portable python: checks the bundled setuptools' data files,
// then tries building imutils (an sdist-only package) with and without build isolation.
const path = require('path');
const fs = require('fs');
const child_process = require('child_process');
const { PortablePython } = require(path.join(process.cwd(), 'node_modules', 'py'));

const [work, version = '3.10'] = process.argv.slice(2);

function run(python, args, label) {
    console.log(`::group::${label}`);
    console.log('$', python, args.join(' '));
    const r = child_process.spawnSync(python, args, { stdio: 'inherit' });
    console.log(`exit: ${r.status}`);
    console.log('::endgroup::');
    return r.status;
}

(async () => {
    const py = new PortablePython(version, path.join(work, 'diag', version));
    await py.install();
    const python = py.executablePath;
    const root = path.resolve(path.dirname(python), '..');
    console.log('python root:', root);

    run(python, ['-c', `
import sys, os, setuptools, pip
print(sys.version)
print('setuptools', setuptools.__version__, setuptools.__file__)
print('pip', pip.__version__)
d = os.path.join(os.path.dirname(setuptools.__file__), '_vendor', 'jaraco', 'text')
print('jaraco/text exists:', os.path.isdir(d))
if os.path.isdir(d):
    for f in sorted(os.listdir(d)): print('  ', repr(f))
# every non-python data file shipped under setuptools/_vendor
vendor = os.path.join(os.path.dirname(setuptools.__file__), '_vendor')
for dp, dn, fn in os.walk(vendor):
    for f in fn:
        if not f.endswith(('.py', '.pyc')):
            print('data:', os.path.relpath(os.path.join(dp, f), vendor))
`], 'inspect setuptools');

    run(python, ['-c', 'import setuptools._vendor.jaraco.text; print("import ok")'], 'import jaraco.text');
    run(python, ['-m', 'pip', 'list'], 'pip list');
    run(python, ['-m', 'pip', 'install', '--no-cache-dir', '--target', path.join(work, 'diag', 'imutils-default'), 'imutils'], 'pip install imutils (default)');
    run(python, ['-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation', '--target', path.join(work, 'diag', 'imutils-noiso'), 'imutils'], 'pip install imutils (--no-build-isolation)');
    run(python, ['-m', 'pip', 'download', '--no-deps', '--no-binary', ':all:', '--no-build-isolation', '-d', path.join(work, 'diag', 'dl'), 'imutils'], 'pip download imutils sdist');
})().catch(e => { console.error(e); process.exit(1); });

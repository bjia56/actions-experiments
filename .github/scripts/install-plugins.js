// Installs portable python + pip requirements for every scrypted python plugin,
// the same way the scrypted server does, and writes a pass/fail summary.
//
// usage: node install-plugins.js <scrypted-checkout> <work-dir> <exclude,comma,list>
const fs = require('fs');
const os = require('os');
const path = require('path');
const child_process = require('child_process');
const { PortablePython } = require(path.join(process.cwd(), 'node_modules', 'py'));

const [scrypted, work, excludeArg] = process.argv.slice(2);
const exclude = new Set((excludeArg || '').split(',').filter(Boolean));
const defaultVersion = require(path.join(scrypted, 'server', 'bin', 'packaged-python.js')).version;

const pluginsDir = path.join(scrypted, 'plugins');
const plugins = fs.readdirSync(pluginsDir).filter(name => {
    if (exclude.has(name))
        return false;
    const pkgPath = path.join(pluginsDir, name, 'package.json');
    if (!fs.existsSync(pkgPath))
        return false;
    const pkg = JSON.parse(fs.readFileSync(pkgPath));
    return pkg.scrypted?.runtime === 'python' && fs.existsSync(path.join(pluginsDir, name, 'src', 'requirements.txt'));
});
console.log('plugins:', plugins.join(', '));

const installed = new Map();
async function getPython(version, options) {
    const key = JSON.stringify([version, options]);
    if (!installed.has(key)) {
        const py = new PortablePython(version, path.join(work, 'py', version), options);
        await py.install();
        installed.set(key, py.executablePath);
    }
    return installed.get(key);
}

(async () => {
    const results = [];
    for (const name of plugins) {
        const dir = path.join(pluginsDir, name);
        const pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json')));
        const pv = pkg.scrypted.pythonVersion;
        const version = pv?.[os.platform()]?.[os.arch()] || pv?.default || defaultVersion;
        const options = pv?.options?.[os.platform()]?.[os.arch()] || pv?.options?.default || {};
        console.log(`::group::${name} (python ${version})`);
        let status = 'pass', detail = '';
        try {
            const python = await getPython(version, options);
            const r = child_process.spawnSync(python, [
                path.join(__dirname, 'install_plugin.py'), scrypted, dir, path.join(work, 'volume', name),
            ], { stdio: 'inherit', env: { ...process.env, PYTHONUNBUFFERED: '1' } });
            if (r.status)
                throw new Error(`install_plugin.py exited ${r.status}`);
        }
        catch (e) {
            status = 'FAIL';
            detail = e.message;
            console.log(`::error title=${name}::${detail}`);
        }
        console.log('::endgroup::');
        results.push({ name, version, status, detail });
    }

    let md = `### ${process.env.RUNNER_OS} ${process.env.RUNNER_ARCH}\n\n| plugin | python | result |\n|---|---|---|\n`;
    for (const r of results)
        md += `| ${r.name} | ${r.version} | ${r.status === 'pass' ? '✅' : '❌ ' + r.detail} |\n`;
    console.log(md);
    if (process.env.GITHUB_STEP_SUMMARY)
        fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, md);
    process.exit(results.some(r => r.status !== 'pass') ? 1 : 0);
})();

// Installs portable python + pip requirements for every scrypted python plugin,
// the same way the scrypted server does, and writes a pass/fail summary.
//
// then runs functional smoke checks (smoke_checks.py) against each plugin's installed dependencies
// unless SMOKE_CHECKS=false.
//
// usage: node install-plugins.js <scrypted-checkout> <work-dir> <exclude,comma,list>
const fs = require('fs');
const os = require('os');
const path = require('path');
const child_process = require('child_process');
const { PortablePython } = require(path.join(process.cwd(), 'node_modules', 'py'));

const [scrypted, work, excludeArg] = process.argv.slice(2);
const exclude = new Set((excludeArg || '').split(',').filter(Boolean));
// packaged-python.js requires 'py', which isn't installed under the scrypted checkout, so parse the version out
const defaultVersion = fs.readFileSync(path.join(scrypted, 'server', 'bin', 'packaged-python.js'), 'utf8').match(/version:\s*'([^']+)'/)[1];
console.log('server default python:', defaultVersion);

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

const smokeEnabled = (process.env.SMOKE_CHECKS || 'true').toLowerCase() !== 'false';

function runSmoke(python, plugin, target) {
    const jsonPath = path.join(work, 'smoke', `${plugin}.json`);
    fs.mkdirSync(path.dirname(jsonPath), { recursive: true });
    fs.rmSync(jsonPath, { force: true });
    const r = child_process.spawnSync(python, [path.join(__dirname, 'smoke_checks.py'), plugin, target].filter(Boolean), {
        stdio: 'inherit',
        env: { ...process.env, PYTHONUNBUFFERED: '1', SMOKE_JSON: jsonPath },
    });
    if (!fs.existsSync(jsonPath)) {
        // the interpreter died before reporting (e.g. a native crash)
        const why = r.signal ? `killed by ${r.signal}` : `exit ${r.status}`;
        console.log(`::error title=${plugin} smoke::checks did not complete (${why})`);
        return { status: 'FAIL', summary: `❌ crashed (${why})`, checks: [] };
    }
    const { checks } = JSON.parse(fs.readFileSync(jsonPath));
    const failed = checks.filter(c => c.status === 'FAIL');
    const skipped = checks.filter(c => c.status === 'skip');
    const ok = checks.length - failed.length - skipped.length;
    for (const c of failed)
        console.log(`::error title=${plugin} smoke ${c.name}::${c.detail}`);
    for (const c of skipped)
        console.log(`::notice title=${plugin} smoke ${c.name}::skipped: ${c.detail}`);
    if (failed.length)
        return { status: 'FAIL', summary: `❌ ${failed.map(c => c.name).join(', ')}`, checks };
    return { status: 'pass', summary: `✅ ${ok} ok${skipped.length ? `, ${skipped.length} skipped` : ''}`, checks };
}

(async () => {
    const results = [];
    const pythonsUsed = new Map();
    for (const name of plugins) {
        const dir = path.join(pluginsDir, name);
        const pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json')));
        const pv = pkg.scrypted.pythonVersion;
        const version = pv?.[os.platform()]?.[os.arch()] || pv?.default || defaultVersion;
        const options = pv?.options?.[os.platform()]?.[os.arch()] || pv?.options?.default || {};
        const target = path.join(work, 'volume', name);
        console.log(`::group::${name} (python ${version})`);
        let status = 'pass', detail = '';
        let smoke = { status: 'n/a', summary: '–', checks: [] };
        try {
            const python = await getPython(version, options);
            pythonsUsed.set(version, python);
            const r = child_process.spawnSync(python, [
                path.join(__dirname, 'install_plugin.py'), scrypted, dir, target,
            ], { stdio: 'inherit', env: { ...process.env, PYTHONUNBUFFERED: '1' } });
            if (r.status)
                throw new Error(`install_plugin.py exited ${r.status}`);
            if (smokeEnabled) {
                console.log(`--- smoke checks: ${name}`);
                smoke = runSmoke(python, name, target);
            }
        }
        catch (e) {
            status = 'FAIL';
            detail = e.message;
            console.log(`::error title=${name}::${detail}`);
        }
        console.log('::endgroup::');
        results.push({ name, version, status, detail, smoke });
    }

    // the interpreter itself, once per python version that was used
    const interpreters = [];
    if (smokeEnabled) {
        for (const [version, python] of pythonsUsed) {
            console.log(`::group::interpreter (python ${version})`);
            interpreters.push({ name: `interpreter`, version, smoke: runSmoke(python, '_interpreter', '') });
            console.log('::endgroup::');
        }
    }

    let md = `### ${process.env.RUNNER_OS} ${process.env.RUNNER_ARCH}\n\n| plugin | python | install | smoke checks |\n|---|---|---|---|\n`;
    for (const r of interpreters)
        md += `| ${r.name} | ${r.version} | – | ${r.smoke.summary} |\n`;
    for (const r of results)
        md += `| ${r.name} | ${r.version} | ${r.status === 'pass' ? '✅' : '❌ ' + r.detail} | ${r.smoke.summary} |\n`;

    // details for anything that was skipped or failed, and every check's detail line
    const detailRows = [...interpreters.map(r => ({ ...r, name: `${r.name} ${r.version}` })), ...results];
    const notable = detailRows.flatMap(r => r.smoke.checks.filter(c => c.status !== 'ok').map(c => `- **${r.name}** \`${c.name}\` ${c.status}: ${c.detail}`));
    if (notable.length)
        md += `\n${notable.join('\n')}\n`;
    const all = detailRows.flatMap(r => r.smoke.checks.filter(c => c.status === 'ok' && c.detail).map(c => `- ${r.name} \`${c.name}\`: ${c.detail}`));
    if (all.length)
        md += `\n<details><summary>check details</summary>\n\n${all.join('\n')}\n\n</details>\n`;
    console.log(md);
    if (process.env.GITHUB_STEP_SUMMARY)
        fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, md);
    const failed = results.some(r => r.status !== 'pass' || r.smoke.status === 'FAIL') || interpreters.some(r => r.smoke.status === 'FAIL');
    process.exit(failed ? 1 : 0);
})();

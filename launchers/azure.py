"""Decode stages on Azure CPUs (the owner's Azure for Students credits): the out-of-fold sweep and the decoding of
new pages.

The subscription may deploy to five regions, at most 6 vCPUs in each and 4 per VM family, so the pool is one 4-vCPU
VM per region that offers one (up to 20 processes). A run spreads its (fold, chunk) tasks over the pool by size; each
VM decodes its share (4 processes, one thread each) on the committed code and the inputs named, and the parts are
combined here into the files the local commands write: a sweep's by `trogs26.decode.sweep.combine`, new pages' by
`trogs26.decode.newpages.write_run`.
The pool is deallocated after each sweep, and each VM deallocates itself after 30 minutes without a sweep worker
(should a run be cut off); `down` deletes it.

    python launchers/azure.py up                       create or start the pool and install the environment
    TROGS26_WORK=<work> python launchers/azure.py sweep --variant P --config <file> --run <name> --inputs a,b,...
                                                        [--folds 0,1] [--chunks 8] [--keep] [--allow-dirty]
                                                        [--collect]
    TROGS26_WORK=<work> python launchers/azure.py newpages --variant P --config <file> --run <name>
                                                        --splits trogs26 --inputs a,b,... [--chunks 4]
                                                        [--keep] [--allow-dirty] [--collect]
    python launchers/azure.py status | stop | down

`--inputs` are paths under TROGS26_WORK the decode reads (symbolic links are followed), e.g. for the page network:
annotations, lm, chains/f{k}/oof/det and chains/f{k}/oof/pagenet_<version>.npz (new pages: chains/f{k}/new/...).
`--collect` (same arguments) only waits for and combines a run already sent, e.g. after this process was cut off.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GROUP, GROUP_REGION = 'trogs26', 'switzerlandnorth'
REGIONS = ('switzerlandnorth', 'belgiumcentral', 'spaincentral', 'polandcentral', 'denmarkeast')
# 4-vCPU sizes of families with quota, the first a region allows (2026-11-07: Belgium Central D4as_v4, Spain Central
# D4s_v4, Denmark East only the older D3_v2, which boots generation-1 images only; Poland Central and Switzerland
# North none of them)
SIZES = ('Standard_D4as_v4', 'Standard_D4s_v4', 'Standard_D4ds_v4', 'Standard_D4s_v3', 'Standard_D3_v2')
GEN1 = {'Standard_D3_v2'}
PROCESSES = 4
USER = 'trogs'
KEY = Path.home() / '.ssh' / 'trogs26_azure'
KNOWN_HOSTS = Path.home() / '.ssh' / 'trogs26_azure_known_hosts'
VENV = '/opt/trogs26'
# The decode environment: the pins of the GPU jobs' environment (gcp/requirements-gpu.txt), torch as its CPU build.
PINS = ('numpy==2.4.6', 'scipy==1.17.1', 'rapidfuzz==3.14.6', 'opencv-python-headless==5.0.0.93',
        'textdistance==4.6.3', 'matplotlib==3.11.2', 'transformers==4.56.2', 'safetensors==0.6.2',
        'huggingface_hub==0.36.2')
CLOUD_INIT = f"""#cloud-config
packages: [rsync]
runcmd:
  - curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
  - env UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.11 {VENV}
  - uv pip install --python {VENV}/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
  - uv pip install --python {VENV}/bin/python {' '.join(PINS)}
  - touch /var/lib/trogs26-ready
"""
# Run by cron every 10 minutes on each VM. Azure's auto-shutdown schedules do not exist in these regions, and a VM shut
# down from inside stays allocated (and billed), so the VM deallocates itself through its managed identity, which may
# do that to this VM only (role 'Virtual Machine Contributor' on its own resource id).
IDLE = r"""#!/bin/bash
marker=/var/lib/trogs26-active
if pgrep -f "azure.py worker" > /dev/null || [ ! -e $marker ]; then touch $marker; exit 0; fi
[ $(( $(date +%s) - $(stat -c %Y $marker) )) -lt 1800 ] && exit 0
meta=http://169.254.169.254/metadata
token=$(curl -s -H Metadata:true "$meta/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')
id=$(curl -s -H Metadata:true "$meta/instance/compute/resourceId?api-version=2021-02-01&format=text")
rm -f $marker
curl -s -X POST -H "Authorization: Bearer $token" -H 'Content-Length: 0' \
    "https://management.azure.com$id/deallocate?api-version=2024-07-01"
"""


def az(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['az', *args, '--only-show-errors'], text=True, capture_output=True, check=check)


def vm_name(region: str) -> str:
    return f'trogs26-{region}'


def ssh_args() -> list[str]:
    return ['-i', str(KEY), '-o', 'StrictHostKeyChecking=accept-new', '-o', f'UserKnownHostsFile={KNOWN_HOSTS}',
            '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes']


def ssh(ip: str, command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['ssh', *ssh_args(), f'{USER}@{ip}', command], text=True, capture_output=True, check=check)


def pool() -> dict[str, dict]:
    """region -> {'name', 'state', 'ip'} of the pool's VMs."""
    found = az('vm', 'list', '-g', GROUP, '-d', '--query', '[].{name:name, region:location, state:powerState, '
               'ip:publicIps}', '-o', 'json', check=False)
    rows = json.loads(found.stdout) if found.returncode == 0 and found.stdout.strip() else []
    return {row['region']: row for row in rows if row['name'] == vm_name(row['region'])}


def create(region: str, ip: str) -> str:
    """Create the region's VM (the first size the region allows); its public IP."""
    name = vm_name(region)
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as init:
        init.write(CLOUD_INIT)
    errors = []
    for size in SIZES:
        image = 'Canonical:ubuntu-24_04-lts:server-gen1:latest' if size in GEN1 else 'Ubuntu2404'
        made = az('vm', 'create', '-g', GROUP, '-n', name, '-l', region, '--size', size, '--image', image,
                  '--admin-username', USER, '--ssh-key-values', f'{KEY}.pub', '--public-ip-sku', 'Standard',
                  '--storage-sku', 'StandardSSD_LRS', '--os-disk-size-gb', '32', '--custom-data', init.name,
                  '--nsg-rule', 'SSH', '-o', 'json', check=False)
        if made.returncode == 0:
            break
        errors.append(f'{size}: {made.stderr.strip().splitlines()[-1] if made.stderr.strip() else made.returncode}')
    else:
        os.unlink(init.name)
        # a failed create leaves the VM's network behind (the public IP is billed)
        for kind, suffix in (('nic', 'VMNic'), ('public-ip', 'PublicIP'), ('nsg', 'NSG'), ('vnet', 'VNET')):
            az('network', kind, 'delete', '-g', GROUP, '-n', f'{name}{suffix}', check=False)
        raise RuntimeError(f'{region}: no size could be created\n' + '\n'.join(errors))
    os.unlink(init.name)
    az('network', 'nsg', 'rule', 'update', '-g', GROUP, '--nsg-name', f'{name}NSG', '-n', 'default-allow-ssh',
       '--source-address-prefixes', f'{ip}/32')
    vm_id = json.loads(made.stdout)['id']
    az('vm', 'identity', 'assign', '-g', GROUP, '-n', name, '--role', 'Virtual Machine Contributor', '--scope', vm_id)
    print(region, size, flush=True)
    return json.loads(made.stdout)['publicIpAddress']


def wait_ready(region: str, ip: str, timeout: float = 1200) -> None:
    """Wait for the VM's environment, then install the idle check and mark the VM active (a marker left from before
    the VM was last stopped would let the check deallocate it as soon as its first worker ends)."""
    deadline = time.time() + timeout
    while ssh(ip, 'test -f /var/lib/trogs26-ready', check=False).returncode != 0:
        if time.time() > deadline:
            raise TimeoutError(f'{region} ({ip}) is not ready after {timeout:.0f} s')
        time.sleep(15)
    subprocess.run(['ssh', *ssh_args(), f'{USER}@{ip}',
                    'sudo tee /usr/local/bin/trogs26-idle > /dev/null && sudo chmod +x /usr/local/bin/trogs26-idle && '
                    "echo '*/10 * * * * root /usr/local/bin/trogs26-idle' | sudo tee /etc/cron.d/trogs26-idle "
                    '> /dev/null && sudo touch /var/lib/trogs26-active'], input=IDLE, text=True, capture_output=True,
                   check=True)


def up(create_missing: bool = True) -> dict[str, str]:
    """Start deallocated VMs (and, with `create_missing`, create those missing) and wait until every one has its
    environment; region -> IP of the VMs ready (a region that allows none of the sizes is left out)."""
    if not KEY.exists():
        subprocess.run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-q', '-f', str(KEY), '-C', 'trogs26-azure'],
                       check=True)
    az('group', 'create', '-n', GROUP, '-l', GROUP_REGION, '-o', 'none')
    ip = urllib.request.urlopen('https://api.ipify.org', timeout=20).read().decode().strip()
    existing = pool()

    def bring(region: str) -> tuple[str, str | None]:
        row = existing.get(region)
        if row is None:
            if not create_missing:
                return region, None
            try:
                address = create(region, ip)
            except RuntimeError as error:
                print(error, flush=True)
                return region, None
        else:
            if row['state'] != 'VM running':
                az('vm', 'start', '-g', GROUP, '-n', row['name'])
            az('network', 'nsg', 'rule', 'update', '-g', GROUP, '--nsg-name', f"{row['name']}NSG", '-n',
               'default-allow-ssh', '--source-address-prefixes', f'{ip}/32')
            address = row['ip'] or pool()[region]['ip']
        wait_ready(region, address)
        return region, address

    with cf.ThreadPoolExecutor(len(REGIONS)) as threads:
        ready = {region: address for region, address in threads.map(bring, REGIONS) if address}
    if not ready:
        sys.exit('no VM could be created')
    print('ready:', ', '.join(f'{r} {a}' for r, a in ready.items()), flush=True)
    return ready


def stop() -> None:
    for row in pool().values():
        az('vm', 'deallocate', '-g', GROUP, '-n', row['name'], '--no-wait')
    print('deallocating', len(pool()), 'VMs')


def down() -> None:
    az('group', 'delete', '-n', GROUP, '--yes', '--no-wait')
    KNOWN_HOSTS.unlink(missing_ok=True)
    print('deleting resource group', GROUP)


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=REPO, text=True, check=True, capture_output=True).stdout.strip()


def balance(tasks: list[tuple[int, list[str]]], weights: list[int], bins: int) -> list[list[int]]:
    """Task indices in `bins` groups of about equal total weight (heaviest first, each to the lightest group)."""
    groups: list[list] = [[0, []] for _ in range(bins)]
    for i in sorted(range(len(tasks)), key=lambda i: -weights[i]):
        lightest = min(groups, key=lambda g: g[0])
        lightest[0] += weights[i]
        lightest[1].append(i)
    return [g[1] for g in groups]


def head_archive(tmp: Path, allow_dirty: bool) -> tuple[Path, str]:
    """HEAD as a tar.gz in `tmp`, and its commit; tracked files must match it unless `allow_dirty`."""
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty and not allow_dirty:
        sys.exit('tracked files differ from HEAD: commit before a run (or --allow-dirty to run HEAD)')
    archive = tmp / 'code.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(archive), 'HEAD'], cwd=REPO, check=True)
    return archive, git('rev-parse', 'HEAD')


def task_weights(work, group: str, tasks: list[tuple[int, list[str]]]) -> list[int]:
    """Each task's detections (its share of the decoding time) in chain group `group` ('oof' or 'new')."""
    from trogs26 import artifacts
    return [sum(len(artifacts.det_page(work.chain(fold, group, bases), b)['boxes']) for b in bases)
            for fold, bases in tasks]


def run_tasks(work_root: Path, name: str, payload: dict, tasks: list[tuple[int, list[str]]], weights: list[int],
              inputs: list[str], archive: Path, commit: str, collect: bool) -> tuple[list[dict], dict]:
    """Spread `tasks` over the pool by `weights`, have each VM's worker run its share with `payload` on the tree of
    `archive` (commit `commit`) and the `inputs`, and return the workers' parts with a record of the run."""
    ready = up(create_missing=False)
    regions = sorted(ready)
    shares = balance(tasks, weights, len(regions))
    remote = f'runs/{name}'
    with tempfile.TemporaryDirectory() as tmp:
        def send(region: str, share: list[int]) -> None:
            ip = ready[region]
            spec = Path(tmp) / f'{region}.json'
            spec.write_text(json.dumps({**payload, 'tasks': [tasks[i] for i in share]}))
            ssh(ip, f'rm -rf repo {remote} && mkdir -p repo work {remote}')
            subprocess.run(['scp', *ssh_args(), str(archive), f'{USER}@{ip}:{remote}/code.tar.gz'], check=True,
                           capture_output=True)
            subprocess.run(['scp', *ssh_args(), str(spec), f'{USER}@{ip}:{remote}/'], check=True,
                           capture_output=True)
            ssh(ip, f'tar -xzf {remote}/code.tar.gz -C repo')
            subprocess.run(['rsync', '-azLR', '--delete-after', '-e', ' '.join(['ssh', *ssh_args()]),
                            *[f'./{p}' for p in inputs], f'{USER}@{ip}:work/'], cwd=work_root, check=True,
                           capture_output=True)
            ssh(ip, f'cd ~ && TROGS26_WORK=$HOME/work PYTHONPATH=$HOME/repo nohup {VENV}/bin/python '
                    f'repo/launchers/azure.py worker {remote}/{region}.json {remote}/parts.json '
                    f'> {remote}/log.txt 2>&1 < /dev/null &')

        if not collect:
            with cf.ThreadPoolExecutor(len(regions)) as threads:
                list(threads.map(send, regions, shares))
            print(f'{len(tasks)} tasks on {len(regions)} VMs, commit {commit[:7]}', flush=True)
        parts, pending = [], set(regions)
        while pending:
            time.sleep(30)
            for region in sorted(pending):
                ip = ready[region]
                if ssh(ip, f'test -f {remote}/parts.json', check=False).returncode == 0:
                    local = Path(tmp) / f'{region}.parts.json'
                    subprocess.run(['scp', *ssh_args(), f'{USER}@{ip}:{remote}/parts.json', str(local)], check=True,
                                   capture_output=True)
                    parts += json.loads(local.read_text())
                    pending.discard(region)
                    print(f'{region} done ({len(regions) - len(pending)}/{len(regions)})', flush=True)
                elif ssh(ip, 'pgrep -f "azure.py worker" > /dev/null', check=False).returncode != 0:
                    log = ssh(ip, f'tail -n 20 {remote}/log.txt', check=False).stdout
                    raise RuntimeError(f'{region}: the worker stopped without results\n{log}')
    return parts, {'commit': commit, 'regions': regions, 'tasks': len(tasks), 'collected': collect}


def sweep(variant: str, config_path: str, run: str, inputs: list[str], folds: str, chunks: int, keep: bool,
          allow_dirty: bool, collect: bool) -> None:
    sys.path.insert(0, str(REPO))
    from trogs26 import config, data
    from trogs26.decode import sweep as decode_sweep
    from trogs26.layout import Work
    cfg, work, started = config.load(config_path), Work.from_env(), time.perf_counter()
    fold_list = [int(k) for k in folds.split(',')] if folds else sorted(data.folds(cfg['data']['folds']))
    tasks = [(fold, bases) for fold in fold_list for bases in decode_sweep.chunks(work, cfg, fold, chunks)]
    payload = {'action': 'sweep', 'cfg': dict(cfg), 'variant': variant}
    with tempfile.TemporaryDirectory() as tmp:
        archive, commit = head_archive(Path(tmp), allow_dirty)
        parts, record = run_tasks(work.root, f'{run}_{variant}', payload, tasks, task_weights(work, 'oof', tasks),
                                  inputs, archive, commit, collect)
    result = decode_sweep.combine(cfg, variant, fold_list, parts, time.perf_counter() - started)
    result['azure'] = record
    path = decode_sweep.sweep_path(work, run, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    print(path, decode_sweep.frozen_combo(result['scores']), 'seconds', result['seconds'], flush=True)
    if not keep:
        stop()


def newpages(variant: str, config_path: str, run: str, splits: list[str], inputs: list[str], chunks: int,
             keep: bool, allow_dirty: bool, collect: bool) -> None:
    """Every chain's decode of the pages of `splits` (`trogs26.decode.newpages`), written under runs/<run>/."""
    sys.path.insert(0, str(REPO))
    from trogs26 import config, data
    from trogs26.decode import newpages as decode_newpages
    from trogs26.layout import Work
    cfg, work, started = config.load(config_path), Work.from_env(), time.perf_counter()
    bases = [b for split in splits for b in data.splits()[split]]
    folds = sorted(data.folds(cfg['data']['folds']))
    tasks = [(fold, bases[i::chunks]) for fold in folds for i in range(min(chunks, len(bases)))]
    payload = {'action': 'newpages', 'cfg': dict(cfg), 'variant': variant}
    with tempfile.TemporaryDirectory() as tmp:
        archive, commit = head_archive(Path(tmp), allow_dirty)
        parts, record = run_tasks(work.root, f'{run}_{variant}_new', payload, tasks, task_weights(work, 'new', tasks),
                                  inputs, archive, commit, collect)
    chains = decode_newpages.merge(folds, parts)
    decode_newpages.write_run(work, cfg, run, variant, splits, chains, time.perf_counter() - started,
                              azure=record)
    print('decoded', len(bases), 'squeezes with', len(folds), 'chains; chain seconds',
          {c['fold']: c['seconds'] for c in chains}, flush=True)
    if not keep:
        stop()


def _one_thread() -> None:
    import torch
    torch.set_num_threads(1)


def worker(spec_path: str, parts_path: str) -> None:
    """On a VM: decode this VM's share of a run and write its parts."""
    from trogs26.decode import newpages as decode_newpages
    from trogs26.decode import sweep as decode_sweep
    from trogs26.layout import Work
    spec, work = json.loads(Path(spec_path).read_text()), Work.from_env()
    cfg, variant = spec['cfg'], spec['variant']
    with cf.ProcessPoolExecutor(PROCESSES, initializer=_one_thread) as processes:
        if spec['action'] == 'sweep':
            pending = [processes.submit(decode_sweep.sweep_pages, work, cfg, variant, fold, bases)
                       for fold, bases in spec['tasks']]
        else:
            combos = decode_newpages.variant_combos(cfg, variant)
            pending = [processes.submit(decode_newpages.decode_chain, work, cfg, variant, fold, bases, combos)
                       for fold, bases in spec['tasks']]
        parts = []
        for done in cf.as_completed(pending):
            parts.append(done.result())
            print(f'fold {parts[-1]["fold"]}: {parts[-1]["seconds"]} s', flush=True)
    partial = Path(parts_path + '.partial')
    partial.write_text(json.dumps(parts))
    partial.replace(parts_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('up', 'status', 'stop', 'down'):
        sub.add_parser(action)
    p_sweep = sub.add_parser('sweep')
    p_sweep.add_argument('--variant', required=True)
    p_sweep.add_argument('--config', required=True)
    p_sweep.add_argument('--run', required=True)
    p_sweep.add_argument('--inputs', required=True, help='comma-separated paths under TROGS26_WORK')
    p_sweep.add_argument('--folds', default='', help='comma-separated folds (default: all)')
    p_sweep.add_argument('--chunks', type=int, default=8, help='tasks per fold')
    p_sweep.add_argument('--keep', action='store_true', help='leave the pool running afterwards')
    p_sweep.add_argument('--allow-dirty', action='store_true')
    p_sweep.add_argument('--collect', action='store_true', help='only wait for and combine a sweep already sent')
    p_new = sub.add_parser('newpages')
    p_new.add_argument('--variant', required=True)
    p_new.add_argument('--config', required=True)
    p_new.add_argument('--run', required=True)
    p_new.add_argument('--splits', required=True, help='comma-separated splits of the new pages')
    p_new.add_argument('--inputs', required=True, help='comma-separated paths under TROGS26_WORK')
    p_new.add_argument('--chunks', type=int, default=4, help='tasks per chain')
    p_new.add_argument('--keep', action='store_true', help='leave the pool running afterwards')
    p_new.add_argument('--allow-dirty', action='store_true')
    p_new.add_argument('--collect', action='store_true', help='only wait for and combine a run already sent')
    p_worker = sub.add_parser('worker')
    p_worker.add_argument('spec')
    p_worker.add_argument('parts')
    args = parser.parse_args()
    if args.action == 'up':
        up()
    elif args.action == 'status':
        for region, row in sorted(pool().items()):
            print(region, row['state'], row['ip'])
    elif args.action == 'stop':
        stop()
    elif args.action == 'down':
        down()
    elif args.action == 'sweep':
        sweep(args.variant, args.config, args.run, [p for p in args.inputs.split(',') if p], args.folds,
              args.chunks, args.keep, args.allow_dirty, args.collect)
    elif args.action == 'newpages':
        newpages(args.variant, args.config, args.run, args.splits.split(','),
                 [p for p in args.inputs.split(',') if p], args.chunks, args.keep, args.allow_dirty,
                 args.collect)
    else:
        worker(args.spec, args.parts)


if __name__ == '__main__':
    main()

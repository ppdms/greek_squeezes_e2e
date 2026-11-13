"""GPU and CPU stages on Google Compute Engine: one VM per job, deleted when its job ends.

A job runs a shell command in the repository at a committed tree, on a fresh VM:
  1. `run` uploads `git archive HEAD` to gs://BUCKET/code/<commit>.tar.gz (tracked files must be unmodified, so the
     job runs exactly the commit it names), writes the job spec to gs://BUCKET/jobs/<name>/spec.json and creates
     the VM in a project without a job VM (--project pins one; none free: the job is refused): g2-standard-8 (8 vCPU,
     32 GB, one L4), Deep Learning VM base image (NVIDIA driver), 200 GB balanced disk, service account
     trogs26-runner of the VM's project, `max-run-duration` = the job's hour cap with termination action DELETE.
     GPU VMs are flex-start (Dynamic Workload Scheduler: the L4 pool that stays available when on-demand L4s are out
     of stock; same price, preemptible quota): each zone is asked for an immediate start, then, with --wait, the
     first zone queues the request for up to that many seconds (2 hours at most; `run` returns when the VM exists).
     CPU VMs (corpus, decontamination, PPM counts, decoding) are on-demand, each zone asked for an immediate start.
  2. gcp/startup.sh installs the environment (gcp/requirements-gpu.txt, Python 3.11 via uv), fetches the public
     squeeze images from the Hub dataset papadimas/trogs-greek-squeezes (and, only for jobs that ask, the TROGS-26
     images from gs://BUCKET/images/trogs26/), copies the job's inputs from gs://BUCKET/work/ into /work and runs the
     command with TROGS26_WORK=/work, uploading its log every minute.
  3. It uploads the job's outputs to gs://BUCKET/work/, writes status.json next to the spec and deletes its VM.

Projects: each holds a quota of one GPU in all regions, so as many jobs run at once as there are projects, one per
project. All bill the same account, read and write the same bucket (in the first project) and share the ledger and
its limit.

Machines (--machine; Cloud Billing catalog in EUR, 2026-11-07):
  g2-standard-8   one L4 24 GB, 8 vCPU, 32 GiB: 0.767/h in the Americas regions (L4 0.50297, vCPU 0.022442, GiB
                  0.002629; flex-start and on-demand G2 prices are equal), 0.763/h in us-east4, 0.806/h in
                  europe-west4; zones across those regions, cheapest first.
  a2-highgpu-1g   one A100 40 GB, 12 vCPU, 85 GiB: 2.113/h in the Americas regions, 2.180/h in europe-west4
                  (flex-start A100 1.44922; A2 vCPU and GiB at their on-demand prices, an upper bound); the regions
                  where the project holds A100 quota (us-central1, us-east1, us-west1, europe-west4).
  c2d-standard-8  no GPU, 8 vCPU (AMD Milan), 32 GiB: 0.326/h in the Americas regions (vCPU 0.026551, GiB 0.003556;
                  catalog 2026-11-21); a project's CPU quota is 12 vCPU in all regions (24 in the first).
The ledger, gs://BUCKET/ledger.jsonl, holds a row per VM created; `status` sums every job's VM time at its zone's
price (a finished job: its startup script's start to its end plus BOOT_HOURS; a running one: from the request on).
The owner's limit is 270 EUR of total GCP spend, everything so far included (budget alert at 200 EUR); a job whose
hour cap could take the ledger past LIMIT_EUR is refused.

    python launchers/gcp.py push chains/f0/oof/det ...     # local TROGS26_WORK paths -> gs://BUCKET/work/
    python launchers/gcp.py run NAME --cmd '...' [--inputs a,b] [--outputs c,d] [--hours 4] [--wait 7200]
                                     [--machine a2-highgpu-1g] [--project trogs26-gpu2] [--no-public-images]
                                     [--trogs26-images]
    python launchers/gcp.py status [NAME]
    python launchers/gcp.py pull models/x ...              # gs://BUCKET/work/ -> local TROGS26_WORK
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = 'vivid-grove-513921-g6'  # the bucket's project; storage commands name it
# Where job VMs may run, one GPU each (gcp/add_project.sh); the billing account takes five projects.
PROJECTS = (PROJECT, 'trogs26-gpu2', 'trogs26-gpu3', 'trogs26-gpu4', 'trogs26-gpu10')
BUCKET = 'trogs26-541485613110'
# machine -> (vCPU, GiB, zones tried in order, region -> (GPU, vCPU, GiB RAM) EUR per hour; '' = any other region)
MACHINES = {
    'g2-standard-8': (8, 32, ('us-central1-a', 'us-central1-b', 'us-central1-c', 'us-east1-b', 'us-east1-c',
                              'us-east1-d', 'us-west1-a', 'us-west1-b', 'us-west1-c', 'us-east4-a', 'us-east4-c',
                              'europe-west4-a', 'europe-west4-b', 'europe-west4-c'),
                      {'': (0.502972, 0.022442, 0.002629), 'us-east4': (0.501325, 0.022368, 0.002621),
                       'europe-west4': (0.528649, 0.023588, 0.002763)}),
    'a2-highgpu-1g': (12, 85, ('us-central1-a', 'us-central1-b', 'us-central1-c', 'us-central1-f', 'us-east1-b',
                               'us-west1-b', 'europe-west4-a', 'europe-west4-b'),
                      {'': (1.449218, 0.028390, 0.003805), 'europe-west4': (1.449218, 0.031256, 0.004189)}),
    'c2d-standard-8': (8, 32, ('us-central1-a', 'us-central1-b', 'us-central1-c', 'us-central1-f', 'us-east1-b',
                               'us-east1-c', 'us-east1-d', 'us-west1-a', 'us-west1-b', 'us-west1-c'),
                       {'': (0.0, 0.026551, 0.003556)}),
}


def eur_per_hour(machine: str, zone: str) -> float:
    cpus, ram, _, prices = MACHINES[machine]
    gpu, cpu, gib = prices.get(zone.rsplit('-', 1)[0], prices[''])
    return gpu + cpus * cpu + ram * gib


def has_gpu(machine: str) -> bool:
    return MACHINES[machine][3][''][0] > 0


def dearest(machine: str) -> float:
    """The machine's price in its dearest zone (the cap check assumes it)."""
    return max(eur_per_hour(machine, z) for z in MACHINES[machine][2])


IMAGE_FAMILY = 'common-cu129-ubuntu-2404-nvidia-580'
IMAGE_PROJECT = 'deeplearning-platform-release'
LIMIT_EUR = 255.0  # the owner's 270 EUR, less a margin for disks, storage and billing delay
BOOT_HOURS = 0.05  # VM time before the startup script runs, added to every finished job
LEDGER = 'ledger.jsonl'  # in the bucket


def gcloud(*args: str, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(['gcloud', *args, f'--project={PROJECT}'], text=True, check=check,
                            capture_output=capture)
    return result.stdout if capture else ''


def gs(path: str) -> str:
    return f'gs://{BUCKET}/{path}'


def _storage(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(['gcloud', 'storage', *args, f'--project={PROJECT}'], text=True, capture_output=True,
                          input=stdin)


def _missing(result: subprocess.CompletedProcess) -> bool:
    return result.returncode != 0 and ('matched no objects' in result.stderr or 'not found' in result.stderr.lower())


def _failed(what: str, result: subprocess.CompletedProcess) -> RuntimeError:
    return RuntimeError(f'{what}: {result.stderr.strip()[-500:]}')


def read_text(*paths: str) -> str | None:
    """The bucket objects `paths` (wildcards allowed) that exist, concatenated; None if none does."""
    result = _storage('cat', *map(gs, paths))
    if _missing(result):
        return result.stdout or None
    if result.returncode:
        raise _failed(f'reading {" ".join(paths)}', result)
    return result.stdout


def read_objects(*paths: str) -> list[dict]:
    """Every JSON object of the bucket objects `paths` (wildcards allowed), in one request."""
    text, decoder, objects, i = read_text(*paths) or '', json.JSONDecoder(), [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        obj, i = decoder.raw_decode(text, i)
        objects.append(obj)
    return objects


def read_versioned(path: str) -> tuple[str | None, int]:
    """The object `path` and its generation (None and 0 if it does not exist)."""
    while True:
        result = _storage('objects', 'describe', gs(path), '--format=value(generation)')
        if _missing(result):
            return None, 0
        if result.returncode:
            raise _failed(f'describing {path}', result)
        generation = int(result.stdout.strip())
        result = _storage('cat', f'{gs(path)}#{generation}')
        if result.returncode == 0:
            return result.stdout, generation
        if not _missing(result):  # (missing: replaced between the two requests, so read it again)
            raise _failed(f'reading {path}', result)


def write_text(path: str, text: str, generation: int | None = None) -> bool:
    """Write the object `path`; with `generation`, only if it is still at that generation (0: does not exist), and
    return whether it was written."""
    precondition = [] if generation is None else [f'--if-generation-match={generation}']
    result = _storage('cp', *precondition, '-', gs(path), stdin=text)
    if result.returncode and generation is not None and (
            '412' in result.stderr or 'precondition' in result.stderr.lower()):
        return False
    if result.returncode:
        raise _failed(f'writing {path}', result)
    return True


def listing(path: str) -> list[str]:
    """The URLs `gcloud storage ls` gives for `path` (wildcards allowed; a directory's entries)."""
    result = _storage('ls', gs(path))
    if _missing(result):
        return []
    if result.returncode:
        raise _failed(f'listing {path}', result)
    return result.stdout.split()


def download(path: str, local: Path) -> None:
    """The bucket's file or directory `path` to `local`."""
    entries = listing(path)
    if not entries:
        raise FileNotFoundError(gs(path))
    if entries == [gs(path)]:
        local.parent.mkdir(parents=True, exist_ok=True)
        result = _storage('cp', gs(path), str(local))
    else:
        local.mkdir(parents=True, exist_ok=True)
        result = _storage('rsync', '-r', gs(path), str(local))
    if result.returncode:
        raise _failed(f'downloading {path}', result)


def upload(local: Path, path: str) -> None:
    """The file or directory `local` to the bucket's `path`, with what file symlinks point to (symlinks to
    directories inside `local` are not followed)."""
    local = local.resolve()
    result = _storage(*(('rsync', '-r', '--no-ignore-symlinks') if local.is_dir() else ('cp',)), str(local), gs(path))
    if result.returncode:
        raise _failed(f'uploading {local}', result)


def work_root() -> Path:
    return Path(os.environ['TROGS26_WORK']).resolve()


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=REPO, text=True, check=True, capture_output=True).stdout.strip()


def upload_code(allow_dirty: bool) -> str:
    """The commit at HEAD, archived to the bucket once. Tracked files must match it, unless `allow_dirty` (the job
    runs HEAD either way; the flag only acknowledges uncommitted edits it will not see)."""
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty and not allow_dirty:
        sys.exit('tracked files differ from HEAD: commit before running a job (or --allow-dirty to run HEAD)')
    if dirty:
        print('running HEAD; not in it:', ' '.join(line.split()[-1] for line in dirty.splitlines()))
    commit = git('rev-parse', 'HEAD')
    target = gs(f'code/{commit}.tar.gz')
    exists = subprocess.run(['gcloud', 'storage', 'ls', target, f'--project={PROJECT}'], capture_output=True)
    if exists.returncode != 0:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'code.tar.gz'
            subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(archive), 'HEAD'], cwd=REPO, check=True)
            gcloud('storage', 'cp', str(archive), target)
    return commit


def ledger_rows() -> list[dict]:
    return [json.loads(line) for line in (read_text(LEDGER) or '').splitlines() if line]


def append_ledger(row: dict) -> None:
    """Add a row to the ledger: read it, append, and write it back unless another writer came first (then again)."""
    for _ in range(20):
        text, generation = read_versioned(LEDGER)
        if write_text(LEDGER, (text or '') + json.dumps(row) + '\n', generation):
            return
    raise RuntimeError('the ledger kept changing while a row was added')


def job_statuses() -> dict[str, dict]:
    """The status.json (gcp/startup.sh writes it) of every job that has ended, by job name (a listing of every job's
    files, about a minute)."""
    return {status['job']: status for status in read_objects('jobs/*/status.json')}


def _hours(row: dict, status: dict | None) -> tuple[float, bool]:
    """VM hours of a ledger row: its startup script's start to its end plus BOOT_HOURS if it ended (`status`), else
    launch to now (it may still be running, or waiting for capacity)."""
    if status:
        start = dt.datetime.fromisoformat(status['start'].replace('Z', '+00:00'))
        end = dt.datetime.fromisoformat(status['end'].replace('Z', '+00:00'))
        return (end - start).total_seconds() / 3600 + BOOT_HOURS, True
    launched = dt.datetime.fromisoformat(row['launched'])
    return (dt.datetime.now(dt.timezone.utc) - launched).total_seconds() / 3600, False


def spent(rows: list[dict], statuses: dict[str, dict]) -> tuple[float, float]:
    """(EUR spent by the finished jobs of ledger `rows`, EUR that the running ones can still reach at their caps)."""
    done = open_cap = 0.0
    for row in rows:
        hours, finished = _hours(row, statuses.get(row['name']))
        if finished:
            done += hours * row['eur_per_hour']
        else:
            open_cap += max(hours, row['hours_cap']) * row['eur_per_hour']
    return done, open_cap


class Refused(Exception):
    """A job the budget cannot take."""


def check_budget(hours: float, machine: str) -> float:
    """The worst case with a new job of `hours` on `machine`; refused past LIMIT_EUR."""
    done, open_cap = spent(ledger_rows(), job_statuses())
    cap = hours * dearest(machine)
    worst = done + open_cap + cap
    if worst > LIMIT_EUR:
        raise Refused(f'spent {done:.2f} + running caps {open_cap:.2f} + this cap {cap:.2f} > {LIMIT_EUR} EUR')
    return worst


def _create(vm: str, machine: str, zone: str, name: str, hours: float, wait: int, project: str
            ) -> subprocess.CompletedProcess:
    provisioning = (['--provisioning-model=FLEX_START', f'--request-valid-for-duration={wait}s'] if has_gpu(machine)
                    else ['--provisioning-model=STANDARD'])
    return subprocess.run(
        ['gcloud', 'compute', 'instances', 'create', vm, f'--project={project}', f'--zone={zone}',
         f'--machine-type={machine}', f'--image-family={IMAGE_FAMILY}', f'--image-project={IMAGE_PROJECT}',
         '--boot-disk-size=200GB', '--boot-disk-type=pd-balanced', '--maintenance-policy=TERMINATE',
         f'--service-account=trogs26-runner@{project}.iam.gserviceaccount.com', '--scopes=cloud-platform',
         *provisioning, f'--max-run-duration={int(hours * 3600)}s',
         '--instance-termination-action=DELETE', f'--metadata=job={name},bucket={BUCKET}',
         f'--metadata-from-file=startup-script={REPO}/gcp/startup.sh', '--labels=trogs26=job'],
        capture_output=True, text=True)


def vms(project: str) -> list[str]:
    """The project's job VMs (running, starting or being deleted): name, zone, status and creation time of each."""
    out = subprocess.run(['gcloud', 'compute', 'instances', 'list', f'--project={project}', '--filter=labels.trogs26=job',
                          '--format=value(name,zone.basename(),status,creationTimestamp)'],
                         text=True, check=True, capture_output=True).stdout
    return out.strip().splitlines()


def create(name: str, job: dict, project: str) -> dict:
    """Write a job's spec and create its VM in `project` (each zone asked for an immediate start, then the first
    queued for the job's `wait` seconds); append its ledger row and return it."""
    spec = {'name': name, **{key: job[key] for key in ('commit', 'cmd', 'inputs', 'outputs', 'public_images',
                                                       'trogs26_images')}}
    write_text(f'jobs/{name}/spec.json', json.dumps(spec, indent=1))
    vm = f'trogs26-{name}'.lower().replace('_', '-')[:62]
    machine, zones = job['machine'], MACHINES[job['machine']][2]
    queued = job['wait'] and has_gpu(machine)
    attempts = [(zone, 0) for zone in zones] + ([(zones[0], min(job['wait'], 7200))] if queued else [])
    last = ''
    for zone, seconds in attempts:
        created = _create(vm, machine, zone, name, job['hours'], seconds, project)
        if created.returncode == 0:
            row = {'name': name, 'vm': vm, 'project': project, 'zone': zone, 'machine': machine,
                   'commit': job['commit'], 'launched': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                   'hours_cap': job['hours'], 'eur_per_hour': round(eur_per_hour(machine, zone), 4)}
            append_ledger(row)
            return row
        last = f'{zone}: ' + (created.stderr.strip().splitlines()[-1] if created.stderr.strip() else
                              f'exit status {created.returncode}')
    raise RuntimeError(f'no zone could create the VM (last {last})')


def free_project(project: str) -> str:
    """`project`, or with 'auto' the first project, that has no job VM; exit if there is none."""
    for candidate in PROJECTS if project == 'auto' else (project,):
        if not vms(candidate):
            return candidate
    sys.exit(f'refused: {"every project has" if project == "auto" else project + " has"} a job VM (status lists them)')


def run(name: str, cmd: str, inputs: list[str], outputs: list[str], hours: float, wait: int, machine: str,
        public_images: bool, trogs26_images: bool, allow_dirty: bool = False, project: str = 'auto') -> None:
    """Create a job's VM in `project` (auto: the first without a job VM); refused if the name is in the ledger, the
    budget cannot take the job or no project is free."""
    if any(row['name'] == name for row in ledger_rows()):
        sys.exit(f'job {name} exists in the ledger')
    try:
        worst = check_budget(hours, machine)
    except Refused as refused:
        sys.exit(f'refused: {refused}')
    project = free_project(project)
    commit = upload_code(allow_dirty)
    try:
        row = create(name, {'commit': commit, 'cmd': cmd, 'inputs': inputs, 'outputs': outputs,
                            'public_images': public_images, 'trogs26_images': trogs26_images, 'hours': hours,
                            'wait': wait, 'machine': machine}, project)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"{name}: {row['project']} {row['zone']}; worst case with it {worst:.2f} EUR of {LIMIT_EUR}")


def status(name: str | None) -> None:
    for project in PROJECTS:
        print(f'VMs in {project}:', '; '.join(vms(project)) or 'none')
    rows, statuses = ledger_rows(), job_statuses()
    for row in rows:
        if name and row['name'] != name:
            continue
        hours, finished = _hours(row, statuses.get(row['name']))
        print(f"{row['name']:32s} {'done' if finished else 'open'} {hours:6.2f} h {hours * row['eur_per_hour']:7.2f} EUR",
              statuses.get(row['name']) or '')
    done, open_cap = spent(rows, statuses)
    print(f'finished jobs {done:.2f} EUR; running jobs at most {open_cap:.2f} EUR more; limit {LIMIT_EUR}')


def push(paths: list[str]) -> None:
    root = work_root()
    for p in paths:
        upload(root / p, f'work/{p}')


def pull(paths: list[str]) -> None:
    root = work_root()
    for p in paths:
        download(f'work/{p}', root / p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='action', required=True)
    p_run = sub.add_parser('run')
    p_run.add_argument('name')
    p_run.add_argument('--cmd', required=True)
    p_run.add_argument('--inputs', default='')
    p_run.add_argument('--outputs', default='')
    p_run.add_argument('--hours', type=float, default=4.0)
    p_run.add_argument('--wait', type=int, default=7200, help='seconds the first zone may queue the request (<= 7200)')
    p_run.add_argument('--machine', choices=sorted(MACHINES), default='g2-standard-8')
    p_run.add_argument('--project', choices=('auto', *PROJECTS), default='auto',
                       help='the project whose GPU the job uses (auto: the first without a job VM)')
    p_run.add_argument('--no-public-images', action='store_true')
    p_run.add_argument('--trogs26-images', action='store_true')
    p_run.add_argument('--allow-dirty', action='store_true', help='run HEAD although tracked files are modified')
    p_status = sub.add_parser('status')
    p_status.add_argument('name', nargs='?')
    for action in ('push', 'pull'):
        sub.add_parser(action).add_argument('paths', nargs='+')
    args = parser.parse_args()
    if args.action == 'run':
        split = lambda s: [x for x in s.split(',') if x]  # noqa: E731
        run(args.name, args.cmd, split(args.inputs), split(args.outputs), args.hours, args.wait, args.machine,
            not args.no_public_images, args.trogs26_images, args.allow_dirty, args.project)
    elif args.action == 'status':
        status(args.name)
    elif args.action == 'push':
        push(args.paths)
    else:
        pull(args.paths)


if __name__ == '__main__':
    main()

"""swap_to_t7.py N [--do]: option A for round train_vN. Replace symlink-free subtrees of ~/Desktop/RallyIQ/train_vN (except
repo/) by symlinks to their verified copies in /Volumes/T7/RallyIQ_training/archive/train_vN. Subtrees that contain symlinks
are kept as directories on the Desktop and handled one level down, so every original symlink stays where it was.
Each replaced path is first checked (file count + size + sha1 of every file vs the T7 copy), renamed to <path>.pre-move,
then symlinked; the .pre-move copy is deleted only after the symlinked path reads identically. Every action is appended to
archive/MOVE_LOG.md (reverse: rm the symlink, mv the T7 copy back or restore from .pre-move if present).
Without --do it only prints the plan. Skips paths opened by any process (lsof)."""
import os, sys, hashlib, shutil, subprocess, time
n = sys.argv[1]; DO = '--do' in sys.argv
S = os.path.expanduser(f'~/Desktop/RallyIQ/train_v{n}'); A = f'/Volumes/T7/RallyIQ_training/archive/train_v{n}'
LOG = '/Volumes/T7/RallyIQ_training/archive/MOVE_LOG.md'
def has_link(p):
    if os.path.islink(p): return True
    if os.path.isdir(p):
        for r, ds, fs in os.walk(p):
            if any(os.path.islink(os.path.join(r, x)) for x in ds + fs): return True
    return False
def files(p):
    if os.path.isfile(p): return {'': p}
    out = {}
    for r, ds, fs in os.walk(p):
        for f in fs:
            if f.startswith('._'): continue
            q = os.path.join(r, f); out[os.path.relpath(q, p)] = q
    return out
def sha(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def same(a, b):
    fa, fb = files(a), files(b)
    if set(fa) != set(fb): return False, f'file sets differ ({len(fa)} vs {len(fb)})'
    for k in fa:
        if os.path.getsize(fa[k]) != os.path.getsize(fb[k]) or sha(fa[k]) != sha(fb[k]): return False, f'content differs: {k}'
    return True, f'{len(fa)} files identical'
def in_use(p):
    r = subprocess.run(['lsof', '+D', p] if os.path.isdir(p) else ['lsof', p], capture_output=True, text=True)
    return bool(r.stdout.strip())
def size(p): return sum(os.path.getsize(q) for q in files(p).values())
freed = 0
def handle(rel):
    global freed
    src, dst = os.path.join(S, rel), os.path.join(A, rel)
    if os.path.islink(src): return
    if has_link(src):
        if os.path.isdir(src):
            for x in sorted(os.listdir(src)): handle(os.path.join(rel, x))
        return
    if rel.split('/')[0] in ('scripts',) or size(src) < 1e6: return  # code and small text stay on the Desktop
    if not os.path.exists(dst): print('NOT IN ARCHIVE', rel); return
    ok, why = same(src, dst)
    if not ok: print('SKIP (verify failed)', rel, why); return
    if in_use(src): print('SKIP (in use)', rel); return
    b = size(src); print(('SWAP ' if DO else 'plan ') + rel, why, f'{b / 1e6:.1f} MB')
    if not DO: freed += b; return
    pre = src + '.pre-move'; os.rename(src, pre); os.symlink(dst, src)
    ok2, why2 = same(pre, src)
    with open(LOG, 'a') as f:
        f.write(f'- {time.strftime("%Y-%m-%d %H:%M:%S")} {src} -> {dst} ({why}; after swap: {why2})\n')
    if ok2:
        shutil.rmtree(pre) if os.path.isdir(pre) else os.remove(pre); freed += b
    else:
        os.remove(src); os.rename(pre, src); print('ROLLED BACK', rel, why2)
for x in sorted(os.listdir(S)):
    if x == 'repo' or x.endswith('.pre-move'): continue
    handle(x)
print(f'train_v{n}: {"freed" if DO else "would free"} {freed / 1e9:.2f} GB')

"""Preregistered V21 C post-hoc diagnosis and source-D prospective acquisition.

Independent oracle verification is a separate process (two_slot_verify.py).
No source-D function name appears in the acquisition method.
"""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

from research.prospective_native_v10.run import scan, score
from research.semantic_v21.source_c_trial import CASES, baseline_score
from research.semantic_v21.two_slot_binding import candidate_bindings, call_bounded


SOURCES = {
    'c_posthoc': {'name': 'pytoolz/toolz', 'sha': '451af60dec590a6010e2babdbf391ea8f815122f',
                  'root': Path('/tmp/toolz-v21-c')},
    'd_prospective': {'name': 'Suor/funcy', 'sha': 'a876c7e1cf11e7c746e91fce4632ebfa93865428',
                      'root': Path('/tmp/funcy-v21-d')},
}


def assert_pinned(root, sha):
    current = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if current != sha:
        raise ValueError('UPSTREAM_PIN_MISMATCH')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)


def resolve(root, record):
    sys.path.insert(0, str(root))
    module = '.'.join(Path(record['file']).with_suffix('').parts)
    return getattr(importlib.import_module(module), record['name'])


def evaluate(root, records):
    matches = []
    attempts = 0
    first_hit_attempt = None
    began = time.perf_counter()
    for record in records:
        try:
            fn = resolve(root, record)
            variants = candidate_bindings(fn)
        except Exception:
            continue
        for binding in variants:
            attempts += 1
            try:
                hit = True
                for xs in CASES:
                    result = call_bounded(fn, binding, xs)
                    expected = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
                    if type(result) is not list or result != expected:
                        hit = False
                        break
            except Exception:
                continue
            if hit:
                if first_hit_attempt is None:
                    first_hit_attempt = attempts
                matches.append({'file': record['file'], 'name': record['name'], 'binding': binding})
    return {'candidates': matches, 'attempts': attempts,
            'first_hit_attempt': first_hit_attempt,
            'elapsed_seconds': time.perf_counter() - began}


def discover(out):
    out.mkdir(exist_ok=True)
    report = {'protocol': 'HELDOUT_SOURCE_D_TWO_SLOT_PROTOCOL.md',
              'discovery_status': 'PENDING_INDEPENDENT_VERIFICATION', 'sources': {}}
    for key, source in SOURCES.items():
        root = source['root']
        assert_pinned(root, source['sha'])
        universe = sorted(scan(root)[:400], key=lambda c: (-score(c), c['file'], c['name']))[:160]
        generic = evaluate(root, universe)
        baseline = evaluate(root, sorted(universe, key=lambda c:(-baseline_score(c), c['file'], c['name'])))
        report['sources'][key] = {'repository': source['name'], 'sha': source['sha'],
                                  'scanned_functions': len(universe),
                                  'source_generic': generic,
                                  'equal_information_baseline': baseline}
        print(json.dumps({'source': key, 'scanned': len(universe),
                          'generic_hits': len(generic['candidates']),
                          'baseline_hits': len(baseline['candidates']),
                          'generic_attempts': generic['attempts'],
                          'baseline_attempts': baseline['attempts']}, indent=2))
    (out/'two_slot_discovery.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('v21_out'))
    a = p.parse_args()
    discover(a.out)

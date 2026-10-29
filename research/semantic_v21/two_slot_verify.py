"""Independent finite oracle for V21 source-D prospective two-slot bindings.

This file is imported only by the verification CI step in a second Python
process, after all discovery candidates have been written to disk.
"""
import importlib
import itertools
import json
from pathlib import Path
import sys

from research.prospective_native_v11.adapter import adaptations
from research.semantic_v21.two_slot_trial import SOURCES, assert_pinned


def verify_candidate(root, record, binding):
    sys.path.insert(0, str(root))
    name = '.'.join(Path(record['file']).with_suffix('').parts)
    try:
        fn = getattr(importlib.import_module(name), record['name'])
        count = 0
        for n in range(7):
            for t in itertools.product((0, 1, 2), repeat=n):
                xs = list(t)
                if binding['input_slot'] == 0:
                    result = fn(xs, **binding['kwargs'])
                elif binding['input_slot'] == 1 and binding['size_value'] == 2:
                    result = fn(2, xs, **binding['kwargs'])
                else:
                    return {'pass': False, 'checked': count, 'reason': 'UNKNOWN_BINDING'}
                if binding['materialize']:
                    result = list(itertools.islice(result, max(1, len(xs))))
                expected = list(zip(xs, xs[1:]))
                if type(result) is not list or result != expected:
                    return {'pass': False, 'checked': count, 'failure': xs}
                count += 1
        return {'pass': True, 'checked': count}
    except Exception as e:
        return {'pass': False, 'error': type(e).__name__+':'+str(e)[:240]}


def posthoc_c_mechanism(root, qualified_candidates):
    """Explicit C diagnosis: looked at source; cannot count as blind discovery."""
    sys.path.insert(0, str(root))
    from toolz.itertoolz import sliding_window, partition
    old = adaptations(sliding_window)
    expected = lambda xs: list(zip(xs, xs[1:]))
    total = 0
    wrong_size_rejected = False
    partition_rejected = False
    exact_replay = True
    for n in range(7):
        for t in itertools.product((0, 1, 2), repeat=n):
            xs = list(t)
            if list(itertools.islice(sliding_window(2, xs), max(1, len(xs)))) != expected(xs):
                exact_replay = False
            if list(itertools.islice(sliding_window(3, xs), max(1, len(xs)))) != expected(xs):
                wrong_size_rejected = True
            if list(itertools.islice(partition(2, xs), max(1, len(xs)))) != expected(xs):
                partition_rejected = True
            total += 1
    discovered = any(c['candidate']['name'] == 'sliding_window' and
                     c['candidate']['file'] == 'toolz/itertoolz.py'
                     for c in qualified_candidates)
    return {'status': 'WARRANTED_FINITE_POSTHOC_CAUSAL_DIAGNOSIS' if
            not old and exact_replay and discovered and wrong_size_rejected and partition_rejected
            else 'UNKNOWN_DIAGNOSTIC', 'old_adapter_variants': len(old),
            'direct_signature_correct_qualified': exact_replay,
            'generic_two_slot_found_sliding_window': discovered,
            'wrong_size_rejected': wrong_size_rejected,
            'partition_rejected': partition_rejected, 'oracle_cases': total,
            'prospective': False}


def posthoc_d_legacy_ablation(root, qualified_candidates):
    """After the frozen D result, check if D required the added binding rule."""
    old_verified = []
    for item in qualified_candidates:
        candidate = item['candidate']
        module = '.'.join(Path(candidate['file']).with_suffix('').parts)
        sys.path.insert(0, str(root))
        fn = getattr(importlib.import_module(module), candidate['name'])
        for kwargs, materialize in adaptations(fn):
            binding = {'input_slot': 0, 'size_value': None,
                       'kwargs': kwargs, 'materialize': materialize}
            replay = verify_candidate(root, candidate, binding)
            if replay.get('pass') and replay.get('checked') == 1093:
                old_verified.append({'file': candidate['file'],
                                     'name': candidate['name'], 'binding': binding})
    new_names = {(i['candidate']['file'], i['candidate']['name']) for i in qualified_candidates}
    old_names = {(i['file'], i['name']) for i in old_verified}
    return {'status': 'POSTHOC_DIAGNOSTIC_ONLY', 'old_verified': old_verified,
            'old_sufficient_for_all_new_hits': new_names <= old_names,
            'new_second_slot_needed_for_d': bool(new_names - old_names),
            'protected_cases': 1093}


def verify(out):
    acquired = json.loads((out/'two_slot_discovery.json').read_text())
    result = {'protocol': acquired['protocol'], 'protected_cases': 1093,
              'source_c_is_posthoc': True, 'source_d_is_prospective': True,
              'unique_mathgraph_advantage': False, 'sources': {}}
    for key, source in SOURCES.items():
        assert_pinned(source['root'], source['sha'])
        observed = acquired['sources'][key]
        if observed['sha'] != source['sha']:
            raise ValueError('FROZEN_SOURCE_IDENTITY_MISMATCH')
        pairs = {}
        for method in ('source_generic', 'equal_information_baseline'):
            checks = []
            for candidate in observed[method]['candidates']:
                verdict = verify_candidate(source['root'], candidate, candidate['binding'])
                checks.append({'candidate': candidate, 'check': verdict})
            qualified = [c for c in checks if c['check'].get('pass') and c['check'].get('checked') == 1093]
            pairs[method] = {'screened': len(checks), 'qualified': qualified,
                             'first_hit_attempt': observed[method]['first_hit_attempt'],
                             'attempts': observed[method]['attempts']}
        if pairs['source_generic']['qualified']:
            status = 'WARRANTED_BOUNDED_SOURCE_'+('C_POSTHOC' if key == 'c_posthoc' else 'D')
        elif any(x['screened'] for x in pairs.values()):
            status = 'REJECTED_CANDIDATE'
        else:
            status = 'UNKNOWN_BUDGET_OR_BINDING'
        result['sources'][key] = {'status': status, 'methods': pairs,
                                  'repository': source['name'], 'sha': source['sha']}
        print(json.dumps({'source': key, 'status': status,
                          'qualified_generic': len(pairs['source_generic']['qualified']),
                          'qualified_baseline': len(pairs['equal_information_baseline']['qualified']),
                          'protected_cases': 1093}, indent=2))
    result['posthoc_c_mechanism'] = posthoc_c_mechanism(
        SOURCES['c_posthoc']['root'], result['sources']['c_posthoc']['methods']['source_generic']['qualified'])
    print(json.dumps(result['posthoc_c_mechanism'], indent=2))
    for key, entry in result['sources'].items():
        print(json.dumps({'verified_candidate_identities': key,
                          'items': [{'file': item['candidate']['file'],
                                     'name': item['candidate']['name'],
                                     'binding': item['candidate']['binding']}
                                    for item in entry['methods']['source_generic']['qualified']]}, indent=2))
    result['posthoc_d_legacy_ablation'] = posthoc_d_legacy_ablation(
        SOURCES['d_prospective']['root'],
        result['sources']['d_prospective']['methods']['source_generic']['qualified'])
    print(json.dumps(result['posthoc_d_legacy_ablation'], indent=2))
    (out/'two_slot_evidence.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('v21_out'))
    verify(p.parse_args().out)

"""V21 regression: a nonterminating iterator must not stall semantic search.

Both tests run the *real* discovery/verification functions in a subprocess.
The subprocess timeout protects CI while making the original list(generator)
bug observably RED. This finite fixture is not the source-native V21 result.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = '''
from itertools import repeat
import time

def adjacent(xs):
    """Return the list of contiguous adjacent pairs."""
    return [(xs[i], xs[i+1]) for i in range(len(xs)-1)]

def repeat_last(xs):
    """Continue indefinitely after iterable exhaustion."""
    yield from xs
    while True:
        time.sleep(0.02)
        yield None
'''


class BoundedMaterialization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v21-bounded-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'fixture.py').write_text(FIXTURE)

    def run_real(self, script):
        # The timeout is a test guard, not an algorithmic search-time budget.
        proc = subprocess.run([sys.executable, '-c', script, str(self.root)],
                              cwd=ROOT, text=True, capture_output=True,
                              timeout=3, check=True)
        return json.loads(proc.stdout.strip())

    def test_discovery_rejects_unbounded_materialization_and_retains_finite_match(self):
        result = self.run_real('''
import json,sys
from pathlib import Path
from research.semantic_v21.discover import fingerprints
candidates,attempts,scanned=fingerprints(Path(sys.argv[1]))
print(json.dumps({'names':[c['name'] for c in candidates],
                  'attempts':attempts,'scanned':scanned}))
''')
        self.assertEqual(result['scanned'], 2)
        # The legitimate list-producing function qualifies with either
        # materialization setting; neither variant of repeat_last may qualify.
        self.assertEqual(result['names'], ['adjacent', 'adjacent'])
        self.assertEqual(result['attempts'], 4)

    def test_independent_verifier_rejects_nonterminating_materialization(self):
        result = self.run_real('''
import json,sys
from pathlib import Path
from research.semantic_v21.verify import check
root=Path(sys.argv[1])
bad=check(root,{'file':'fixture.py','name':'repeat_last',
                'kwargs':{},'materialize':True})
good=check(root,{'file':'fixture.py','name':'adjacent',
                 'kwargs':{},'materialize':False})
print(json.dumps({'bad':bad,'good':good}))
''')
        self.assertEqual(result['bad'], {'pass':False, 'checked':0})
        self.assertEqual(result['good'], {'pass':True, 'checked':1093})


if __name__ == '__main__':
    unittest.main()

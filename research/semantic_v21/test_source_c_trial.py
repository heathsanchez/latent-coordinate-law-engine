"""Source-C acquisition mechanics, independent of the chosen held-out source."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SourceCMechanics(unittest.TestCase):
    def test_semantic_search_rejects_infinite_output_without_losing_finite_match(self):
        with tempfile.TemporaryDirectory(prefix='source-c-contract-fixture-') as folder:
            root=Path(folder)
            (root/'trialfixture.py').write_text('''
import time
def adjacent(xs):
    """List of adjacent pairs."""
    return [(xs[i],xs[i+1]) for i in range(len(xs)-1)]
def infinite(xs):
    """Infinite iterable, not adjacent pairs."""
    yield from xs
    while True:
        time.sleep(0.02)
        yield None
''')
            script='''
import json,sys
from pathlib import Path
from research.semantic_v21.source_c_trial import evaluate
records=[{'file':'trialfixture.py','name':'adjacent'},
         {'file':'trialfixture.py','name':'infinite'}]
print(json.dumps(evaluate(Path(sys.argv[1]),records)))
'''
            process=subprocess.run([sys.executable,'-c',script,str(root)],
                                   cwd=Path(__file__).resolve().parents[2],
                                   capture_output=True,text=True,check=True,timeout=3)
            result=json.loads(process.stdout)
            self.assertEqual(result['attempts'],4)
            self.assertEqual(result['first_hit_attempt'],1)
            self.assertEqual([c['name'] for c in result['candidates']],
                             ['adjacent','adjacent'])


if __name__=='__main__':unittest.main()

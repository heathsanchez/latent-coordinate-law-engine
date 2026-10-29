"""Source-independent controls for the preregistered two-slot grammar."""
import itertools
import unittest

from research.prospective_native_v11.adapter import adaptations
from research.semantic_v21.two_slot_binding import candidate_bindings, call_bounded


def size_first(n, seq):
    return (tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1)))


def iterable_first(seq, n=2):
    return (tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1)))


def partition(n, seq):
    return (tuple(seq[i:i + n]) for i in range(0, len(seq), n))


def infinite_tail(n, seq):
    yield from itertools.repeat((0, 0))


class TwoSlotBindingTest(unittest.TestCase):
    def test_legacy_ablation_and_new_size_first_transport(self):
        self.assertEqual(adaptations(size_first), [])
        variants = candidate_bindings(size_first)
        self.assertEqual({v['input_slot'] for v in variants}, {1})
        self.assertTrue(any(v['materialize'] for v in variants))
        for variant in variants:
            if variant['materialize']:
                self.assertEqual(call_bounded(size_first, variant, [0, 1, 2]), [(0, 1), (1, 2)])

    def test_existing_iterable_first_and_nonsensical_partition(self):
        self.assertEqual({v['input_slot'] for v in candidate_bindings(iterable_first)}, {0})
        matched = [v for v in candidate_bindings(partition)
                   if v['materialize'] and
                   call_bounded(partition, v, [0, 1, 2]) == [(0, 1), (1, 2)]]
        self.assertEqual(matched, [])

    def test_reject_extra_required_parameters_and_bound_infinite_iterables(self):
        def unknown(pattern, stream, flag):
            return []
        self.assertEqual(candidate_bindings(unknown), [])
        variant = next(v for v in candidate_bindings(infinite_tail) if v['materialize'])
        self.assertEqual(len(call_bounded(infinite_tail, variant, [])), 1)


if __name__ == '__main__':
    unittest.main()

# coding=utf-8
"""FeatureDataLoader: queue + store -> TrainBatch, with injected transform/collate (CPU)."""

import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from functools import partial
from unittest import mock

import torch

from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)

def _simple_collate(features):
    keys = features[0].keys()
    return {k: torch.cat([f[k] for f in features], dim=0) for k in keys}


class _CountingStore(LocalFeatureStore):
    """LocalFeatureStore that counts get() calls (seek must not materialize)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gets = 0

    def get(self, sample_ref, *, device="cpu", names=None):
        self.gets += 1
        return super().get(sample_ref, device=device, names=names)


class _AdoptingLocalStore(LocalFeatureStore):
    """Local cleanup behavior plus Mooncake-style adopt observability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_events = []

    def adopt(self, ref):
        self.cleanup_events.append(("adopt", ref.sample_id))

    def abort(self, sample_id, *, reason="aborted"):
        self.cleanup_events.append(("abort", sample_id))
        super().abort(sample_id, reason=reason)


class _DurableQueue(SampleRefQueue):
    """Queue fake whose close cleanup must never advance its ack counter."""

    loader_close_retryable = True

    def __init__(self):
        super().__init__()
        self.consumed = 0
        self.failures = []

    def get_interruptible(self, n, *, stop_event):
        while not stop_event.is_set():
            refs = super().get(n, timeout_s=0.01)
            if refs:
                return refs
        return []

    def ack_durable(self, refs):
        super().ack(refs)
        self.consumed += len(refs)

    def fail(self, refs, reason, retryable):
        self.failures.append(([ref.sample_id for ref in refs], reason, retryable))
        super().fail(refs, reason=reason, retryable=retryable)


class _CloneFailure:
    def numel(self):
        return 1

    def element_size(self):
        return 4

    def clone(self):
        raise RuntimeError("clone failed")


class _LeaseTrackingStore:
    def __init__(self):
        self.active_leases = 0
        self.releases = 0

    def get(self, _ref, *, device="cpu"):
        del device
        self.active_leases += 1
        return {"x": _CloneFailure()}, object()

    def release(self, _handle, *, reason):
        del reason
        self.active_leases -= 1
        self.releases += 1


class _BlockingStore:
    """Wrap a real store while holding get() until the test releases it."""

    def __init__(self, inner):
        self.inner = inner
        self.entered = threading.Event()
        self.resume = threading.Event()

    def get(self, ref, *, device="cpu"):
        self.entered.set()
        self.resume.wait()
        return self.inner.get(ref, device=device)

    def release(self, handle, *, reason):
        self.inner.release(handle, reason=reason)

    def gc(self):
        return self.inner.gc()

    def health(self):
        return self.inner.health()


class TestFeatureDataLoader(unittest.TestCase):
    def _write_offline_files(self, d, n=4, seq=8, h=4, aux=12):
        for i in range(n):
            torch.save(
                {
                    "input_ids": torch.arange(seq) + i,
                    "loss_mask": torch.ones(seq, dtype=torch.long),
                    "hidden_state": torch.randn(1, seq, h),
                    "aux_hidden_state": torch.randn(1, seq, aux),
                },
                os.path.join(d, f"{i:03d}.ckpt"),
            )

    def test_queue_partial_batch_drops_terminally_without_failing_run(self):
        with tempfile.TemporaryDirectory() as d:
            store = _AdoptingLocalStore("st")
            refs = [
                store.put(
                    {"x": torch.tensor([[float(index)]])},
                    sample_id=f"s{index}",
                    metadata={"run_id": "run", "target_repr": "hidden_state"},
                )
                for index in range(3)
            ]
            producer = StreamingRefChannel(os.path.join(d, "refs.jsonl"))
            producer.publish_many(refs)
            producer.close()
            q = StreamingRefQueue(StreamingRefChannel(producer.path))
            loader = FeatureDataLoader(
                store,
                q,
                batch_size=2,
                drop_last=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)
            self.assertEqual(producer.consumed_remote(), 3)
            self.assertEqual(store.health()["resident_samples"], 0)
            self.assertEqual(
                store.cleanup_events,
                [("adopt", "s2"), ("abort", "s2")],
            )

    def test_prefetch_queue_partial_batch_drops_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalFeatureStore("st")
            refs = [
                store.put(
                    {"x": torch.tensor([[float(index)]])},
                    sample_id=f"s{index}",
                    metadata={"run_id": "run", "target_repr": "hidden_state"},
                )
                for index in range(3)
            ]
            q = _DurableQueue()
            q.put(refs)
            loader = FeatureDataLoader(
                store,
                q,
                batch_size=2,
                drop_last=True,
                num_workers=1,
            )

            batches = list(loader)
            loader.close()

        self.assertEqual(len(batches), 1)
        self.assertEqual(q.depth(), 0)
        self.assertEqual(q.in_flight(), 0)
        self.assertEqual(len(q.failures), 1)
        self.assertFalse(q.failures[0][2])
        self.assertEqual(store.health()["resident_samples"], 0)
        self.assertIsNone(loader._prefetch_state)

    def test_queue_partial_batch_cleanup_failure_stays_loud(self):
        store = _AdoptingLocalStore("st")
        ref = store.put(
            {"x": torch.tensor([[1.0]])},
            sample_id="s0",
            metadata={"run_id": "run", "target_repr": "hidden_state"},
        )
        q = SampleRefQueue()
        q.put([ref])
        loader = FeatureDataLoader(store, q, batch_size=2, drop_last=True)

        with mock.patch.object(
            store,
            "abort",
            side_effect=RuntimeError("injected abort failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup errors"):
                list(loader)

        self.assertEqual(q.depth(), 0)
        self.assertEqual(q.in_flight(), 0)
        self.assertEqual(store.cleanup_events, [("adopt", "s0")])

    def test_materialize_releases_handle_when_clone_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=1)
            ref = OfflineManifestReader(d, run_id="run").read()[0]
            store = _LeaseTrackingStore()
            loader = FeatureDataLoader(store, refs=[ref])

            with self.assertRaisesRegex(RuntimeError, "clone failed"):
                loader._materialize(ref)

        self.assertEqual(store.active_leases, 0)
        self.assertEqual(store.releases, 1)

    def test_prefetch_materialization_exception_leaves_no_lease_or_thread(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=1)
            refs = OfflineManifestReader(d, run_id="run").read()
            queue = SampleRefQueue()
            queue.put(refs)
            store = LocalFeatureStore("st")

            def fail_transform(_raw):
                raise RuntimeError("transform failed")

            loader = FeatureDataLoader(
                store,
                queue,
                batch_size=1,
                per_sample_transform=fail_transform,
                num_workers=1,
            )
            with self.assertRaisesRegex(RuntimeError, "transform failed"):
                list(loader)
            loader.close()
            loader.close()

        self.assertEqual(queue.in_flight(), 0)
        self.assertEqual(store.health()["active_leases"], 0)
        self.assertIsNone(loader._prefetch_state)

    def test_requires_exactly_one_source(self):
        store = LocalFeatureStore("st")
        with self.assertRaises(ValueError):
            FeatureDataLoader(store)  # neither queue nor refs
        with self.assertRaises(ValueError):
            FeatureDataLoader(store, SampleRefQueue(), refs=[])  # both


if __name__ == "__main__":
    unittest.main(verbosity=2)

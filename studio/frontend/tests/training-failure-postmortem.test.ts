import assert from "node:assert/strict";
import { test } from "node:test";
import { diagnoseTrainingFailure } from "../src/features/training/lib/training-failure-postmortem.ts";

test("no message means no diagnosis", () => {
  assert.equal(diagnoseTrainingFailure(null), null);
  assert.equal(diagnoseTrainingFailure(undefined), null);
  assert.equal(diagnoseTrainingFailure("   "), null);
});

test("out-of-memory variants name the same cause", () => {
  for (const message of [
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: HIP out of memory",
    "RuntimeError: Unable to allocate 1.2 GiB",
    "CUDA error: out of memory",
  ]) {
    assert.deepEqual(diagnoseTrainingFailure(message), { kind: "oom" });
  }
});

test("a non-finite loss is a NaN failure, not an OOM", () => {
  assert.deepEqual(diagnoseTrainingFailure("Loss is NaN at step 12"), {
    kind: "nan",
  });
});

test("disk, auth, dataset and timeout each keep their own cause", () => {
  assert.deepEqual(diagnoseTrainingFailure("OSError: [Errno 28] No space left on device"), {
    kind: "disk",
  });
  assert.deepEqual(diagnoseTrainingFailure("GatedRepoError: 403 Client Error"), {
    kind: "auth",
  });
  assert.deepEqual(
    diagnoseTrainingFailure("Failed to check dataset format: dataset scripts are no longer supported"),
    { kind: "dataset" },
  );
  assert.deepEqual(diagnoseTrainingFailure("Request timed out after 300s"), {
    kind: "timeout",
  });
});

test("an unrecognised log still gets a named bucket", () => {
  assert.deepEqual(diagnoseTrainingFailure("Something odd happened"), {
    kind: "unknown",
  });
});

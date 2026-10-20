// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DATASET_GB_STOPS,
  datasetIsGated,
  datasetSizeBytes,
  matchesDatasetFilters,
} from "../src/features/hub/lib/dataset-filters.ts";

const GB = 1e9;

function dataset(
  overrides: {
    sizeBytes?: number;
    gated?: false | "auto" | "manual";
    sizeBucket?: number;
    modalities?: string[];
  } = {},
) {
  return {
    modalities: overrides.modalities ?? ["text"],
    sizeBucket: overrides.sizeBucket,
    sizeBytes: overrides.sizeBytes,
    gated: overrides.gated,
  };
}

test("the card's byte sizes add up, and download_size stands in when it is alone", () => {
  assert.equal(
    datasetSizeBytes({ dataset_size: 4 * GB, download_size: GB }),
    4 * GB,
    "dataset_size names the unpacked spread, which is what the filter promises",
  );
  assert.equal(datasetSizeBytes({ download_size: 2 * GB }), 2 * GB);
  assert.equal(
    datasetSizeBytes([{ dataset_size: GB }, { dataset_size: 3 * GB }]),
    4 * GB,
    "a multi-config card totals its configs",
  );
  assert.equal(datasetSizeBytes({}), undefined);
  assert.equal(datasetSizeBytes(undefined), undefined);
  assert.equal(
    datasetSizeBytes({ dataset_size: 0, download_size: Number.NaN }),
    undefined,
    "a zero or NaN size is not a published size",
  );
});

test("gated means auto or manual, never false", () => {
  assert.equal(datasetIsGated({ gated: "auto" }), true);
  assert.equal(datasetIsGated({ gated: "manual" }), true);
  assert.equal(datasetIsGated({ gated: false }), false);
  assert.equal(datasetIsGated({}), false);
});

test("a GB range keeps only datasets inside it", () => {
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBytes: 5 * GB }),
      [],
      undefined,
      undefined,
      1,
      10,
    ),
    true,
  );
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBytes: 0.5 * GB }),
      [],
      undefined,
      undefined,
      1,
      10,
    ),
    false,
  );
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBytes: 50 * GB }),
      [],
      undefined,
      undefined,
      1,
      10,
    ),
    false,
  );
  // A dataset publishing no size cannot answer the filter, so it is left out rather than guessed.
  assert.equal(
    matchesDatasetFilters(dataset({}), [], undefined, undefined, 1, 10),
    false,
  );
  // No bounds means every dataset, published size or not.
  assert.equal(
    matchesDatasetFilters(dataset({}), [], undefined, undefined),
    true,
  );
});

test("the GB stops are ascending and the bounds are inclusive where the thumb sits", () => {
  for (let i = 1; i < DATASET_GB_STOPS.length; i++) {
    assert.ok(DATASET_GB_STOPS[i] > DATASET_GB_STOPS[i - 1]);
  }
  const one = DATASET_GB_STOPS[3];
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBytes: one * GB }),
      [],
      undefined,
      undefined,
      one,
      undefined,
    ),
    true,
  );
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBytes: one * GB }),
      [],
      undefined,
      undefined,
      undefined,
      one,
    ),
    true,
  );
  // A card that rounds a 1 GB repo to 999.6 MB still counts as 1 GB.
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBytes: 0.9996 * GB }),
      [],
      undefined,
      undefined,
      1,
      undefined,
    ),
    true,
  );
});

test("gated, open, and any split the list without touching the other filters", () => {
  const gated = dataset({ gated: "manual" });
  const open = dataset({ gated: false });
  assert.equal(
    matchesDatasetFilters(
      gated,
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      "gated",
    ),
    true,
  );
  assert.equal(
    matchesDatasetFilters(
      open,
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      "gated",
    ),
    false,
  );
  assert.equal(
    matchesDatasetFilters(
      open,
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      "public",
    ),
    true,
  );
  assert.equal(
    matchesDatasetFilters(
      gated,
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      "public",
    ),
    false,
  );
  assert.equal(
    matchesDatasetFilters(
      gated,
      [],
      undefined,
      undefined,
      undefined,
      undefined,
      "any",
    ),
    true,
  );
});

test("the filters compose: modality, access, bytes and rows all have to agree", () => {
  const row = dataset({
    sizeBytes: 3 * GB,
    gated: false,
    sizeBucket: 6,
    modalities: ["text"],
  });
  assert.equal(
    matchesDatasetFilters(row, ["text"], 1e6, 1e9, 1, 10, "public"),
    true,
  );
  assert.equal(
    matchesDatasetFilters(row, ["image"], 1e6, 1e9, 1, 10, "public"),
    false,
  );
  assert.equal(
    matchesDatasetFilters(row, ["text"], 1e6, 1e9, 1, 10, "gated"),
    false,
  );
  assert.equal(
    matchesDatasetFilters(row, ["text"], 1e6, 1e9, 10, 100, "public"),
    false,
  );
  // Rows still work on their own for datasets that published a bucket but no bytes.
  assert.equal(
    matchesDatasetFilters(
      dataset({ sizeBucket: 6 }),
      [],
      1e6,
      1e9,
      undefined,
      undefined,
      "any",
    ),
    true,
  );
});

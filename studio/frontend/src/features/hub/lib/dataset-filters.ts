// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The Hub's dataset modalities, as its `modality:` tags spell them. */
export const DATASET_MODALITIES = [
  "3d",
  "audio",
  "document",
  "geospatial",
  "image",
  "tabular",
  "text",
  "timeseries",
  "video",
] as const;
export type DatasetModality = (typeof DATASET_MODALITIES)[number];

// The Hub's size_categories buckets, smallest first: bucket i lies between stop i - 1 and stop i.
const SIZE_BUCKETS = [
  "n<1k",
  "1k<n<10k",
  "10k<n<100k",
  "100k<n<1m",
  "1m<n<10m",
  "10m<n<100m",
  "100m<n<1b",
  "1b<n<10b",
  "10b<n<100b",
  "100b<n<1t",
  "n>1t",
];
export const DATASET_ROW_STOPS = [
  1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
] as const;

// Size on disk, in GB, for the datasets that publish it. Card data reports bytes; a dataset
// that publishes neither dataset_size nor download_size is left to the rows filter instead.
export const DATASET_GB_STOPS = [
  0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000,
] as const;

/** Whose datasets to list: any, the ones that open without a click, or the gated ones. */
export const DATASET_ACCESS_FILTERS = ["any", "public", "gated"] as const;
export type DatasetAccessFilter = (typeof DATASET_ACCESS_FILTERS)[number];

/** Total bytes of the dataset's published shards, when its card says. */
export function datasetSizeBytes(
  datasetInfo:
    | { dataset_size?: unknown; download_size?: unknown }
    | Array<{ dataset_size?: unknown; download_size?: unknown }>
    | undefined,
): number | undefined {
  if (!datasetInfo) {
    return undefined;
  }
  const infos = Array.isArray(datasetInfo) ? datasetInfo : [datasetInfo];
  let total = 0;
  let found = false;
  for (const info of infos) {
    const size = [info?.dataset_size, info?.download_size].find(
      (value): value is number =>
        typeof value === "number" && Number.isFinite(value) && value > 0,
    );
    if (size !== undefined) {
      total += size;
      found = true;
    }
  }
  return found ? total : undefined;
}

/** A gated dataset needs a token and a click on the Hub; "auto" still admits an approved token. */
export function datasetIsGated(dataset: {
  gated?: false | "auto" | "manual";
}): boolean {
  return dataset.gated === "auto" || dataset.gated === "manual";
}

export function datasetModalities(tags: readonly string[]): string[] {
  return tags
    .filter((tag) => tag.startsWith("modality:"))
    .map((tag) => tag.slice("modality:".length).toLowerCase());
}

export function datasetSizeBucket(tags: readonly string[]): number | undefined {
  for (const tag of tags) {
    if (tag.startsWith("size_categories:")) {
      const index = SIZE_BUCKETS.indexOf(
        tag.slice("size_categories:".length).toLowerCase(),
      );
      if (index >= 0) {
        return index;
      }
    }
  }
  return undefined;
}

/** A dataset with any picked modality (none picked: all) whose size bucket meets the row range,
 *  whose published bytes meet the GB range, and whose access matches the pick. Datasets that
 *  don't publish what a filter asks for are left out of it, so the list never shows a result
 *  the filter could not have judged. */
export function matchesDatasetFilters(
  dataset: {
    modalities: readonly string[];
    sizeBucket?: number;
    sizeBytes?: number;
    gated?: false | "auto" | "manual";
  },
  modalities: readonly string[],
  minRows: number | undefined,
  maxRows: number | undefined,
  minGb?: number,
  maxGb?: number,
  access: DatasetAccessFilter = "any",
): boolean {
  if (
    modalities.length > 0 &&
    !dataset.modalities.some((modality) => modalities.includes(modality))
  ) {
    return false;
  }
  if (access !== "any" && datasetIsGated(dataset) !== (access === "gated")) {
    return false;
  }
  if (minGb !== undefined || maxGb !== undefined) {
    if (dataset.sizeBytes === undefined) {
      return false;
    }
    const gb = dataset.sizeBytes / 1e9;
    // The thumbs read in GB; a dataset just under a whole-GB stop still belongs to it.
    if (minGb !== undefined && gb < minGb * 0.999) {
      return false;
    }
    if (maxGb !== undefined && gb > maxGb * 1.001) {
      return false;
    }
  }
  if (minRows === undefined && maxRows === undefined) {
    return true;
  }
  const bucket = dataset.sizeBucket;
  if (bucket === undefined) {
    return false;
  }
  const low = bucket === 0 ? 0 : DATASET_ROW_STOPS[bucket - 1];
  const high =
    bucket === DATASET_ROW_STOPS.length
      ? Number.POSITIVE_INFINITY
      : DATASET_ROW_STOPS[bucket];
  // Both thumbs on one stop: the buckets either side of it.
  if (minRows !== undefined && minRows === maxRows) {
    return low <= minRows && high >= minRows;
  }
  return (
    (minRows === undefined || high > minRows) &&
    (maxRows === undefined || low < maxRows)
  );
}

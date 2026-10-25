// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "vision" | "audio" | "embeddings" | "text" | "decision";
export type TrainingMethod =
  | "qlora"
  | "lora"
  | "full"
  | "cpt"
  | "rl"
  | "cora"
  // Labs experiment: capacity matched training runs on its own engine, not the trainer.
  | "cmt";

export function isTrainingMethod(value: unknown): value is TrainingMethod {
  return (
    value === "qlora" ||
    value === "lora" ||
    value === "full" ||
    value === "cpt" ||
    value === "rl" ||
    value === "cora" ||
    value === "cmt"
  );
}

export function isAdapterMethod(method: TrainingMethod): boolean {
  // CMT inserts whole transformer blocks and trains them densely, so no LoRA variant applies.
  return (
    method === "lora" ||
    method === "qlora" ||
    method === "cpt" ||
    method === "rl" ||
    method === "cora"
  );
}
export type DatasetSource = "huggingface" | "upload" | "s3";

/** S3 bucket configuration for loading datasets */
export interface S3Config {
  bucket: string;
  region: string;
  prefix?: string;
  accessKeyId?: string;
  secretAccessKey?: string;
  useIamRole?: boolean;
}
export type DatasetFormat = "auto" | "alpaca" | "chatml" | "sharegpt" | "raw";
export type GradientCheckpointing = "none" | "true" | "unsloth" | "mlx";

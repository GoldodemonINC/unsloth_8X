// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAdapterMethod } from "@/types/training";
import type {
  DatasetFormat,
  ModelType,
  TrainingMethod,
} from "@/types/training";
import type { LoraVariant } from "../types/config";

// CMT is absent on purpose: it trains on the Labs engine, which takes its own
// datasets and steps, so there is no training_type to send the trainer.
const BACKEND_TRAINING_TYPE: Record<
  Exclude<TrainingMethod, "cmt">,
  string
> = {
  cora: "LoRA/QLoRA",
  qlora: "LoRA/QLoRA",
  lora: "LoRA/QLoRA",
  full: "Full Finetuning",
  cpt: "Continued Pretraining",
  rl: "Policy Alignment",
};

const TRAINING_METHOD_LABELS: Record<TrainingMethod, string> = {
  cora: "CoRA",
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full",
  cpt: "CPT",
  rl: "RL",
  cmt: "CMT",
};

export function toBackendTrainingType(trainingMethod: TrainingMethod): string {
  if (trainingMethod === "cmt") {
    throw new Error(
      "CMT runs on its own engine; it is never sent to the trainer.",
    );
  }
  return BACKEND_TRAINING_TYPE[trainingMethod];
}

export function getTrainingMethodLabel(
  trainingMethod: TrainingMethod | string,
): string {
  if (
    Object.prototype.hasOwnProperty.call(TRAINING_METHOD_LABELS, trainingMethod)
  ) {
    return TRAINING_METHOD_LABELS[trainingMethod as TrainingMethod];
  }
  return TRAINING_METHOD_LABELS.full;
}

export function isTrainingMethodSupportedOnDevice(
  trainingMethod: TrainingMethod,
  deviceType?: string,
  cpuOnly = false,
  simulated = false,
): boolean {
  // A simulated run (Settings -> Debugging) never loads the trainer, so the host's limits
  // must not hide a method: the fake backend ignores the payload and any method can be exercised.
  if (simulated) {
    return true;
  }
  // CMT expands the checkpoint, then trains the new blocks densely: it needs a CUDA-class
  // accelerator and refuses quantized loads, so CPU and Apple Silicon hosts cannot run it.
  if (trainingMethod === "cmt") {
    return !cpuOnly && deviceType !== "cpu" && deviceType !== "mac";
  }
  // No torch accelerator: CoRA and QLoRA are the two recipes the CPU path runs (both are LoRA
  // adapters; QLoRA's 4-bit load is dropped there because bitsandbytes has no CPU training
  // kernel). Full, CPT and RL need Unsloth's GPU loaders and would crash. deviceType stays here
  // for callers passing the legacy "cpu" value; the real signal is cpuOnly.
  if (cpuOnly || deviceType === "cpu") {
    return trainingMethod === "cora" || trainingMethod === "qlora";
  }
  if (deviceType === "mac") {
    return (
      trainingMethod !== "cpt" &&
      trainingMethod !== "rl" &&
      trainingMethod !== "cora"
    );
  }
  return true;
}

export function isTrainingModelTypeSupportedOnDevice(
  modelType: ModelType | null,
  deviceType?: string,
): boolean {
  return (
    deviceType !== "mac" ||
    (modelType !== "audio" && modelType !== "embeddings")
  );
}

export function isTrainingLoraVariantSupportedOnDevice(
  loraVariant: LoraVariant,
  trainingMethod: TrainingMethod,
  deviceType?: string,
  cpuOnly = false,
  simulated = false,
): boolean {
  if (simulated) {
    return true;
  }
  const usesAdapter = isAdapterMethod(trainingMethod);
  if (!usesAdapter || loraVariant !== "loftq") return true;
  // LoftQ initialises through bitsandbytes, which needs Apple MLX or a CUDA-class accelerator.
  return !cpuOnly && deviceType !== "mac" && deviceType !== "cpu";
}

export function parseBackendTrainingMethod(
  trainingType: unknown,
  loadIn4Bit: unknown,
): TrainingMethod {
  if (trainingType === "Continued Pretraining") {
    return "cpt";
  }
  if (trainingType === "Policy Alignment") {
    return "rl";
  }
  if (trainingType === "LoRA/QLoRA") {
    return loadIn4Bit ? "qlora" : "lora";
  }
  return "full";
}

export function isRawTextDatasetFormat(datasetFormat: DatasetFormat): boolean {
  return datasetFormat === "raw";
}

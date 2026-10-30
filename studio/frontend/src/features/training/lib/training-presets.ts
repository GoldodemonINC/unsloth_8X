// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { useTrainingConfigStore } from "../stores/training-config-store";
import type {
  TrainingPreset,
  TrainingPresetValues,
} from "./training-preset-data";

export * from "./training-preset-data";

export function capturePresetValues(): TrainingPresetValues {
  const state = useTrainingConfigStore.getState();
  return {
    trainingMethod: state.trainingMethod,
    loraRank: state.loraRank,
    loraAlpha: state.loraAlpha,
    loraDropout: state.loraDropout,
    learningRate: state.learningRate,
    epochs: state.epochs,
    batchSize: state.batchSize,
    gradientAccumulation: state.gradientAccumulation,
    contextLength: state.contextLength,
    gradientCheckpointing: state.gradientCheckpointing,
    packing: state.packing,
    trainOnCompletions: state.trainOnCompletions,
  };
}

/** Apply only training hyperparameters; the model and dataset stay untouched. */
export function applyPreset(preset: TrainingPreset): void {
  const store = useTrainingConfigStore.getState();
  const values = preset.values;
  store.setTrainingMethod(values.trainingMethod);
  store.setLoraRank(values.loraRank);
  store.setLoraAlpha(values.loraAlpha);
  store.setLoraDropout(values.loraDropout);
  store.setLearningRate(values.learningRate);
  store.setEpochs(values.epochs);
  store.setBatchSize(values.batchSize);
  store.setGradientAccumulation(values.gradientAccumulation);
  store.setContextLength(values.contextLength);
  store.setGradientCheckpointing(values.gradientCheckpointing);
  store.setPacking(values.packing);
  store.setTrainOnCompletions(values.trainOnCompletions);
}

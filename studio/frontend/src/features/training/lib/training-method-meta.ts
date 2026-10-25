// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { TrainingMethod } from "@/types/training";

export const TRAINING_METHOD_ORDER: readonly TrainingMethod[] = [
  "cora",
  "qlora",
  "lora",
  "full",
  "cpt",
  "rl",
  "cmt",
];

/**
 * The rows the method picker shows. CMT is a Labs experiment, so it exists only
 * while that switch is on, and a decision model has no backbone for it anyway.
 */
export function visibleTrainingMethods(input: {
  cmtEnabled: boolean;
  isDecision: boolean;
  decisionTakesQlora: boolean;
}): TrainingMethod[] {
  return TRAINING_METHOD_ORDER.filter(
    (method) =>
      (method !== "cmt" || input.cmtEnabled) &&
      (!input.isDecision ||
        method === "lora" ||
        method === "full" ||
        (input.decisionTakesQlora && method === "qlora")),
  );
}

interface TrainingMethodMeta {
  labelKey: TranslationKey;
  hintKey: TranslationKey;
  noteKey: TranslationKey;
  dotClass: string;
}

export const TRAINING_METHOD_META: Record<TrainingMethod, TrainingMethodMeta> =
  {
    cora: {
      labelKey: "studio.methods.cora.label",
      hintKey: "studio.methods.cora.hint",
      noteKey: "studio.methods.cora.note",
      dotClass: "bg-blue-500/70",
    },
    qlora: {
      labelKey: "studio.methods.qlora.label",
      hintKey: "studio.methods.qlora.hint",
      noteKey: "studio.methods.qlora.note",
      dotClass: "bg-emerald-500/70",
    },
    lora: {
      labelKey: "studio.methods.lora.label",
      hintKey: "studio.methods.lora.hint",
      noteKey: "studio.methods.lora.note",
      dotClass: "bg-sky-500/70",
    },
    full: {
      labelKey: "studio.methods.full.label",
      hintKey: "studio.methods.full.hint",
      noteKey: "studio.methods.full.note",
      dotClass: "bg-amber-500/70",
    },
    cpt: {
      labelKey: "studio.methods.cpt.label",
      hintKey: "studio.methods.cpt.hint",
      noteKey: "studio.methods.cpt.note",
      dotClass: "bg-violet-500/70",
    },
    rl: {
      labelKey: "studio.methods.rl.label",
      hintKey: "studio.methods.rl.hint",
      noteKey: "studio.methods.rl.note",
      dotClass: "bg-rose-500/70",
    },
    // CMT's copy lives under Labs because it is only offered while that experiment is on,
    // and an experiment's labels should travel with it rather than with the stable trainer.
    cmt: {
      labelKey: "labs.cmt.methodLabel",
      hintKey: "labs.cmt.methodHint",
      noteKey: "labs.cmt.methodNote",
      dotClass: "bg-teal-500/70",
    },
  };

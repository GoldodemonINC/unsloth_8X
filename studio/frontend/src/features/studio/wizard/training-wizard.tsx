// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_TRIGGER_CLASS } from "@/components/resource-picker/picker-focus";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Tabs } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { HfTokenIndicator } from "@/features/hub";
import {
  DECISION_MODEL_LABELS,
  useDebugTrainingSimulation,
  useSettingsDialogStore,
} from "@/features/settings";
import { TrainModelSelector } from "@/features/train-model-picker";
import {
  type DecisionCheckpoint,
  TRAINING_METHOD_META,
  TRAINING_METHOD_ORDER,
  isTrainingMethodSupportedOnDevice,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { TrainingMethod } from "@/types/training";
import {
  AiBrain01Icon,
  ArrowExpand02Icon,
  Database02Icon,
  FloppyDiskIcon,
  Settings05Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type CSSProperties, type ReactNode, useEffect } from "react";
import { DatasetPanel } from "../sections/dataset-section";
import { DatasetSourceToggleAction } from "../sections/dataset-source-toggle";
import { FieldHint } from "../sections/field-hint";
import { ParamsSection } from "../sections/params-section";
import { ConfigActions } from "./config-actions";
import { AdvancedTrainingSection } from "@/features/advanced/components/advanced-training-section";
import { ScalerPanel } from "@/features/advanced/components/scaler-panel";
import type { ParamMode } from "./training-param-mode";

function SectionBox({
  title,
  description,
  icon,
  chipTint,
  titleAction,
  children,
  dataTour,
}: {
  title?: string;
  description?: string;
  icon?: IconSvgElement;
  chipTint?: string;
  titleAction?: ReactNode;
  children: ReactNode;
  dataTour?: string;
}) {
  return (
    <section
      data-tour={dataTour}
      className="@container/train-card elevated-card flex flex-col gap-7 bg-card p-5 pb-7"
    >
      {title && (
        <div className="flex flex-col items-stretch gap-3 @md/train-card:flex-row @md/train-card:items-center @md/train-card:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            {icon && (
              <span
                className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
                style={
                  chipTint
                    ? ({ "--chip-tint": chipTint } as CSSProperties)
                    : undefined
                }
              >
                <HugeiconsIcon
                  icon={icon}
                  strokeWidth={1.5}
                  className="size-[calc(18px*var(--ui-space-scale,1))]"
                />
              </span>
            )}
            <div className="min-w-0">
              <h2 className="select-none text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
                {title}
              </h2>
              {description && (
                <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
                  {description}
                </p>
              )}
            </div>
          </div>
          {titleAction && (
            <div className="w-full min-w-0 @md/train-card:w-auto @md/train-card:shrink-0">
              {titleAction}
            </div>
          )}
        </div>
      )}
      <div className="@container/train-section min-w-0">{children}</div>
    </section>
  );
}

function ParamModeToggle({
  mode,
  onChange,
}: {
  mode: ParamMode;
  onChange: (next: ParamMode) => void;
}) {
  const t = useT();
  const options = [
    { value: "simple", label: t("studio.params.mode.simple") },
    { value: "advanced", label: t("studio.params.mode.advanced") },
  ] as const;

  return (
    <Tabs
      value={mode}
      onValueChange={(value) => onChange(value as ParamMode)}
      className="contents"
    >
      <SegmentedTabsList
        value={mode}
        options={options}
        ariaLabel={t("studio.params.mode.ariaLabel")}
        size="compact"
        className="@md/train-card:w-[calc(180px*var(--ui-space-scale,1))]"
      />
    </Tabs>
  );
}

function SetupField({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {label}
        {hint ? <FieldHint text={hint} label={label} /> : null}
      </span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function TrainingMethodSelect() {
  const t = useT();
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const setTrainingMethod = useTrainingConfigStore((s) => s.setTrainingMethod);
  const isDecision = useTrainingConfigStore((s) => s.modelType === "decision");
  // Clef and LLM decision models have an LLM backbone, so they take QLoRA; Laya does not.
  const decisionTakesQlora = useTrainingConfigStore(
    (s) =>
      s.modelType === "decision" &&
      (s.decisionLayout === "clef" || s.decisionLayout === "llm"),
  );
  const deviceType = usePlatformStore((state) => state.deviceType);
  const cpuOnly = usePlatformStore((state) => state.isCpuTrainingHost());
  // Settings -> Debugging: the simulated trainer ignores the method, so no host limit
  // narrows the list and nothing gets rewritten under the user.
  const simulated = useDebugTrainingSimulation();
  // A method persisted from another host (or the default) would sit selected while its row is
  // disabled, and Start would refuse it. On CPU, CoRA is the only choice, so make it the choice.
  useEffect(() => {
    if (
      !simulated &&
      cpuOnly &&
      !isTrainingMethodSupportedOnDevice(trainingMethod, deviceType, cpuOnly)
    ) {
      setTrainingMethod("cora");
    }
  }, [simulated, cpuOnly, deviceType, trainingMethod, setTrainingMethod]);
  const activeMeta = TRAINING_METHOD_META[trainingMethod];
  const activeLabel = activeMeta ? t(activeMeta.labelKey) : trainingMethod;
  return (
    <Select
      value={trainingMethod}
      onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
    >
      <SelectTrigger
        aria-label={`${t("studio.wizard.methodLabel")}: ${activeLabel}`}
        className={cn(
          PICKER_TRIGGER_CLASS,
          "w-full min-w-[calc(148px*var(--ui-space-scale,1))] justify-between",
        )}
        data-tour="studio-method"
      >
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className={cn(
              "size-2 shrink-0 rounded-full",
              activeMeta?.dotClass ?? "bg-muted-foreground",
            )}
          />
          <span className="truncate font-medium text-foreground">
            {activeLabel}
          </span>
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        className="rounded-[14px] ring-0"
      >
        {TRAINING_METHOD_ORDER.filter(
          (method) =>
            !isDecision ||
            method === "lora" ||
            method === "full" ||
            (decisionTakesQlora && method === "qlora"),
        ).map((method) => {
          const meta = TRAINING_METHOD_META[method];
          const unsupported = !isTrainingMethodSupportedOnDevice(
            method,
            deviceType,
            cpuOnly,
            simulated,
          );
          const unsupportedLabel = t(
            cpuOnly
              ? "studio.params.notSupportedCpu"
              : "studio.params.notSupportedAppleSilicon",
          );
          return (
            <Tooltip key={method} delayDuration={300}>
              <TooltipTrigger asChild={true} disableClickToggle={true}>
                <SelectItem value={method} disabled={unsupported}>
                  <span className="flex items-center gap-2">
                    <span
                      aria-hidden="true"
                      className={cn(
                        "size-2 shrink-0 rounded-full",
                        meta.dotClass,
                      )}
                    />
                    {t(meta.labelKey)}
                    {unsupported && (
                      <span className="text-ui-10 text-muted-foreground">
                        · {unsupportedLabel}
                      </span>
                    )}
                  </span>
                </SelectItem>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                sideOffset={10}
                className="max-w-[calc(220px*var(--ui-space-scale,1))] text-ui-11p5 leading-snug"
              >
                {unsupported ? unsupportedLabel : t(meta.hintKey)}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </SelectContent>
    </Select>
  );
}

function TrainAsSelect() {
  const t = useT();
  const trainAsDecision = useTrainingConfigStore((s) => s.trainAsDecision);
  const setTrainAsDecision = useTrainingConfigStore(
    (s) => s.setTrainAsDecision,
  );
  const value = trainAsDecision ? "decision" : "language";
  const label = (option: string) =>
    option === "decision"
      ? t("studio.wizard.trainAsDecision")
      : t("studio.wizard.trainAsLanguage");
  return (
    <Select
      value={value}
      onValueChange={(next) => setTrainAsDecision(next === "decision")}
    >
      <SelectTrigger
        aria-label={`${t("studio.wizard.trainAsLabel")}: ${label(value)}`}
        className={cn(PICKER_TRIGGER_CLASS, "w-full justify-between")}
      >
        <span className="truncate font-medium text-foreground">
          {label(value)}
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        className="rounded-[14px] ring-0"
      >
        {["language", "decision"].map((option) => (
          <SelectItem key={option} value={option}>
            {label(option)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function CheckpointSelect({
  checkpoints,
}: {
  checkpoints: DecisionCheckpoint[];
}) {
  const t = useT();
  const modelSubfolder = useTrainingConfigStore((s) => s.modelSubfolder);
  const setModelSubfolder = useTrainingConfigStore((s) => s.setModelSubfolder);
  const active =
    checkpoints.find((checkpoint) => checkpoint.subfolder === modelSubfolder) ??
    checkpoints[0];
  const checkpointLabel = (checkpoint: DecisionCheckpoint) =>
    DECISION_MODEL_LABELS[checkpoint.name]
      ? t(DECISION_MODEL_LABELS[checkpoint.name])
      : checkpoint.name;
  return (
    <Select
      value={active.name}
      onValueChange={(name) => {
        const next = checkpoints.find((checkpoint) => checkpoint.name === name);
        if (next) {
          setModelSubfolder(next.subfolder);
        }
      }}
    >
      <SelectTrigger
        aria-label={`${t("studio.wizard.checkpointLabel")}: ${checkpointLabel(active)}`}
        className={cn(PICKER_TRIGGER_CLASS, "w-full justify-between")}
      >
        <span className="truncate font-medium text-foreground">
          {checkpointLabel(active)}
        </span>
      </SelectTrigger>
      <SelectContent
        position="popper"
        side="bottom"
        align="start"
        sideOffset={8}
        className="rounded-[14px] ring-0"
      >
        {checkpoints.map((checkpoint) => (
          <SelectItem key={checkpoint.name} value={checkpoint.name}>
            {checkpointLabel(checkpoint)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ModelPanel() {
  const t = useT();
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const decisionCheckpoints = useTrainingConfigStore((s) =>
    s.modelType === "decision" ? s.decisionCheckpoints : null,
  );
  // Text and vision LLMs can also train as decision models; Laya, Clef, audio and embeddings cannot.
  const canTrainAsDecision = useTrainingConfigStore(
    (s) =>
      s.selectedModel !== null &&
      (s.decisionLayout === "llm" ||
        s.modelType === "text" ||
        s.modelType === "vision"),
  );
  return (
    <div className="grid grid-cols-1 gap-4 @md/train-section:grid-cols-2 @2xl/train-section:grid-cols-[minmax(0,1fr)_180px_200px]">
      <div className="@md/train-section:col-span-2 @2xl/train-section:col-span-1">
        <SetupField
          label={t("studio.wizard.modelLabel")}
          hint={t("studio.wizard.modelTooltip")}
        >
          <TrainModelSelector />
        </SetupField>
      </div>
      <SetupField
        label={t("studio.wizard.methodLabel")}
        hint={t("studio.wizard.methodTooltip")}
      >
        <TrainingMethodSelect />
      </SetupField>
      <SetupField
        label={t("picker.hfToken.label")}
        hint={t("studio.wizard.hfTokenDescription")}
      >
        <HfTokenIndicator
          showLabel={true}
          onOpenSettings={() => openSettings("general")}
        />
      </SetupField>
      {canTrainAsDecision ? (
        <SetupField
          label={t("studio.wizard.trainAsLabel")}
          hint={t("studio.wizard.trainAsTooltip")}
        >
          <TrainAsSelect />
        </SetupField>
      ) : null}
      {decisionCheckpoints && decisionCheckpoints.length > 0 ? (
        <SetupField
          label={t("studio.wizard.checkpointLabel")}
          hint={t("studio.wizard.checkpointTooltip")}
        >
          <CheckpointSelect checkpoints={decisionCheckpoints} />
        </SetupField>
      ) : null}
    </div>
  );
}

function CoraRecipeCard() {
  // The GPU recipe quantizes through bitsandbytes; a CPU host has neither that kernel nor LoftQ,
  // so the card names what actually runs there instead of a recipe this host cannot apply.
  const cpuOnly = usePlatformStore((state) => state.isCpuTrainingHost());
  const rows: Array<[string, string]> = [
    [
      "Adapter",
      cpuOnly ? "r=8, alpha=24, plain LoRA" : "r=8, alpha=24, LoftQ init",
    ],
    [
      "Base precision",
      cpuOnly
        ? "fp32 (a CPU has no 4-bit training kernel)"
        : "4-bit (Q4_1), exported later to the quant you pick",
    ],
    ["Optimizer", cpuOnly ? "AdamW (torch)" : "Paged AdamW 8-bit"],
    ["Schedule", "Cosine"],
    ["Batch", "1 x 3 gradient accumulation"],
    ["Context", "1024 tokens"],
    [
      "Quantization-aware",
      cpuOnly ? "Int4 QAT, best-effort on CPU" : "Int4 QAT on",
    ],
    ["Layer LRs", "Embedding 0.25x to head 1.0x"],
    ["Dataset", "0.01 GB to 0.8 GB of text"],
  ];
  return (
    <SectionBox
      title="CoRA recipe"
      description="The CPU-first adapter recipe, fixed for every CoRA run"
      icon={Settings05Icon}
      chipTint="var(--chart-2)"
    >
      <dl className="grid gap-x-6 gap-y-2 text-ui-11 sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="flex items-baseline justify-between gap-3 border-b border-border/40 pb-1.5">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="text-right font-medium text-foreground">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-ui-10p5 leading-ui-14 text-muted-foreground/70">
        {cpuOnly
          ? "QAT and layer-wise learning rates are on by default for CoRA; adjust them in Advanced training below. The 4-bit pieces are dropped on a CPU host."
          : "QAT and layer-wise learning rates are on by default for CoRA; adjust them in Advanced training below."}
      </p>
    </SectionBox>
  );
}

export function TrainingWizard({
  paramMode,
  onParamModeChange,
}: {
  paramMode: ParamMode;
  onParamModeChange: (next: ParamMode) => void;
}) {
  const t = useT();
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const selectedModel = useTrainingConfigStore((s) => s.selectedModel);
  return (
    <div className="flex flex-col gap-6">
      <SectionBox
        title={t("studio.wizard.modelTitle")}
        description={t("studio.wizard.modelDescription")}
        icon={AiBrain01Icon}
        chipTint="var(--chart-1)"
      >
        <ModelPanel />
      </SectionBox>

      <SectionBox
        title="Model scaling"
        description="Grow the model before training it, from its own parameter count up to base + 3B"
        icon={ArrowExpand02Icon}
        chipTint="var(--chart-2)"
      >
        <ScalerPanel
          defaultSource={selectedModel ?? ""}
          onUseCheckpoint={(path) =>
            useTrainingConfigStore
              .getState()
              .selectTrainingModel(path, null, { localPath: path })
          }
        />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.datasetTitle")}
        description={t("studio.wizard.datasetDescription")}
        icon={Database02Icon}
        chipTint="var(--chart-4)"
        dataTour="studio-dataset"
        titleAction={<DatasetSourceToggleAction />}
      >
        <DatasetPanel />
      </SectionBox>

      <SectionBox
        title={t("studio.wizard.paramsTitle")}
        description={t("studio.wizard.paramsDescription")}
        icon={Settings05Icon}
        chipTint="var(--chart-5)"
        dataTour="studio-params"
        titleAction={
          <ParamModeToggle mode={paramMode} onChange={onParamModeChange} />
        }
      >
        <ParamsSection mode={paramMode} />
      </SectionBox>

      {trainingMethod === "cora" ? <CoraRecipeCard /> : null}
      <AdvancedTrainingSection />

      <SectionBox
        title={t("studio.wizard.configTitle")}
        description={t("studio.wizard.configDescription")}
        icon={FloppyDiskIcon}
        chipTint="var(--chart-3)"
      >
        <ConfigActions />
      </SectionBox>
    </div>
  );
}

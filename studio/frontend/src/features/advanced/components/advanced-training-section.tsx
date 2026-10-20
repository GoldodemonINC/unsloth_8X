// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useTrainingConfigStore } from "@/features/training";
import {
  DEFAULT_LAYER_KIND_MULTIPLIERS,
  LAYER_PARAM_KINDS,
  LAYER_ZONES,
  MOE_EXPERT_SCOPES,
  MOE_SCOPE_DESCRIPTIONS,
  MOE_SCOPE_LABELS,
  type MoeExpertScope,
  QAT_FORMATS,
  REWARD_PRESETS,
  RL_METHODS,
  newBlendSlot,
  type LayerParamKind,
  type LayerZone,
  type QatFormat,
  type RewardPreset,
  type RlMethod,
} from "../types";
import { PlusSignIcon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, type ReactNode } from "react";

const EVAL_SUITES = [
  { id: "mmlu", label: "MMLU" },
  { id: "gsm8k", label: "GSM8K" },
  { id: "arc_challenge", label: "ARC-Challenge" },
  { id: "hellaswag", label: "HellaSwag" },
  { id: "truthfulqa", label: "TruthfulQA" },
  { id: "humaneval", label: "HumanEval" },
  { id: "ifeval", label: "IFEval" },
] as const;

const KIND_LABELS: Record<LayerParamKind, string> = {
  attention: "Attention",
  mlp: "MLP / FFN",
  norm: "Layer norms",
  moe_expert: "MoE experts",
  moe_router: "MoE routers",
  hybrid_block: "Mamba / hybrid",
};

const ZONE_LABELS: Record<LayerZone, string> = {
  embedding: "Embeddings",
  early: "Early layers",
  middle: "Middle layers",
  late: "Late layers",
  head: "Output head",
  adapter: "Adapters",
};

function Row({
  title,
  description,
  checked,
  onCheckedChange,
}: {
  title: string;
  description: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <p className="text-ui-12p5 font-medium text-foreground">{title}</p>
        <p className="mt-0.5 text-ui-11 leading-ui-15 text-muted-foreground/85">
          {description}
        </p>
      </div>
      <Switch checked={checked} onCheckedChange={onCheckedChange} size="sm" />
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {label}
      </span>
      {children}
      {hint ? (
        <span className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
          {hint}
        </span>
      ) : null}
    </label>
  );
}

function NumberField({
  value,
  onChange,
  min,
  max,
  step = 1,
  suffix,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <Input
        type="number"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
            onChange(Math.min(max, Math.max(min, next)));
          }
        }}
        className="h-8 w-28"
      />
      {suffix ? (
        <span className="text-ui-11 text-muted-foreground">{suffix}</span>
      ) : null}
    </div>
  );
}

export function AdvancedTrainingSection() {
  const advanced = useTrainingConfigStore((s) => s.advanced);
  const setAdvanced = useTrainingConfigStore((s) => s.setAdvanced);
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const isMoeModel = useTrainingConfigStore((s) => s.isMoeModel);
  const numExperts = useTrainingConfigStore((s) => s.numExperts);
  const numExpertsPerToken = useTrainingConfigStore(
    (s) => s.numExpertsPerToken,
  );
  const moeExpertScope = useTrainingConfigStore((s) => s.moeExpertScope);
  const setMoeExpertScope = useTrainingConfigStore((s) => s.setMoeExpertScope);
  const isPolicyAlignment = trainingMethod === "rl";

  const totalRatio = useMemo(
    () =>
      advanced.blendSlots.reduce(
        (sum, slot) => sum + (Number.isFinite(slot.ratio) ? slot.ratio : 0),
        0,
      ),
    [advanced.blendSlots],
  );

  return (
    <section className="@container/train-card elevated-card flex flex-col gap-5 bg-card p-5 pb-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
            Advanced training
          </h2>
          <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
            Precision simulation, multi-dataset pipelines, and post-run hooks.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {advanced.qatEnabled ? <Badge variant="secondary">QAT</Badge> : null}
          {advanced.curriculumEnabled ? (
            <Badge variant="outline">Curriculum · experimental</Badge>
          ) : null}
          {advanced.layerLrEnabled ? (
            <Badge variant="secondary">Layer LR</Badge>
          ) : null}
          {advanced.blendEnabled ? (
            <Badge variant="secondary">Blend</Badge>
          ) : null}
          {advanced.autoEvalEnabled ? (
            <Badge variant="secondary">Regression Guard</Badge>
          ) : null}
          {isPolicyAlignment ? <Badge variant="secondary">RL</Badge> : null}
        </div>
      </div>

      <Accordion type="multiple" className="border-border/60 bg-transparent">
        <AccordionItem value="precision">
          <AccordionTrigger className="px-3 text-ui-12p5">
            Precision &amp; quantization
          </AccordionTrigger>
          <AccordionContent className="h-auto! px-3 pb-5">
            <div className="flex flex-col gap-4">
              <Row
                title="Enable QAT simulation"
                description="Fake-quantizes weights inside the forward graph so the model learns to compensate for INT4 / NVFP4 noise."
                checked={advanced.qatEnabled}
                onCheckedChange={(qatEnabled) => setAdvanced({ qatEnabled })}
              />
              {advanced.qatEnabled ? (
                <div className="grid gap-4 sm:grid-cols-2">
                  <Field
                    label="Quantization grid"
                    hint="INT4 and NVFP4 match the export formats; NF4 matches the QLoRA grid."
                  >
                    <Select
                      value={advanced.qatFormat}
                      onValueChange={(value) =>
                        setAdvanced({ qatFormat: value as QatFormat })
                      }
                    >
                      <SelectTrigger className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {QAT_FORMATS.map((format) => (
                          <SelectItem key={format} value={format}>
                            {format.toUpperCase()}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <Field
                    label="Group size"
                    hint="Weights sharing one scale. Smaller groups are more accurate and slower."
                  >
                    <Select
                      value={String(advanced.qatGroupSize)}
                      onValueChange={(value) =>
                        setAdvanced({
                          qatGroupSize: Number(value) as
                            | 16
                            | 32
                            | 64
                            | 128
                            | 256,
                        })
                      }
                    >
                      <SelectTrigger className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {[16, 32, 64, 128, 256].map((size) => (
                          <SelectItem key={size} value={String(size)}>
                            {size}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                </div>
              ) : null}
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="moe">
          <AccordionTrigger className="px-3 text-ui-12p5">
            MoE experts
            {isMoeModel ? (
              <Badge variant="secondary" className="ml-2">
                {numExperts ? `${numExperts} experts` : "mixture of experts"}
              </Badge>
            ) : null}
          </AccordionTrigger>
          <AccordionContent className="h-auto! px-3 pb-5">
            <div className="flex flex-col gap-4">
              {!isMoeModel ? (
                <p className="text-ui-11p5 leading-ui-16 text-muted-foreground/85">
                  The selected model has no mixture-of-experts layers, so there
                  are no experts to train. Pick a MoE checkpoint (Mixtral,
                  Qwen3-MoE, DeepSeek-MoE, gpt-oss) to see this choice.
                </p>
              ) : (
                <>
                  <p className="text-ui-11p5 leading-ui-16 text-muted-foreground/85">
                    {numExperts ?? "Some"} routed experts
                    {numExpertsPerToken
                      ? `, ${numExpertsPerToken} per token`
                      : ""}
                    .
                    {(numExpertsPerToken ?? 0) > 0 && numExperts
                      ? ` Each token reads ${Math.round(((numExpertsPerToken ?? 0) / numExperts) * 100)}% of the experts, so a LoRA on the experts only moves the weights a prompt actually routes to.`
                      : ""}
                  </p>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Field
                      label="Train scope"
                      hint="What the run teaches. Attention and router stay frozen unless the scope names them."
                    >
                      <Select
                        value={moeExpertScope ?? "default"}
                        disabled={trainingMethod === "full"}
                        onValueChange={(value) =>
                          setMoeExpertScope(
                            value === "default"
                              ? null
                              : (value as MoeExpertScope),
                          )
                        }
                      >
                        <SelectTrigger className="h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="default">
                            {MOE_SCOPE_LABELS.experts_and_attention} (Studio
                            default)
                          </SelectItem>
                          {MOE_EXPERT_SCOPES.filter(
                            (scope) => scope !== "experts_and_attention",
                          ).map((scope) => (
                            <SelectItem key={scope} value={scope}>
                              {MOE_SCOPE_LABELS[scope]}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Field>
                  </div>
                  <div className="rounded-xl bg-muted/30 p-3 text-ui-10p5 leading-ui-15 text-muted-foreground">
                    {
                      MOE_SCOPE_DESCRIPTIONS[
                        moeExpertScope ?? "experts_and_attention"
                      ]
                    }
                  </div>
                  {trainingMethod === "full" ? (
                    <p className="text-ui-10p5 leading-ui-15 text-amber-600 dark:text-amber-400">
                      Full finetuning ignores this choice: it already trains
                      every expert.
                    </p>
                  ) : (
                    <p className="text-ui-10p5 leading-ui-15 text-muted-foreground/70">
                      The run resolves this into PEFT&apos;s expert parameters,
                      so the payload and the manifest record the same leaves the
                      loader trained.
                    </p>
                  )}
                </>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="pipeline">
          <AccordionTrigger className="px-3 text-ui-12p5">
            Training pipeline &amp; datasets
          </AccordionTrigger>
          <AccordionContent className="h-auto! px-3 pb-5">
            <div className="flex flex-col gap-6">
              <div className="flex flex-col gap-3">
                <Row
                  title="Curriculum co-training (experimental)"
                  description="Alternate batch steps between domains on a fixed cadence. Experimental: there is no equal-compute A/B showing it beats plain blending, so treat any gain as unproven."
                  checked={advanced.curriculumEnabled}
                  onCheckedChange={(curriculumEnabled) =>
                    setAdvanced({ curriculumEnabled })
                  }
                />
                {advanced.curriculumEnabled ? (
                  <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
                    <Field
                      label="Step cadence"
                      hint="Format: name:steps,name:steps — e.g. reasoning:2,code:1."
                    >
                      <Input
                        value={advanced.curriculumCadence}
                        onChange={(event) =>
                          setAdvanced({
                            curriculumCadence: event.target.value,
                          })
                        }
                        className="h-8 font-mono text-ui-12"
                      />
                    </Field>
                    <div className="self-end pb-5">
                      <PlanBadge cadence={advanced.curriculumCadence} />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="h-px bg-border/60" />

              <div className="flex flex-col gap-3">
                <Row
                  title="Layer-wise learning rates"
                  description="Scale the LR by structural zone: protect early attention, let the head move faster."
                  checked={advanced.layerLrEnabled}
                  onCheckedChange={(layerLrEnabled) =>
                    setAdvanced({ layerLrEnabled })
                  }
                />
                {advanced.layerLrEnabled ? (
                  <div className="flex flex-col gap-3 rounded-xl border border-border/60 p-3">
                    {LAYER_ZONES.map((zone) => (
                      <div
                        key={zone}
                        className="grid grid-cols-[minmax(104px,0.8fr)_minmax(0,1.4fr)_48px] items-center gap-3"
                      >
                        <span className="text-ui-11p5 text-muted-foreground">
                          {ZONE_LABELS[zone]}
                        </span>
                        <Slider
                          value={[advanced.layerLrZones[zone] ?? 1]}
                          min={0}
                          max={2}
                          step={0.05}
                          onValueChange={([value]) =>
                            setAdvanced({
                              layerLrZones: {
                                ...advanced.layerLrZones,
                                [zone]: value,
                              },
                            })
                          }
                          aria-label={`${ZONE_LABELS[zone]} learning-rate multiplier`}
                        />
                        <span className="text-right font-mono text-ui-11 text-foreground">
                          {(advanced.layerLrZones[zone] ?? 1).toFixed(2)}×
                        </span>
                      </div>
                    ))}
                    <div className="mt-1 flex items-center gap-3 border-t border-border/50 pt-3">
                      <Switch
                        size="sm"
                        checked={advanced.layerLrDepthCurveEnabled}
                        onCheckedChange={(layerLrDepthCurveEnabled) =>
                          setAdvanced({ layerLrDepthCurveEnabled })
                        }
                      />
                      <span className="text-ui-11p5 text-muted-foreground">
                        Smooth depth curve
                      </span>
                      {advanced.layerLrDepthCurveEnabled ? (
                        <>
                          <NumberField
                            value={advanced.layerLrDepthStart}
                            min={0}
                            max={3}
                            step={0.05}
                            onChange={(layerLrDepthStart) =>
                              setAdvanced({ layerLrDepthStart })
                            }
                            suffix="start"
                          />
                          <NumberField
                            value={advanced.layerLrDepthEnd}
                            min={0}
                            max={3}
                            step={0.05}
                            onChange={(layerLrDepthEnd) =>
                              setAdvanced({ layerLrDepthEnd })
                            }
                            suffix="end"
                          />
                        </>
                      ) : null}
                    </div>
                    <div className="flex flex-col gap-2 border-t border-border/50 pt-3">
                      <div>
                        <p className="text-ui-11p5 font-medium text-foreground">
                          Module groups
                        </p>
                        <p className="mt-0.5 text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                          A second axis by weight type. Multiplies with the zone
                          above, so 0.5 early attention is 0.25× overall.
                        </p>
                      </div>
                      {LAYER_PARAM_KINDS.map((kind) => (
                        <div
                          key={kind}
                          className="grid grid-cols-[minmax(104px,0.8fr)_minmax(0,1.4fr)_48px] items-center gap-3"
                        >
                          <span className="text-ui-11p5 text-muted-foreground">
                            {KIND_LABELS[kind]}
                          </span>
                          <Slider
                            value={[
                              (advanced.layerLrKinds ??
                                DEFAULT_LAYER_KIND_MULTIPLIERS)[kind] ?? 1,
                            ]}
                            min={0}
                            max={2}
                            step={0.05}
                            onValueChange={([value]) =>
                              setAdvanced({
                                layerLrKinds: {
                                  ...(advanced.layerLrKinds ??
                                    DEFAULT_LAYER_KIND_MULTIPLIERS),
                                  [kind]: value,
                                },
                              })
                            }
                            aria-label={`${KIND_LABELS[kind]} learning-rate multiplier`}
                          />
                          <span className="text-right font-mono text-ui-11 text-foreground">
                            {(
                              (advanced.layerLrKinds ??
                                DEFAULT_LAYER_KIND_MULTIPLIERS)[kind] ?? 1
                            ).toFixed(2)}
                            ×
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="h-px bg-border/60" />

              <div className="flex flex-col gap-3">
                <Row
                  title="Dynamic dataset blending"
                  description="Queue several dataset streams and interleave them live by percentage instead of pre-merging files."
                  checked={advanced.blendEnabled}
                  onCheckedChange={(blendEnabled) =>
                    setAdvanced({
                      blendEnabled,
                      blendSlots:
                        blendEnabled && advanced.blendSlots.length === 0
                          ? [newBlendSlot(1), newBlendSlot(2)]
                          : advanced.blendSlots,
                    })
                  }
                />
                {advanced.blendEnabled ? (
                  <div className="flex flex-col gap-2">
                    {advanced.blendSlots.map((slot) => {
                      const percent =
                        totalRatio > 0 ? (slot.ratio / totalRatio) * 100 : 0;
                      return (
                        <div
                          key={slot.id}
                          className="rounded-xl border border-border/60 p-3"
                        >
                          <div className="flex items-center gap-2">
                            <Input
                              value={slot.name}
                              onChange={(event) =>
                                setAdvanced({
                                  blendSlots: advanced.blendSlots.map((item) =>
                                    item.id === slot.id
                                      ? { ...item, name: event.target.value }
                                      : item,
                                  ),
                                })
                              }
                              className="h-8 w-36 text-ui-12"
                              aria-label="Dataset stream name"
                            />
                            <Input
                              value={slot.dataset}
                              placeholder={
                                slot.source === "local"
                                  ? "C:\\datasets\\reasoning.jsonl"
                                  : "org/dataset"
                              }
                              onChange={(event) =>
                                setAdvanced({
                                  blendSlots: advanced.blendSlots.map((item) =>
                                    item.id === slot.id
                                      ? { ...item, dataset: event.target.value }
                                      : item,
                                  ),
                                })
                              }
                              className="h-8 min-w-0 flex-1 font-mono text-ui-11p5"
                            />
                            <Select
                              value={slot.source}
                              onValueChange={(value) =>
                                setAdvanced({
                                  blendSlots: advanced.blendSlots.map((item) =>
                                    item.id === slot.id
                                      ? {
                                          ...item,
                                          source: value as
                                            | "huggingface"
                                            | "local",
                                        }
                                      : item,
                                  ),
                                })
                              }
                            >
                              <SelectTrigger className="h-8 w-32">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="huggingface">Hub</SelectItem>
                                <SelectItem value="local">
                                  Local file
                                </SelectItem>
                              </SelectContent>
                            </Select>
                            <Button
                              variant="ghost"
                              size="icon-sm"
                              aria-label={`Remove ${slot.name}`}
                              onClick={() =>
                                setAdvanced({
                                  blendSlots: advanced.blendSlots.filter(
                                    (item) => item.id !== slot.id,
                                  ),
                                })
                              }
                            >
                              <HugeiconsIcon
                                icon={Delete02Icon}
                                className="size-3.5"
                              />
                            </Button>
                          </div>
                          <div className="mt-2 flex items-center gap-3">
                            <Slider
                              value={[slot.ratio]}
                              min={0}
                              max={10}
                              step={0.1}
                              onValueChange={([value]) =>
                                setAdvanced({
                                  blendSlots: advanced.blendSlots.map((item) =>
                                    item.id === slot.id
                                      ? { ...item, ratio: value }
                                      : item,
                                  ),
                                })
                              }
                              aria-label={`${slot.name} ratio`}
                            />
                            <span className="w-12 text-right font-mono text-ui-11 text-foreground">
                              {percent.toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      );
                    })}
                    <Button
                      variant="outline"
                      size="sm"
                      className="self-start"
                      onClick={() =>
                        setAdvanced({
                          blendSlots: [
                            ...advanced.blendSlots,
                            newBlendSlot(advanced.blendSlots.length + 1),
                          ],
                        })
                      }
                    >
                      <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                      Add dataset stream
                    </Button>
                    <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                      Streams are concatenated and sampled per optimizer step
                      using these weights. Hub streams are downloaded and
                      formatted at start.
                    </p>
                    {advanced.blendSlots.length > 0 &&
                    advanced.blendSlots.every(
                      (slot) => !slot.dataset.trim(),
                    ) ? (
                      <p className="text-ui-10p5 text-amber-600 dark:text-amber-400">
                        Every stream needs a dataset id or local path.
                      </p>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="hooks">
          <AccordionTrigger className="px-3 text-ui-12p5">
            Post-run hooks
          </AccordionTrigger>
          <AccordionContent className="h-auto! px-3 pb-5">
            <div className="flex flex-col gap-4">
              <Row
                title="Run Regression Guard on completion"
                description="Scores the finished checkpoint against its baseline and reports the per-benchmark delta. Deltas under ~1.5 points on a 1.5B model are labelled within noise, not as wins."
                checked={advanced.autoEvalEnabled}
                onCheckedChange={(autoEvalEnabled) =>
                  setAdvanced({ autoEvalEnabled })
                }
              />
              {advanced.autoEvalEnabled ? (
                <div className="flex flex-col gap-3">
                  <div className="flex flex-wrap gap-2">
                    {EVAL_SUITES.map((suite) => {
                      const selected = advanced.autoEvalSuites.includes(
                        suite.id,
                      );
                      return (
                        <button
                          key={suite.id}
                          type="button"
                          onClick={() =>
                            setAdvanced({
                              autoEvalSuites: selected
                                ? advanced.autoEvalSuites.filter(
                                    (id) => id !== suite.id,
                                  )
                                : [...advanced.autoEvalSuites, suite.id],
                            })
                          }
                          className={`rounded-full border px-2.5 py-1 text-ui-11 transition-colors ${
                            selected
                              ? "border-foreground/30 bg-foreground/10 text-foreground"
                              : "border-border/60 text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {suite.label}
                        </button>
                      );
                    })}
                  </div>
                  <Field
                    label="Examples per suite"
                    hint="A low limit keeps the post-run check to minutes; clear for a full run."
                  >
                    <NumberField
                      value={advanced.autoEvalLimit ?? 0}
                      min={0}
                      max={100000}
                      onChange={(value) =>
                        setAdvanced({ autoEvalLimit: value > 0 ? value : null })
                      }
                      suffix="rows"
                    />
                  </Field>
                  <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                    Requires lm-evaluation-harness in the Studio environment. If
                    it is missing the run reports an install hint instead of
                    failing.
                  </p>
                </div>
              ) : null}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {isPolicyAlignment ? <PolicyAlignmentPanel /> : null}
    </section>
  );
}

function PlanBadge({ cadence }: { cadence: string }) {
  const stages = cadence
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [name, raw] = part.includes(":")
        ? [
            part.slice(0, part.lastIndexOf(":")),
            part.slice(part.lastIndexOf(":") + 1),
          ]
        : ["stage", part];
      return { name: name.trim(), steps: Number(raw) };
    });
  const cycle = stages.reduce(
    (sum, stage) => sum + (Number.isFinite(stage.steps) ? stage.steps : 0),
    0,
  );
  if (cycle <= 0) {
    return (
      <span className="text-ui-10p5 text-destructive">Invalid cadence</span>
    );
  }
  return (
    <span className="text-ui-10p5 text-muted-foreground">
      {stages.map((stage) => `${stage.name}×${stage.steps}`).join(" → ")} ·{" "}
      {cycle} steps/cycle
    </span>
  );
}

function PolicyAlignmentPanel() {
  const advanced = useTrainingConfigStore((s) => s.advanced);
  const setAdvanced = useTrainingConfigStore((s) => s.setAdvanced);
  return (
    <div className="rounded-2xl border border-rose-500/25 bg-rose-500/5 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-ui-12p5 font-semibold text-foreground">
            Policy Alignment lab
          </h3>
          <p className="text-ui-11 leading-ui-15 text-muted-foreground/85">
            DPO learns from chosen/rejected pairs; GRPO scores a group of
            samples against a reward. Dataset columns are validated before the
            run starts.
          </p>
        </div>
        <div className="flex gap-1 rounded-full border border-border/60 p-0.5">
          {RL_METHODS.map((method) => (
            <button
              key={method}
              type="button"
              onClick={() => setAdvanced({ rlMethod: method as RlMethod })}
              className={`rounded-full px-3 py-1 text-ui-11 uppercase tracking-wide transition-colors ${
                advanced.rlMethod === method
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {method}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <Field
          label={advanced.rlMethod === "dpo" ? "Beta" : "KL coefficient"}
          hint={
            advanced.rlMethod === "dpo"
              ? "How strongly the policy is pulled toward the reference."
              : "Penalty against drifting from the reference (0 = off)."
          }
        >
          <NumberField
            value={
              advanced.rlMethod === "dpo"
                ? advanced.rlBeta
                : advanced.rlKlCoefficient
            }
            min={0}
            max={1}
            step={0.01}
            onChange={(value) =>
              setAdvanced(
                advanced.rlMethod === "dpo"
                  ? { rlBeta: value }
                  : { rlKlCoefficient: value },
              )
            }
          />
        </Field>
        <Field label="Learning rate">
          <Input
            type="number"
            value={advanced.rlLearningRate}
            min={1e-9}
            max={0.01}
            step="any"
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isFinite(value)) {
                setAdvanced({ rlLearningRate: value });
              }
            }}
            className="h-8 w-32"
          />
        </Field>
        {advanced.rlMethod === "grpo" ? (
          <>
            <Field
              label="Generations per prompt"
              hint="Group size used for the relative baseline."
            >
              <NumberField
                value={advanced.rlNumGenerations}
                min={2}
                max={64}
                onChange={(rlNumGenerations) =>
                  setAdvanced({ rlNumGenerations })
                }
              />
            </Field>
            <Field label="Reward">
              <Select
                value={advanced.rlRewardPreset}
                onValueChange={(value) =>
                  setAdvanced({ rlRewardPreset: value as RewardPreset })
                }
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {REWARD_PRESETS.map((preset) => (
                    <SelectItem key={preset} value={preset}>
                      {preset.replaceAll("_", " ")}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </>
        ) : null}
        <Field label="Max completion tokens">
          <NumberField
            value={advanced.rlMaxCompletionLength}
            min={16}
            max={8192}
            onChange={(rlMaxCompletionLength) =>
              setAdvanced({ rlMaxCompletionLength })
            }
          />
        </Field>
        <Field label="Temperature">
          <NumberField
            value={advanced.rlTemperature}
            min={0.1}
            max={2}
            step={0.05}
            onChange={(rlTemperature) => setAdvanced({ rlTemperature })}
          />
        </Field>
      </div>
    </div>
  );
}

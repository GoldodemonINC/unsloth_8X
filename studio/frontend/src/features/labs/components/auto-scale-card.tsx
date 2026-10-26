// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { fetchJob } from "@/features/advanced/api/advanced-api";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowExpand02Icon,
  PlayIcon,
  Refresh01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import {
  type AutoScalePlan,
  type AutoScaleResult,
  type AutoScaleSpec,
  planAutoScale,
  runAutoScale,
} from "../api";

const INPUT_CLASS =
  "h-8 min-w-0 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

function pathsFrom(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "");
}

/**
 * The one-step version of the Scaler below: it sizes the growth from the
 * datasets the run will train on, instead of a slider position. CMT does the
 * same sizing inside its own card and then trains the new blocks; for every
 * other method this card grows the checkpoint and hands it back as the model.
 */
export function AutoScaleCard({
  defaultModel = "",
}: {
  defaultModel?: string;
}) {
  const t = useT();
  const adoptTrainingModel = useTrainingConfigStore(
    (state) => state.selectTrainingModel,
  );
  const [model, setModel] = useState(defaultModel);
  const [datasets, setDatasets] = useState("");
  const [added, setAdded] = useState("");
  const [plan, setPlan] = useState<AutoScalePlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [result, setResult] = useState<AutoScaleResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Follow the wizard's model until the user types their own base.
  const previousDefault = useRef(defaultModel);
  useEffect(() => {
    setModel((current) =>
      current === "" || current === previousDefault.current
        ? defaultModel
        : current,
    );
    previousDefault.current = defaultModel;
  }, [defaultModel]);

  const buildSpec = (): AutoScaleSpec => {
    const paths = pathsFrom(datasets);
    const override = added.trim() === "" ? null : Number(added);
    return {
      model: model.trim(),
      datasets: paths.map((path) => ({ path })),
      method: "depth",
      added_params: Number.isFinite(override) ? override : null,
    };
  };

  const makePlan = async () => {
    const spec = buildSpec();
    if (!spec.model) {
      setError(t("labs.autoScale.needModel"));
      return;
    }
    setPlanning(true);
    setError(null);
    setResult(null);
    try {
      setPlan(await planAutoScale(spec));
    } catch (failure) {
      setPlan(null);
      setError(
        failure instanceof Error
          ? failure.message
          : t("labs.autoScale.planFailed"),
      );
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    if (!plan?.runnable) return;
    setRunning(true);
    setError(null);
    try {
      const job = await runAutoScale(buildSpec());
      for (;;) {
        const current = await fetchJob(job.job_id);
        setPhase(
          String(
            current.progress?.message ??
              current.progress?.stage ??
              t("labs.autoScale.running"),
          ),
        );
        if (current.status === "completed") {
          setResult((current.result ?? {}) as unknown as AutoScaleResult);
          toast.success(t("labs.autoScale.completed"));
          break;
        }
        if (current.status === "failed") {
          throw new Error(
            current.error || t("labs.autoScale.failed"),
          );
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch (failure) {
      const message =
        failure instanceof Error ? failure.message : t("labs.autoScale.failed");
      setError(message);
      toast.error(t("labs.autoScale.failed"), { description: message });
    } finally {
      setRunning(false);
      setPhase(null);
    }
  };

  const useCheckpoint = () => {
    if (!result?.checkpoint) return;
    adoptTrainingModel(result.checkpoint, null, {
      localPath: result.checkpoint,
    });
    toast.success(t("labs.autoScale.adopted"));
  };

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-border/60 bg-muted/20 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-ui-12 font-semibold text-foreground">
          {t("labs.autoScale.title")}
        </h3>
        <Badge variant="outline" className="text-ui-9">
          {t("labs.experimental")}
        </Badge>
      </div>
      <p className="max-w-3xl text-ui-10p5 leading-relaxed text-muted-foreground">
        {t("labs.autoScale.description")}
      </p>

      <div className="grid gap-3 @2xl/train-card:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <label className="flex min-w-0 flex-col gap-1.5">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.autoScale.model")}
          </span>
          <input
            className={INPUT_CLASS}
            value={model}
            onChange={(event) => {
              setModel(event.target.value);
              setPlan(null);
            }}
            placeholder={t("labs.autoScale.modelPlaceholder")}
          />
        </label>
        <label className="flex min-w-0 flex-col gap-1.5">
          <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
            {t("labs.autoScale.added")}
          </span>
          <input
            className={INPUT_CLASS}
            value={added}
            inputMode="numeric"
            onChange={(event) => {
              setAdded(event.target.value);
              setPlan(null);
            }}
            placeholder={t("labs.autoScale.addedPlaceholder")}
          />
        </label>
      </div>

      <label className="flex min-w-0 flex-col gap-1.5">
        <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
          {t("labs.autoScale.datasets")}
        </span>
        <textarea
          className="min-h-[72px] w-full resize-y rounded-lg border border-border bg-background px-2 py-1.5 font-mono text-ui-11 text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
          value={datasets}
          onChange={(event) => {
            setDatasets(event.target.value);
            setPlan(null);
          }}
          placeholder={t("labs.autoScale.datasetsPlaceholder")}
        />
        <span className="text-ui-10 text-muted-foreground/70">
          {t("labs.autoScale.datasetsHint")}
        </span>
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => void makePlan()}
          disabled={planning || running}
        >
          {planning ? <Spinner /> : null}
          <HugeiconsIcon icon={ArrowExpand02Icon} className="size-3.5" />
          {planning ? t("labs.autoScale.planning") : t("labs.autoScale.plan")}
        </Button>
        {plan?.runnable ? (
          <Button size="sm" onClick={() => void start()} disabled={running}>
            {running ? <Spinner /> : null}
            <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
            {running ? t("labs.autoScale.running") : t("labs.autoScale.run")}
          </Button>
        ) : null}
        {result ? (
          <Button size="sm" variant="outline" onClick={useCheckpoint}>
            <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
            {t("labs.autoScale.use")}
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            setPlan(null);
            setResult(null);
            setError(null);
          }}
        >
          <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
          {t("labs.autoScale.clear")}
        </Button>
      </div>

      {phase ? (
        <p className="text-ui-11 text-muted-foreground">{phase}</p>
      ) : null}

      {plan ? (
        <div className="flex flex-col gap-2">
          <div className="grid gap-2 text-ui-11 sm:grid-cols-3">
            <div className="rounded-lg border border-border/50 px-2.5 py-2">
              <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                {t("labs.autoScale.base")}
              </div>
              <div className="font-medium text-foreground">
                {plan.expansion.base_label}
              </div>
            </div>
            <div className="rounded-lg border border-border/50 px-2.5 py-2">
              <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                {t("labs.autoScale.addedLabel")}
              </div>
              <div className="font-medium text-foreground">
                {plan.expansion.added_label}
              </div>
            </div>
            <div className="rounded-lg border border-border/50 px-2.5 py-2">
              <div className="text-ui-9 uppercase tracking-[0.05em] text-muted-foreground">
                {t("labs.autoScale.target")}
              </div>
              <div className="font-medium text-foreground">
                {plan.expansion.target_label}
              </div>
            </div>
          </div>
          <p className="text-ui-10p5 leading-relaxed text-muted-foreground">
            {t("labs.autoScale.rule", {
              rule: plan.sizing.source,
              rows: String(plan.sizing.rows),
              tokens: String(plan.sizing.tokens),
            })}
          </p>
          {plan.blocked_reason ? (
            <p className="flex items-start gap-1.5 text-ui-10p5 leading-relaxed text-destructive">
              <HugeiconsIcon
                icon={Alert02Icon}
                className="mt-0.5 size-3 shrink-0"
              />
              {plan.blocked_reason}
            </p>
          ) : null}
          {[...plan.warnings, ...plan.notes].map((note) => (
            <p
              key={note}
              className={cn(
                "text-ui-10p5 leading-relaxed",
                plan.warnings.includes(note)
                  ? "text-status-warning"
                  : "text-muted-foreground/80",
              )}
            >
              {note}
            </p>
          ))}
          <p className="text-ui-10p5 leading-relaxed text-muted-foreground">
            {t("labs.autoScale.trainingNote")}
          </p>
        </div>
      ) : null}

      {result ? (
        <p className="text-ui-10p5 leading-relaxed text-muted-foreground">
          {t("labs.autoScale.result", {
            path: result.checkpoint,
            layers: String(result.new_layers),
          })}
        </p>
      ) : null}

      {error ? (
        <p className="text-ui-10p5 leading-relaxed text-destructive">{error}</p>
      ) : null}
    </section>
  );
}

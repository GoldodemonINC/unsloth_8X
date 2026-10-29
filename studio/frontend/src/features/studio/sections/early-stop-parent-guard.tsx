import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { getTrainingRunDisplayTitle, useTrainingConfigStore } from "@/features/training";
import { listTrainingRuns } from "@/features/training/api/history-api";
import type { TrainingRunSummary } from "@/features/training/types/history";
import { useT } from "@/i18n";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow } from "./params-section-controls";

const DEFAULT_PATIENCE_STEPS = 50;

export function EarlyStopParentGuard(): ReactElement {
  const t = useT();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      parentRunId: state.earlyStopParentRunId,
      patienceSteps: state.earlyStopPatienceSteps,
      minDelta: state.earlyStopMinDelta,
      setParentRunId: state.setEarlyStopParentRunId,
      setPatienceSteps: state.setEarlyStopPatienceSteps,
      setMinDelta: state.setEarlyStopMinDelta,
    })),
  );
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    listTrainingRuns(100, 0, controller.signal)
      .then((response) => {
        setRuns(response.runs);
        setError(false);
      })
      .catch((requestError: unknown) => {
        if (!(requestError instanceof DOMException && requestError.name === "AbortError")) {
          setError(true);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const eligibleRuns = useMemo(
    () =>
      runs.filter(
        (run) => run.status !== "running" && run.final_loss != null,
      ),
    [runs],
  );
  const parentRun = eligibleRuns.find((run) => run.id === store.parentRunId) ?? null;
  const enabled = store.parentRunId !== null && store.patienceSteps > 0;

  const setEnabled = (next: boolean) => {
    if (!next) {
      store.setParentRunId(null);
      return;
    }
    const chosen = store.parentRunId ?? eligibleRuns[0]?.id ?? null;
    if (!chosen) return;
    store.setParentRunId(chosen);
    if (store.patienceSteps <= 0) {
      store.setPatienceSteps(DEFAULT_PATIENCE_STEPS);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-muted-foreground">
            {t("studio.params.earlyStopParentTitle")}
          </span>
          <span className="text-ui-11 leading-relaxed text-muted-foreground">
            {enabled && parentRun
              ? t("studio.params.earlyStopParentSummary", {
                  parent: getTrainingRunDisplayTitle(parentRun),
                  steps: store.patienceSteps,
                  delta: store.minDelta,
                })
              : t("studio.params.earlyStopParentHint")}
          </span>
        </div>
        <Switch
          aria-label={t("studio.params.earlyStopParentTitle")}
          checked={enabled}
          disabled={loading || eligibleRuns.length === 0}
          onCheckedChange={setEnabled}
        />
      </div>
      {loading ? (
        <div className="flex items-center gap-2 text-ui-11 text-muted-foreground">
          <Spinner className="size-3.5" />
          {t("studio.params.earlyStopParentLoading")}
        </div>
      ) : error ? (
        <p role="alert" className="text-ui-11 text-destructive">
          {t("studio.params.earlyStopParentLoadError")}
        </p>
      ) : eligibleRuns.length === 0 ? (
        <p className="text-ui-11 text-muted-foreground">
          {t("studio.params.earlyStopParentNoRuns")}
        </p>
      ) : enabled ? (
        <>
          <ParamsRow label={t("studio.params.earlyStopParentRun")}>
            <Select
              value={store.parentRunId ?? undefined}
              onValueChange={store.setParentRunId}
            >
              <SelectTrigger className="w-full max-w-[calc(280px*var(--ui-space-scale,1))]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {eligibleRuns.map((run) => (
                  <SelectItem key={run.id} value={run.id}>
                    {getTrainingRunDisplayTitle(run)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </ParamsRow>
          <ParamsRow
            label={t("studio.params.earlyStopPatienceSteps")}
            tooltip={t("studio.params.earlyStopPatienceStepsTooltip")}
          >
            <Input
              type="number"
              step="1"
              min="1"
              max="100000"
              value={store.patienceSteps}
              onChange={(event) =>
                store.setPatienceSteps(Number(event.target.value))
              }
              className="w-28 font-mono"
            />
          </ParamsRow>
          <ParamsRow
            label={t("studio.params.earlyStopMinDelta")}
            tooltip={t("studio.params.earlyStopMinDeltaTooltip")}
          >
            <Input
              type="number"
              step="0.001"
              min="0"
              value={store.minDelta}
              onChange={(event) =>
                store.setMinDelta(Number(event.target.value))
              }
              className="w-28 font-mono"
            />
          </ParamsRow>
        </>
      ) : null}
    </div>
  );
}

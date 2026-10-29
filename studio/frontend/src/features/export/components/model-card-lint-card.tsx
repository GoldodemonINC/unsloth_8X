import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import {
  type ModelCardCheckId,
  type ModelCardLintReport,
  lintCheckpointModelCard,
} from "@/features/training/api/model-card-api";
import { type TranslationKey, useT } from "@/i18n";
import { type ReactElement, useEffect, useState } from "react";

const CHECK_LABELS: Record<ModelCardCheckId, TranslationKey> = {
  base_model: "studio.exportModelCard.baseModel",
  license: "studio.exportModelCard.license",
  dataset: "studio.exportModelCard.dataset",
  evaluation: "studio.exportModelCard.evaluation",
};

export function ModelCardLintCard({
  path,
}: {
  path: string | null;
}): ReactElement | null {
  const t = useT();
  const [report, setReport] = useState<ModelCardLintReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!path) {
      setReport(null);
      setError(false);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(false);
    lintCheckpointModelCard(path, controller.signal)
      .then((result) => {
        setReport(result);
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
  }, [path]);

  if (!path) return null;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border/60 bg-muted/20 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium">
            {t("studio.exportModelCard.title")}
          </span>
          <span className="text-ui-11 leading-relaxed text-muted-foreground">
            {t("studio.exportModelCard.description")}
          </span>
        </div>
        {report && report.card_found ? (
          <Badge variant={report.ok ? "outline" : "secondary"}>
            {report.ok
              ? t("studio.exportModelCard.allPresent")
              : t("studio.exportModelCard.missingCount", {
                  count: report.missing.length,
                })}
          </Badge>
        ) : null}
      </div>
      {loading ? (
        <div className="flex items-center gap-2 text-ui-11 text-muted-foreground">
          <Spinner className="size-3.5" />
          {t("studio.exportModelCard.loading")}
        </div>
      ) : error ? (
        <p role="alert" className="text-ui-11 text-destructive">
          {t("studio.exportModelCard.error")}
        </p>
      ) : report ? (
        <>
          {!report.card_found ? (
            <p className="text-ui-11 leading-relaxed text-muted-foreground">
              {t("studio.exportModelCard.noCard")}
            </p>
          ) : null}
          <ul className="flex flex-col gap-1.5">
            {report.checks.map((check) => (
              <li
                key={check.id}
                className="grid grid-cols-[auto_1fr] items-start gap-2"
              >
                <Badge
                  variant={check.status === "present" ? "outline" : "secondary"}
                  className="mt-0.5"
                >
                  {check.status === "present"
                    ? t("studio.exportModelCard.present")
                    : t("studio.exportModelCard.missing")}
                </Badge>
                <span className="min-w-0">
                  <span className="block text-ui-11 font-medium text-foreground">
                    {t(CHECK_LABELS[check.id])}
                  </span>
                  {check.evidence ? (
                    <span
                      className="block truncate font-mono text-ui-10 text-muted-foreground"
                      title={check.evidence}
                    >
                      {check.evidence}
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
          {report.missing.length > 0 ? (
            <p className="text-ui-10 text-muted-foreground">
              {t("studio.exportModelCard.addHint")}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

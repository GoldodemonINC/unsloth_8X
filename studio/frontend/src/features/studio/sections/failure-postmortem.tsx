// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  type TranslationKey,
  useT,
} from "@/i18n";
import { diagnoseTrainingFailure } from "@/features/training/lib/training-failure-postmortem";
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

const COPY: Record<
  NonNullable<ReturnType<typeof diagnoseTrainingFailure>>["kind"],
  { title: TranslationKey; fix: TranslationKey }
> = {
  oom: {
    title: "studio.postmortem.oomTitle",
    fix: "studio.postmortem.oomFix",
  },
  nan: {
    title: "studio.postmortem.nanTitle",
    fix: "studio.postmortem.nanFix",
  },
  disk: {
    title: "studio.postmortem.diskTitle",
    fix: "studio.postmortem.diskFix",
  },
  auth: {
    title: "studio.postmortem.authTitle",
    fix: "studio.postmortem.authFix",
  },
  dataset: {
    title: "studio.postmortem.datasetTitle",
    fix: "studio.postmortem.datasetFix",
  },
  timeout: {
    title: "studio.postmortem.timeoutTitle",
    fix: "studio.postmortem.timeoutFix",
  },
  unknown: {
    title: "studio.postmortem.unknownTitle",
    fix: "studio.postmortem.unknownFix",
  },
};

export function FailurePostmortem({
  message,
  onOpenWizard,
  wizardDisabled = false,
}: {
  message: string | null | undefined;
  onOpenWizard?: () => void;
  wizardDisabled?: boolean;
}): ReactElement | null {
  const t = useT();
  const diagnosis = diagnoseTrainingFailure(message);
  if (!diagnosis) {
    return null;
  }
  const copy = COPY[diagnosis.kind];

  return (
    <section
      aria-labelledby="training-postmortem-title"
      className="flex flex-col gap-2.5 rounded-xl border border-destructive/30 bg-destructive/5 px-3.5 py-3"
    >
      <div className="flex items-start gap-2.5">
        <HugeiconsIcon
          icon={Alert02Icon}
          strokeWidth={1.75}
          className="mt-0.5 size-4 shrink-0 text-destructive"
        />
        <div className="min-w-0 flex-1 space-y-0.5">
          <p
            id="training-postmortem-title"
            className="text-ui-12 font-medium text-foreground"
          >
            {t("studio.postmortem.title")}: {t(copy.title)}
          </p>
          <p className="text-ui-11 leading-relaxed text-muted-foreground">
            {t(copy.fix)}
          </p>
        </div>
        {onOpenWizard && (
          <Button
            type="button"
            size="xs"
            variant="outline"
            disabled={wizardDisabled}
            onClick={onOpenWizard}
          >
            {t("studio.postmortem.openWizard")}
          </Button>
        )}
      </div>
      <details className="group">
        <summary className="w-fit cursor-pointer select-none text-ui-10 text-muted-foreground hover:text-foreground">
          {t("studio.postmortem.logLabel")}
        </summary>
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-background/60 p-2.5 font-mono text-ui-10 leading-relaxed text-muted-foreground">
          {message}
        </pre>
      </details>
    </section>
  );
}

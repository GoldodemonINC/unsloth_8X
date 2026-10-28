// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { type LineageNode, type LineageReport, loadLineage } from "../api";
import { LabsPanel } from "./labs-panel";

interface ChildLink {
  node: LineageNode;
  label: string;
}

function Branch({
  node,
  byId,
  childrenOf,
  visited,
}: {
  node: LineageNode;
  byId: ReadonlyMap<string, LineageNode>;
  childrenOf: ReadonlyMap<string, ChildLink[]>;
  visited: ReadonlySet<string>;
}): ReactElement {
  const t = useT();
  const links = (childrenOf.get(node.id) ?? []).filter(
    (link) => !visited.has(link.node.id),
  );
  const nextVisited = new Set(visited);
  nextVisited.add(node.id);

  return (
    <li className="relative">
      <article
        className={cn(
          "rounded-xl border bg-background/60 px-3 py-2.5",
          node.valid ? "border-border/60" : "border-destructive/40",
        )}
      >
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            className="shrink-0 text-ui-9 uppercase tracking-wide"
          >
            {node.stage}
          </Badge>
          <span className="min-w-0 truncate font-mono text-ui-11 text-foreground">
            {node.label}
          </span>
          <span className="shrink-0 text-ui-10 text-muted-foreground">
            {node.created_at.slice(0, 10)}
          </span>
        </div>
        <p className="mt-1 truncate font-mono text-ui-10 text-muted-foreground">
          {node.id}
          {node.studio_version ? ` · ${node.studio_version}` : null}
        </p>
        {node.problems.length > 0 ? (
          <p className="mt-1 text-ui-10 leading-snug text-destructive">
            {t("labs.lineage.incomplete", { count: node.problems.length })}
          </p>
        ) : null}
      </article>

      {links.length > 0 ? (
        <ul className="ml-4 mt-2 flex list-none flex-col gap-2 border-l border-border/60 pl-4">
          {links.map((link) => {
            const child = byId.get(link.node.id);
            if (!child) return null;
            return (
              <li key={`${link.label}-${link.node.id}`} className="relative">
                <span className="absolute -left-4 top-4 w-3 border-t border-border/60" />
                <span className="mb-1 block text-ui-9 uppercase tracking-wide text-muted-foreground/70">
                  {link.label}
                </span>
                <Branch
                  node={child}
                  byId={byId}
                  childrenOf={childrenOf}
                  visited={nextVisited}
                />
              </li>
            );
          })}
        </ul>
      ) : null}
    </li>
  );
}

export function LineagePanel(): ReactElement {
  const t = useT();
  const [report, setReport] = useState<LineageReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const next = await loadLineage(controller.signal);
      if (!controller.signal.aborted) setReport(next);
    } catch (failure) {
      if (controller.signal.aborted) return;
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => controllerRef.current?.abort();
  }, [refresh]);

  const graph = useMemo(() => {
    if (!report) return null;
    const byId = new Map(report.nodes.map((node) => [node.id, node]));
    const childrenOf = new Map<string, ChildLink[]>();
    for (const edge of report.edges) {
      const child = byId.get(edge.to);
      if (!child) continue;
      const list = childrenOf.get(edge.from) ?? [];
      list.push({ node: child, label: edge.label });
      childrenOf.set(edge.from, list);
    }
    const roots = report.roots
      .map((id) => byId.get(id))
      .filter((node): node is LineageNode => Boolean(node));
    // A cycle has no root; showing its nodes beats showing an empty panel.
    const shown = new Set(roots.map((node) => node.id));
    const extra = report.nodes.filter((node) => !shown.has(node.id));
    return { byId, childrenOf, roots, extra };
  }, [report]);

  return (
    <LabsPanel
      title={t("labs.lineage.title")}
      description={t("labs.lineage.description")}
      onRefresh={() => void refresh()}
      refreshing={loading}
    >
      {error ? <p className="text-ui-11 text-destructive">{error}</p> : null}

      {!error && report && report.nodes.length === 0 ? (
        <div className="space-y-1">
          <p className="text-ui-12 font-medium text-foreground">
            {t("labs.lineage.empty")}
          </p>
          <p className="max-w-2xl text-ui-11 leading-relaxed text-muted-foreground">
            {t("labs.lineage.emptyHint")}
          </p>
        </div>
      ) : null}

      {graph && report && report.nodes.length > 0 ? (
        <div className="space-y-3">
          <p className="text-ui-10 text-muted-foreground">
            {t("labs.lineage.counts", {
              runs: report.nodes.length,
              links: report.edges.length,
            })}
          </p>
          {report.truncated ? (
            <p className="text-ui-10 text-amber-600 dark:text-amber-400">
              {t("labs.lineage.truncated")}
            </p>
          ) : null}
          <ul className="flex list-none flex-col gap-3">
            {graph.roots.map((node) => (
              <Branch
                key={node.id}
                node={node}
                byId={graph.byId}
                childrenOf={graph.childrenOf}
                visited={new Set()}
              />
            ))}
          </ul>
          {graph.extra.length > 0 ? (
            <details className="text-ui-11 text-muted-foreground">
              <summary className="cursor-pointer hover:text-foreground">
                {t("labs.lineage.unlinked", { count: graph.extra.length })}
              </summary>
              <ul className="mt-2 flex list-none flex-col gap-2">
                {graph.extra.map((node) => (
                  <Branch
                    key={node.id}
                    node={node}
                    byId={graph.byId}
                    childrenOf={graph.childrenOf}
                    visited={new Set()}
                  />
                ))}
              </ul>
            </details>
          ) : null}
        </div>
      ) : null}

      {!error && !report && loading ? (
        <div className="flex flex-col gap-2" aria-busy="true">
          <Skeleton className="h-16 rounded-xl" />
          <Skeleton className="ml-4 h-16 rounded-xl" />
          <Skeleton className="ml-4 h-16 rounded-xl" />
        </div>
      ) : null}
    </LabsPanel>
  );
}

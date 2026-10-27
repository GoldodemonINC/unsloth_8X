// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { RAISED_SURFACE } from "../surface";

const FADE_PX = 48;
const DRAG_SLOP_PX = 4;

export interface HeaderTab {
  key: string;
  label: string;
}

export type HeaderTabs = { items: HeaderTab[]; active: string; onChange: (key: string) => void };

function fadeMask(left: boolean, right: boolean): string | undefined {
  if (!left && !right) return undefined;
  const start = left ? `transparent 0, black ${FADE_PX}px` : "black 0";
  const end = right ? `black calc(100% - ${FADE_PX}px), transparent 100%` : "black 100%";
  return `linear-gradient(to right, ${start}, ${end})`;
}

function TabStrip({
  tabs,
  active,
  onChange,
}: {
  tabs: HeaderTab[];
  active: string;
  onChange: (key: string) => void;
}) {
  const t = useT();
  const scrollerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false });
  const drag = useRef<{ id: number; x: number; left: number; moved: boolean } | null>(null);
  const dragged = useRef(false);

  useEffect(() => {
    const scroller = scrollerRef.current;
    const list = listRef.current;
    if (!scroller || !list) return;
    const measure = () =>
      setEdges({
        left: scroller.scrollLeft > 1,
        right: scroller.scrollLeft + scroller.clientWidth < scroller.scrollWidth - 1,
      });
    const observer = new ResizeObserver(measure);
    observer.observe(scroller);
    observer.observe(list);
    scroller.addEventListener("scroll", measure, { passive: true });
    return () => {
      observer.disconnect();
      scroller.removeEventListener("scroll", measure);
    };
  }, []);

  useEffect(() => {
    const scroller = scrollerRef.current;
    const tab = scroller?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!scroller || !tab) return;
    const left = tab.offsetLeft - scroller.offsetLeft;
    const right = left + tab.offsetWidth;
    if (left < scroller.scrollLeft) scroller.scrollLeft = left - FADE_PX;
    else if (right > scroller.scrollLeft + scroller.clientWidth) {
      scroller.scrollLeft = right - scroller.clientWidth + FADE_PX;
    }
  }, [active]);

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    dragged.current = false;
    if (event.pointerType !== "mouse" || event.button !== 0 || !(edges.left || edges.right)) {
      return;
    }
    drag.current = {
      id: event.pointerId,
      x: event.clientX,
      left: event.currentTarget.scrollLeft,
      moved: false,
    };
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const current = drag.current;
    if (!current || current.id !== event.pointerId) return;
    // Released off the strip before a drag captured the pointer: the strip never saw the release,
    // and a hover must not scroll it.
    if ((event.buttons & 1) === 0) {
      drag.current = null;
      return;
    }
    const dx = event.clientX - current.x;
    if (!current.moved) {
      if (Math.abs(dx) < DRAG_SLOP_PX) return;
      current.moved = true;
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    event.currentTarget.scrollLeft = current.left - dx;
  }

  function onPointerEnd(event: ReactPointerEvent<HTMLDivElement>) {
    if (drag.current?.id !== event.pointerId) return;
    dragged.current = drag.current.moved;
    drag.current = null;
  }

  // The click that ends a drag must not also switch tab.
  function onClickCapture(event: ReactMouseEvent) {
    if (!dragged.current) return;
    dragged.current = false;
    event.preventDefault();
    event.stopPropagation();
  }

  const mask = fadeMask(edges.left, edges.right);
  return (
    <div
      ref={scrollerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onLostPointerCapture={onPointerEnd}
      onClickCapture={onClickCapture}
      // Important: index.css's unlayered `* { scrollbar-width: thin }` outranks any utility.
      className="-mx-2 -mb-3 -mt-1 overflow-x-auto px-2 pb-3 pt-1 [scrollbar-width:none]! [&::-webkit-scrollbar]:hidden!"
      style={{ maskImage: mask, WebkitMaskImage: mask }}
    >
      <nav ref={listRef} className="flex w-max gap-1" aria-label={t("library.tabs.ariaLabel")}>
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            aria-current={tab.key === active ? "page" : undefined}
            onClick={() => onChange(tab.key)}
            className={cn(
              "flex h-9 shrink-0 items-center rounded-full px-3.5 text-ui-15 text-foreground/80 transition-colors hover:text-foreground",
              tab.key === active && cn(RAISED_SURFACE, "font-medium text-foreground"),
            )}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </div>
  );
}

export function LibraryHeader({
  title,
  description,
  trail,
  toolbar,
  tabs,
}: {
  title: ReactNode;
  description?: ReactNode;
  /** Navigation that belongs under the title, e.g. an open folder's ancestors. */
  trail?: ReactNode;
  toolbar?: ReactNode;
  tabs?: HeaderTabs;
}) {
  const titleRef = useRef<HTMLDivElement>(null);
  const rowRef = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    const row = rowRef.current;
    const titleNode = titleRef.current;
    if (!row || !titleNode) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      // Stuck once the title block scrolls on past the row's margin.
      const gap =
        row.getBoundingClientRect().top - titleNode.getBoundingClientRect().bottom;
      setStuck(gap > Number.parseFloat(getComputedStyle(row).marginTop) + 1);
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    document.addEventListener("scroll", schedule, { capture: true, passive: true });
    window.addEventListener("resize", schedule);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("scroll", schedule, { capture: true });
      window.removeEventListener("resize", schedule);
    };
  }, []);

  return (
    <>
      {/* The page's own title block, as every other Studio page has it: the title, and its muted
          description under it. Nothing floats over it. */}
      <div
        ref={titleRef}
        className={cn(
          "font-heading flex min-w-0 flex-col gap-0.5",
          // With no tab row under it, the title block carries the separation the row would have.
          tabs ? undefined : "mb-6",
        )}
      >
        {title}
        {description ? (
          <p className="page-title-halo text-sm text-muted-foreground">
            {description}
          </p>
        ) : null}
        {trail ? (
          <div className="page-title-halo text-sm text-muted-foreground">{trail}</div>
        ) : null}
      </div>
      {/* No tabs, no row: an empty sticky band would push the toolbar down for nothing. */}
      {tabs ? (
        <div
          ref={rowRef}
          data-stuck={stuck || undefined}
          className="library-header-row sticky top-0 z-20 -mx-4 mt-2 bg-background px-4 py-4 transition-shadow sm:-mx-9 sm:px-9"
        >
          <TabStrip tabs={tabs.items} active={tabs.active} onChange={tabs.onChange} />
        </div>
      ) : null}
      {/* The controls take their own row in the body, under the tabs, and this row is their
          container: their search field narrows with it (cqw rather than the window). */}
      {toolbar ? <div className="@container min-w-0">{toolbar}</div> : null}
    </>
  );
}

// SPDX-License-Identifier: NRC-1
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const LabsPage = lazyRouteComponent(
  () => import("@/features/labs/labs-page"),
  "LabsPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/labs",
  staticData: { title: "Labs" },
  beforeLoad: () => requireAuth(),
  component: LabsPage,
});

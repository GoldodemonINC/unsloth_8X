// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ScalePage = lazyRouteComponent(
  () => import("@/features/advanced/scale-page"),
  "ScalePage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/scale",
  staticData: { title: "Scaler" },
  beforeLoad: () => requireAuth(),
  component: ScalePage,
});

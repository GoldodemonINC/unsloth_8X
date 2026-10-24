# Theming and the accent color

The UI has four independent theming dimensions, all driven from
`studio/frontend/src/index.css` and coordinated by
`features/settings/stores/appearance-custom-store.ts`:

| Dimension | Where it lives | Values |
| --- | --- | --- |
| Mode | `<html class="dark">` | light, dark |
| Palette | `<html data-palette>` | *unset* (Standard), `classic`, `minimal`, plus the flavor themes in `features/settings/lib/color-themes.ts` |
| Contrast | `<html data-contrast-adjust>` | slider, mixes tokens toward the ink |
| Custom accent | `ACCENT_VARS` | any `#rrggbb` the user picks |

## The accent is #9ceaf0

The Standard palette's brand accent used to be Unsloth green (`#17b88b`). It is
now **`#9ceaf0`**, a light cyan, in both modes:

```css
--primary: #9ceaf0;
--primary-foreground: #0f1114;
```

The foreground is the dark ink rather than white on purpose. `#9ceaf0` is light:
white on it scores about **1.3:1**, far under the 4.5:1 AA floor, while
`#0f1114` on it scores about **15:1**. Every primary button, active control and
switch track therefore reads with dark text in both light and dark mode.

The same pair is set for the sidebar (`--sidebar-primary`,
`--sidebar-primary-foreground`), and `--control-accent` follows `--primary` in
the Standard palette, so toggles, badges and chips pick the accent up
automatically. The Standard entry in `COLOR_THEMES`
(`features/settings/lib/color-themes.ts`) carries the same value, which is what
the appearance picker shows as the default swatch.

### What is still green, and why

Two greens are deliberately untouched:

- `--verified: #17b88b` — the status color for verified badges and token
  indicators. Status is a signal, not a theme color: it stays put in every
  palette and mode (see `tests/appearance-accent-vars.test.ts`).
- The chart series (`--chart-1` … `--chart-5`) — data colors. A light cyan first
  series would be hard to read on a white chart surface, so the series keep
  their hues. Note that a *custom* accent does overwrite `--chart-1` (it is in
  `ACCENT_VARS`), so the two only diverge for the default palette.

### Consequence to know about

`text-primary` is used for links and active labels (161 call sites). At
`#9ceaf0` on the light page those gain very little contrast, which is the
trade-off of a pale accent. If accent-colored text needs to stay readable,
either darken the accent for text only (a `--primary-ink` mixed toward
`--foreground`) or keep links on `--foreground` with an accent underline.

## Where to change the accent

1. `src/index.css` — the `:root` block (light) and the `.dark` block, both
   `--primary` / `--primary-foreground` and `--sidebar-primary` /
   `--sidebar-primary-foreground`.
2. `src/features/settings/lib/color-themes.ts` — the `standard.light.accent` and
   `standard.dark.accent` values.
3. Leave `--verified` and the chart tokens alone unless the change is about
   status or data colors.

Run `node --experimental-strip-types --test tests/appearance-accent-vars.test.ts tests/contrast-slider.test.ts`
afterwards: the first covers accent-to-foreground contrast, the second fails on
any chrome that hardcodes an alpha wash instead of a themed token.

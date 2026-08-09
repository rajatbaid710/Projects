/**
 * Design tokens — the white-label layer's source of truth.
 *
 * These are plain data, not Tailwind classes and not CSS. That is the whole
 * point: web consumes them through the CSS custom properties declared in
 * `globals.css`, and a React Native client can consume this same file directly
 * through a theme provider. Nothing here may import from `next`, `react`, or a
 * styling library, or that portability is lost.
 *
 * Phase 4 replaces the literals below with a `ThemeConfig` record fetched from
 * `GET /config/theme` at startup; `applyTheme()` is the seam that makes the
 * swap a one-line change rather than a refactor.
 */

export type ColorTokens = {
  brand: string;
  brandContrast: string;
  surface: string;
  surfaceMuted: string;
  fg: string;
  fgMuted: string;
  border: string;
  success: string;
  warning: string;
  danger: string;
};

export type ThemeTokens = {
  colors: ColorTokens;
  radius: { sm: string; md: string; lg: string };
  font: { sans: string; mono: string };
};

export const lightTheme: ThemeTokens = {
  colors: {
    brand: "#4f46e5",
    brandContrast: "#ffffff",
    surface: "#ffffff",
    surfaceMuted: "#f4f4f5",
    fg: "#18181b",
    fgMuted: "#52525b",
    border: "#e4e4e7",
    success: "#15803d",
    warning: "#a16207",
    danger: "#b91c1c",
  },
  radius: { sm: "0.25rem", md: "0.5rem", lg: "0.75rem" },
  font: {
    sans: "ui-sans-serif, system-ui, -apple-system, sans-serif",
    mono: "ui-monospace, SFMono-Regular, Menlo, monospace",
  },
};

export const darkTheme: ThemeTokens = {
  ...lightTheme,
  colors: {
    brand: "#818cf8",
    brandContrast: "#1e1b4b",
    surface: "#09090b",
    surfaceMuted: "#18181b",
    fg: "#fafafa",
    fgMuted: "#a1a1aa",
    border: "#27272a",
    success: "#4ade80",
    warning: "#fbbf24",
    danger: "#f87171",
  },
};

/** camelCase token key -> the `--kebab-case` custom property in globals.css. */
function cssVarName(key: string): string {
  return `--${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;
}

/**
 * Apply a theme at runtime by writing CSS custom properties onto `:root`.
 *
 * This is how a branded deployment restyles itself without a rebuild: the
 * tokens arrive as JSON from the API and land here. Tailwind's utilities read
 * the same variables, so `bg-brand` follows automatically.
 */
export function applyTheme(theme: Partial<ColorTokens>, root?: HTMLElement): void {
  const target = root ?? (typeof document === "undefined" ? null : document.documentElement);
  if (!target) return; // no-op during server rendering

  for (const [key, value] of Object.entries(theme)) {
    if (value) target.style.setProperty(cssVarName(key), value);
  }
}

import type { CSSProperties, ReactNode } from "react";

import { Icon, type IconName } from "./Icon";

// The shared controls. Every interactive element in the three views is one of these, so hover,
// pressed, selected, focus and disabled states look the same everywhere (DESIGN_SPEC.md §6).

export function Button({
  children,
  icon,
  variant = "default",
  pressed,
  disabled,
  title,
  onClick,
  testId,
}: {
  children?: ReactNode;
  icon?: IconName;
  variant?: "default" | "primary" | "ghost" | "hud";
  pressed?: boolean;
  disabled?: boolean;
  title?: string;
  onClick?: () => void;
  testId?: string;
}) {
  const iconOnly = children === undefined || children === null;
  return (
    <button
      type="button"
      className={`btn btn-${variant}${iconOnly ? " btn-icon" : ""}`}
      aria-pressed={pressed}
      aria-label={iconOnly ? title : undefined}
      data-tip={title}
      disabled={disabled}
      onClick={onClick}
      data-testid={testId}
    >
      {icon && <Icon name={icon} />}
      {children}
    </button>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  variant = "default",
}: {
  options: { id: T; label: string; icon?: IconName; hint?: string; disabled?: boolean }[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  variant?: "default" | "hud";
}) {
  return (
    <div className={`segmented segmented-${variant}`} role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          aria-pressed={o.id === value}
          disabled={o.disabled}
          data-tip={o.hint}
          onClick={() => onChange(o.id)}
          data-testid={`seg-${o.id}`}
        >
          {o.icon && <Icon name={o.icon} size={14} />}
          {o.label && <span className="seg-text">{o.label}</span>}
        </button>
      ))}
    </div>
  );
}

export type ChipKind = "run" | "ok" | "warn" | "bad" | "idle" | "accent" | "plain";

const CHIP_ICON: Partial<Record<ChipKind, IconName>> = { ok: "flag", bad: "cross", warn: "stall", idle: "clock" };

/** A status chip: colour, icon and word together, never colour alone. */
export function Chip({ kind, children, icon }: { kind: ChipKind; children: ReactNode; icon?: IconName | null }) {
  const name = icon === undefined ? CHIP_ICON[kind] : icon;
  return (
    <span className="chip" data-kind={kind}>
      {kind === "run" && <span className="chip-live" aria-hidden="true" />}
      {name && <Icon name={name} size={12} />}
      {children}
    </span>
  );
}

export function Panel({
  title,
  eyebrow,
  actions,
  children,
  className = "",
  flush = false,
  testId,
}: {
  title?: ReactNode;
  /** The small capital label above the title: what kind of thing this panel is. */
  eyebrow?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  /** No padding around the body (for tables and stages that go edge to edge). */
  flush?: boolean;
  testId?: string;
}) {
  return (
    <section className={`panel ${className}`} data-testid={testId}>
      {(title || actions || eyebrow) && (
        <header className="panel-head">
          <div className="panel-titles">
            {eyebrow && <span className="eyebrow">{eyebrow}</span>}
            {title && <h3 className="panel-title">{title}</h3>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      <div className={flush ? "panel-body panel-flush" : "panel-body"}>{children}</div>
    </section>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="field" data-tip={hint}>
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

export function Select<T extends string | number>({
  value,
  options,
  onChange,
  disabled,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <span className="select">
      <select
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => {
          const raw = e.target.value;
          onChange((typeof value === "number" ? Number(raw) : raw) as T);
        }}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <Icon name="down" size={12} />
    </span>
  );
}

/** A figure with its label: the label in condensed capitals, the value in the number face. */
export function Stat({
  label,
  value,
  unit,
  note,
  size = "md",
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  note?: ReactNode;
  size?: "sm" | "md" | "lg";
}) {
  return (
    <div className={`stat stat-${size}`}>
      <span className="eyebrow">{label}</span>
      <span className="stat-value">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

/**
 * A bar for a value in −1..1 drawn from the middle (signed), or 0..1 from the left. Blue is
 * positive, orange negative: the same pair as every other signed quantity in the app.
 */
export function ValueBar({
  value,
  signed = true,
  tone,
  marker,
}: {
  value: number;
  signed?: boolean;
  tone?: "pos" | "neg" | "accent" | "neutral";
  /** A second value shown as a thin tick (for "now" beside a what-if value). */
  marker?: number;
}) {
  const v = Math.max(-1, Math.min(1, value));
  const colour = tone ?? (v < 0 ? "neg" : "pos");
  const style: CSSProperties = signed
    ? { left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%` }
    : { left: 0, width: `${Math.max(0, v) * 100}%` };
  const tick = marker === undefined ? null : signed ? 50 + Math.max(-1, Math.min(1, marker)) * 50 : marker * 100;
  return (
    <span className={`vbar${signed ? " vbar-signed" : ""}`} aria-hidden="true">
      <span className={`vbar-fill vbar-${colour}`} style={style} />
      {tick !== null && <span className="vbar-tick" style={{ left: `${tick}%` }} />}
    </span>
  );
}

/** A swatch + words: one legend entry. The text stays in text colour; the mark carries identity. */
export function LegendItem({ children, swatch }: { children: ReactNode; swatch: ReactNode }) {
  return (
    <span className="legend-item">
      {swatch}
      <span>{children}</span>
    </span>
  );
}

export function Swatch({ color, shape = "square" }: { color: string; shape?: "square" | "line" | "dot" | "ring" }) {
  return <span className={`swatch swatch-${shape}`} style={{ "--swatch": color } as CSSProperties} aria-hidden="true" />;
}

/** A tag that marks a part of the design the production app cannot do yet (DESIGN_SPEC §12). */
export function Planned({ children }: { children: ReactNode }) {
  return (
    <span className="planned" data-tip="Shown as a design proposal. Not in the current app: see PRODUCTION_MAPPING.md.">
      {children}
    </span>
  );
}

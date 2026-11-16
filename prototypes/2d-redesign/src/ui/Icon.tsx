// One small icon set, drawn on a 16 px grid with a 1.5 px stroke, so every icon has the same
// weight as the text beside it. Icons never carry meaning alone: every button that uses one also
// has a text label or an aria-label and a tooltip.

const PATHS: Record<string, string> = {
  play: "M5 3.5v9l7.5-4.5z",
  pause: "M5 3.5v9M11 3.5v9",
  step: "M4 3.5v9l6-4.5zM12 3.5v9",
  reset: "M3.5 8a4.5 4.5 0 1 0 1.4-3.3M3.5 2.8v2.7h2.7",
  follow: "M8 2v2.5M8 11.5V14M2 8h2.5M11.5 8H14M8 10.3a2.3 2.3 0 1 0 0-4.6 2.3 2.3 0 0 0 0 4.6z",
  fit: "M2.5 6V3.5a1 1 0 0 1 1-1H6M10 2.5h2.5a1 1 0 0 1 1 1V6M13.5 10v2.5a1 1 0 0 1-1 1H10M6 13.5H3.5a1 1 0 0 1-1-1V10",
  plus: "M8 3.5v9M3.5 8h9",
  minus: "M3.5 8h9",
  sun: "M8 10.8a2.8 2.8 0 1 0 0-5.6 2.8 2.8 0 0 0 0 5.6zM8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1",
  moon: "M13 9.6A5.5 5.5 0 0 1 6.4 3 5.5 5.5 0 1 0 13 9.6z",
  auto: "M8 13.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11zM8 2.5v11",
  left: "M9.5 3.5 5 8l4.5 4.5",
  right: "M6.5 3.5 11 8l-4.5 4.5",
  down: "M3.5 6 8 10.5 12.5 6",
  flag: "M4 14V2.5M4 3h7.5l-1.6 2.7 1.6 2.8H4",
  cross: "M4 4l8 8M12 4l-8 8",
  clock: "M8 13.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11zM8 5v3.2l2 1.3",
  stall: "M8 13.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11zM6.5 6v4M9.5 6v4",
  track: "M5.5 3.5h5a3 3 0 0 1 3 3v0a3 3 0 0 1-3 3H8.2a1.7 1.7 0 0 0 0 3.4h2.3M5.5 3.5a3 3 0 0 0 0 6",
  network: "M3 4.5a1.3 1.3 0 1 0 0-.01zM3 11.5a1.3 1.3 0 1 0 0-.01zM13 8a1.3 1.3 0 1 0 0-.01zM8 8a1.3 1.3 0 1 0 0-.01zM4.2 4.9 6.9 7.4M4.2 11.1l2.7-2.5M9.3 8h2.4",
  flask: "M6.2 2.5h3.6M6.8 2.5v4L3.4 12a1 1 0 0 0 .9 1.5h7.4a1 1 0 0 0 .9-1.5L9.2 6.5v-4M5 10h6",
  split: "M2.5 3.5h11v9h-11zM8 3.5v9",
  page: "M2.5 3.5h11v9h-11zM10.5 3.5v9",
  rays: "M3 13 13 3M3 13l9.5-4M3 13l4-9.5",
  trail: "M2.5 12.5c2-5 4-1 6-4s2.5-4 5-5",
  info: "M8 13.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11zM8 7.3v3.5M8 5.2v.1",
  check: "M3.5 8.5 6.5 11.5 12.5 5",
  dot: "M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  crown: "M2.5 5.5 5 8l3-4.5L11 8l2.5-2.5-1 6.5h-9z",
  ghost: "M4 13.5V7a4 4 0 0 1 8 0v6.5l-2-1.3-2 1.3-2-1.3zM6.6 7.2v.1M9.4 7.2v.1",
  chart: "M2.5 13.5h11M4 11V8M7 11V4.5M10 11V6.5M13 11V3",
  compare: "M2.5 11.5c2-.5 3-5 5.5-5s3 3 5.5 2.5M2.5 6c2 .5 3 4.5 5.5 4.5s3-5 5.5-6",
  rocket: "M9.5 2.5c2 .2 3.8 2 4 4L9 11 5 7zM5 7 2.8 7.4 2.5 10l2.6.5M9 11l-.4 2.2-2.6.3-.5-2.6M5.5 10.5 3 13",
  data: "M2.5 4.5c0-1.1 2.5-2 5.5-2s5.5.9 5.5 2-2.5 2-5.5 2-5.5-.9-5.5-2zM2.5 4.5v7c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2v-7M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2",
  dna: "M5 2.5c0 3 6 3.5 6 5.5s-6 2.5-6 5.5M11 2.5c0 3-6 3.5-6 5.5s6 2.5 6 5.5M6 4.5h4M6 11.5h4",
  drawer: "M2.5 3.5h11v9h-11zM10 3.5v9",
  eye: "M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8zM8 9.8a1.8 1.8 0 1 0 0-3.6 1.8 1.8 0 0 0 0 3.6z",
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const filled = name === "play";
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

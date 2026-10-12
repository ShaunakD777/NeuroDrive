/** A placeholder card for one of the three views, until the real view is built. */
export function ViewPanel({ title, comingIn }: { title: string; comingIn: string }) {
  return (
    <section className="view">
      <h2>{title}</h2>
      <p>Placeholder. The real view arrives in {comingIn}.</p>
    </section>
  );
}

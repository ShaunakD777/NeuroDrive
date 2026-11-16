import { Icon, type IconName } from "../../components/ui/Icon";
import { useLive } from "../../state/live";
import { useSelection } from "../../state/selection";
import { bodyCss, bodyOf } from "./cars";
import { carRole } from "./population";
import { standingsOrder, statusCounts } from "./stageLayout";
import { describeStatus } from "./status";

const STATUS_ICON: Record<number, IconName> = { 1: "cross", 2: "flag", 3: "stall", 4: "clock" };

/**
 * Standings (task E3, recommendation E14), in the Environment page's drawer: every car in the order
 * they are on the road now (by progress, which the server sends). In a GA race each row also shows
 * the colour band and elite ring the GA's ranking gave it and its fitness F; for other drivers its
 * speed. A click selects the car in every view (`selectCar`).
 */
export function Standings() {
  const frame = useLive((s) => s.frame);
  const scenario = useLive((s) => s.scenario);
  const geometry = useLive((s) => s.geometry);
  if (frame === null || scenario === null) {
    return <p className="hint standings-empty">Waiting for the first frame…</p>;
  }
  const info = scenario.population;
  const { progress, status, v } = frame.cars;
  const count = progress.length;
  const selected = frame.selected?.index ?? 0;
  const total = geometry === null ? null : Math.max(1, scenario.target_laps) * geometry.length;
  const counts = statusCounts(status);
  const select = (i: number) => useSelection.getState().selectCar(i);

  return (
    <div className="standings" data-testid="standings">
      <div className="standings-summary">
        <span>
          <b className="num">{counts.running}</b> running
        </span>
        <span>
          <b className="num">{counts.finished}</b> finished
        </span>
        <span>
          <b className="num">{counts.crashed}</b> crashed
        </span>
        <span>
          <b className="num">{counts.stopped}</b> stopped
        </span>
      </div>
      <table className="standings-table">
        <thead>
          <tr>
            <th className="r">Pos</th>
            <th>Car</th>
            <th>Progress</th>
            {info !== null ? (
              <th className="r" title={`The GA's fitness F over all ${info.n_starts} starts`}>
                F
              </th>
            ) : (
              <th className="r">km/h</th>
            )}
          </tr>
        </thead>
        <tbody>
          {standingsOrder(progress).map((i, position) => {
            const role = carRole(info, i);
            const ghost = role?.kind === "ghost";
            const text = describeStatus(status[i] ?? 0);
            const icon = STATUS_ICON[status[i] ?? 0];
            return (
              <tr
                key={i}
                data-selected={i === selected}
                data-testid="standings-row"
                data-car={i}
                onClick={() => select(i)}
                title={`Select car ${i + 1}`}
              >
                <td className="r num">{position + 1}</td>
                <td>
                  <span className="standings-car">
                    {ghost ? (
                      <i className="legend-ghost" aria-hidden="true" />
                    ) : (
                      <i
                        className="standings-swatch"
                        style={{ background: bodyCss(bodyOf(info, count, i, selected)) }}
                        data-elite={role?.elite ?? false}
                        aria-hidden="true"
                      />
                    )}
                    {ghost ? "Ghost" : `Car ${i + 1}`}
                    {role?.champion && (
                      <span className="standings-role" title="This generation's champion">
                        <Icon name="crown" size={12} />
                      </span>
                    )}
                    {icon !== undefined && (
                      <span className="standings-status" data-kind={text.kind} title={text.label}>
                        <Icon name={icon} size={12} />
                      </span>
                    )}
                  </span>
                </td>
                <td>
                  <span className="standings-progress" title={`${progress[i].toFixed(0)} m`}>
                    <span
                      style={{ width: `${total === null ? 0 : Math.max(0, Math.min(1, progress[i] / total)) * 100}%` }}
                      data-kind={text.kind}
                    />
                  </span>
                </td>
                <td className="r num">
                  {info === null
                    ? Math.round((v[i] ?? 0) * 3.6)
                    : ghost
                      ? (info.ghost_fitness ?? 0).toFixed(3)
                      : (info.fitness[i] ?? 0).toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

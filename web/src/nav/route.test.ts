import { describe, expect, it } from "vitest";

import { hashFor, parseHash } from "./route";
import { VIEW_IDS } from "./views";

describe("parseHash", () => {
  it("reads each view's hash", () => {
    expect(parseHash("#/environment")).toBe("environment");
    expect(parseHash("#/network")).toBe("network");
    expect(parseHash("#/experiment")).toBe("experiment");
  });

  it("tolerates a missing slash, upper case and trailing parts", () => {
    expect(parseHash("#network")).toBe("network");
    expect(parseHash("#/Network")).toBe("network");
    expect(parseHash("#/network/extra?x=1")).toBe("network");
  });

  it("returns null for empty or unknown hashes", () => {
    expect(parseHash("")).toBeNull();
    expect(parseHash("#/")).toBeNull();
    expect(parseHash("#/garage")).toBeNull();
  });

  it("round-trips every view through hashFor", () => {
    for (const id of VIEW_IDS) expect(parseHash(hashFor(id))).toBe(id);
  });
});

import { describe, expect, it } from "vitest";
import { createZip, dataUrlToBytes } from "./captureArchive";

describe("capture archive", () => {
  it("decodes base64 image data", () => {
    expect(Array.from(dataUrlToBytes("data:image/jpeg;base64,AQID"))).toEqual([
      1, 2, 3,
    ]);
  });

  it("creates a ZIP containing each requested filename", async () => {
    const archive = createZip([
      { name: "captures.json", data: new TextEncoder().encode("{}") },
      { name: "images/capture.jpg", data: new Uint8Array([1, 2, 3]) },
    ]);
    const bytes = new Uint8Array(await archive.arrayBuffer());
    const text = new TextDecoder().decode(bytes);

    expect(Array.from(bytes.slice(0, 4))).toEqual([0x50, 0x4b, 0x03, 0x04]);
    expect(text).toContain("captures.json");
    expect(text).toContain("images/capture.jpg");
    expect(Array.from(bytes.slice(-22, -18))).toEqual([0x50, 0x4b, 0x05, 0x06]);
  });
});


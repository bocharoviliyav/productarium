// Headless Mermaid diagram validator.
//
// Invoked by api/mermaid_verifier.py::verify_diagram as:
//     node api/_mermaid_validate.mjs
// Reads a single diagram body (the text between the ```mermaid fences) from
// stdin, validates it headlessly with mermaid.parse(), and writes ONE JSON
// object to stdout:
//     {"ok": true, "diagramType": "flowchart"}          -> diagram is valid
//     {"ok": false, "error": "Parse error ..."}          -> genuine parse error
//     {"ok": false, "unverifiable": true, "error": "..."}-> diagram type that
//         cannot be validated without a DOM (e.g. C4 -> "Bt.addHook is not a
//         function"). The Python side treats this as "cannot judge" and leaves
//         the diagram in place rather than burning repair attempts.
//
// No browser, no puppeteer, no network. The mermaid ESM bundle is imported from
// the repo's node_modules via an absolute path resolved relative to this file,
// so the validator works regardless of the current working directory.
//
// Failure modes:
// - If the mermaid import itself fails (e.g. node_modules absent in a
//   pure-Python container), the script exits non-zero with the marker string
//   "MERMAID_IMPORT_FAILED" on stderr. The Python caller downgrades that to an
//   unverifiable result so generation is never blocked by a missing frontend
//   dependency.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Resolve the mermaid ESM bundle shipped with the frontend. Preferred path is
// <repo>/node_modules/mermaid/dist/mermaid.esm.min.mjs; fall back to the
// package "module" entry resolved via the bundler-aware require.
function resolveMermaidPath() {
  const candidates = [
    path.resolve(__dirname, "..", "node_modules", "mermaid", "dist", "mermaid.esm.min.mjs"),
    path.resolve(__dirname, "..", "node_modules", "mermaid", "dist", "mermaid.core.mjs"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  // Last resort: use the bundler-aware require to load the package exports map.
  try {
    const require = createRequire(import.meta.url);
    const pkgPath = require.resolve("mermaid/package.json");
    const pkgDir = path.dirname(pkgPath);
    const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
    const entry = pkg.module || pkg.main || "./dist/mermaid.esm.min.mjs";
    const resolved = path.resolve(pkgDir, entry);
    if (fs.existsSync(resolved)) return resolved;
  } catch {
    // fall through to marker below
  }
  return null;
}

let mermaid;
try {
  const mermaidPath = resolveMermaidPath();
  if (!mermaidPath) {
    throw new Error("mermaid package not found in node_modules");
  }
  mermaid = (await import(mermaidPath)).default || (await import(mermaidPath)).mermaid;
} catch (err) {
  process.stderr.write("MERMAID_IMPORT_FAILED: " + String(err && err.message || err) + "\n");
  process.exit(3);
}

// Suppress the in-page error rendering path (we only call parse, not render,
// but initialize keeps the bundle quiet). suppressErrorReporting avoids the
// library attempting to touch DOM error containers.
try {
  mermaid.initialize({ startOnLoad: false, suppressErrorRendering: true });
} catch {
  // initialize is best-effort; parse() works without it.
}

// Diagram types whose parse path needs DOM hooks (e.g. C4 uses cytoscape which
// calls addHook). In a headless Node process these throw a distinctive error
// ("Bt.addHook is not a function" in mermaid 11.x) regardless of whether the
// diagram is valid, so they cannot be judged headlessly. We flag them as
// "unverifiable" so the orchestrator leaves them untouched instead of looping.
const UNVERIFIABLE_SIGNATURES = [
  "addHook is not a function",
  "cytoscape",
];

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => { data += chunk; });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

async function main() {
  const body = await readStdin();
  if (!body || !body.trim()) {
    // Empty diagram body: nothing to render, treat as valid (skip).
    process.stdout.write(JSON.stringify({ ok: true, diagramType: null, empty: true }) + "\n");
    return;
  }

  let diagramType = "unknown";
  try {
    // detectType does not throw for syntax errors and does not need a DOM.
    try {
      diagramType = mermaid.detectType(body) || "unknown";
    } catch {
      // keep "unknown"
    }

    await mermaid.parse(body);
    process.stdout.write(JSON.stringify({ ok: true, diagramType }) + "\n");
  } catch (err) {
    const msg = String((err && err.message) || err);
    const unverifiable = UNVERIFIABLE_SIGNATURES.some((sig) => msg.includes(sig));
    process.stdout.write(
      JSON.stringify({ ok: false, diagramType, unverifiable, error: msg }) + "\n"
    );
  }
}

main().catch((err) => {
  process.stdout.write(
    JSON.stringify({ ok: false, unverifiable: true, error: String(err && err.message || err) }) + "\n"
  );
});

import { cpSync, createReadStream, existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin, type ResolvedConfig } from "vite";

/** Precomputed motion the static viewer plays without a Python or GPU server. */
const DEMO_FILES = [
  "data/overhead_video/pose_rokoko.jsonl",
  "outputs/overhead_video_segments_rokoko.json",
  "outputs/overhead_video_assist_plan_rokoko.json",
];

function staticDemoFiles(): Plugin {
  let config: ResolvedConfig;

  return {
    name: "hit-exolimb-static-demo",
    configResolved(resolved) {
      config = resolved;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const rawUrl = req.url?.split("?")[0] ?? "";
        let pathname = "";
        try {
          pathname = decodeURIComponent(rawUrl);
        } catch {
          next();
          return;
        }
        const base = config.base || "/";
        if (base !== "/" && pathname.startsWith(base)) {
          pathname = `/${pathname.slice(base.length)}`;
        }
        const rel = pathname.replace(/^\/+/, "");
        if (!DEMO_FILES.includes(rel)) {
          next();
          return;
        }
        const filePath = path.resolve(config.root, rel);
        if (!existsSync(filePath)) {
          next();
          return;
        }
        const type = rel.endsWith(".json") ? "application/json; charset=utf-8" : "text/plain; charset=utf-8";
        res.setHeader("Content-Type", type);
        createReadStream(filePath).pipe(res);
      });
    },
    closeBundle() {
      if (!config || config.command !== "build") return;
      const outDir = path.resolve(config.root, config.build.outDir);
      for (const rel of DEMO_FILES) {
        const source = path.resolve(config.root, rel);
        const dest = path.resolve(outDir, rel);
        mkdirSync(path.dirname(dest), { recursive: true });
        cpSync(source, dest);
      }
    },
  };
}

export default defineConfig({
  // GitHub Pages project site. `vite build --base` overrides this in CI.
  base: "/hit_exolimb_motion/",
  plugins: [staticDemoFiles()],
});

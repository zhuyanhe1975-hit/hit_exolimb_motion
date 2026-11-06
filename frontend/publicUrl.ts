/// <reference types="vite/client" />

/** Resolve a site-root path against Vite's `base` (for example `/hit_exolimb_motion/`). */
export function publicAsset(path: string): string {
  const normalized = path.replace(/^\/+/, "");
  const base = import.meta.env.BASE_URL || "/";
  return `${base}${normalized}`;
}

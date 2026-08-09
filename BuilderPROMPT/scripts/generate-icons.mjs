// Generates the app's PWA / favicon / apple-touch icons from a single inline
// SVG so there's no binary design asset to keep in sync. Re-run with
// `node scripts/generate-icons.mjs` after changing the brand color.
import sharp from "sharp";
import fs from "fs/promises";
import path from "path";

const BRAND = "#4338ca";
const OUT_DIR = path.join(process.cwd(), "public", "icons");

function receiptZigzagPath(x, yTop, width, toothCount, toothHeight, yZigStart) {
  const toothWidth = width / toothCount;
  let d = `M ${x} ${yTop} L ${x + width} ${yTop} L ${x + width} ${yZigStart} `;
  for (let i = toothCount; i >= 1; i--) {
    const xRight = x + i * toothWidth;
    const xMid = xRight - toothWidth / 2;
    const xLeft = xRight - toothWidth;
    d += `L ${xRight} ${yZigStart} L ${xMid} ${yZigStart + toothHeight} L ${xLeft} ${yZigStart} `;
  }
  d += `Z`;
  return d;
}

function buildSvg() {
  const s = 512;
  const receiptPath = receiptZigzagPath(156, 96, 200, 7, 18, 380);
  return `
<svg width="${s}" height="${s}" viewBox="0 0 ${s} ${s}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${s}" height="${s}" rx="112" fill="${BRAND}"/>
  <path d="${receiptPath}" fill="white"/>
  <rect x="184" y="150" width="144" height="14" rx="7" fill="${BRAND}"/>
  <rect x="184" y="188" width="144" height="14" rx="7" fill="${BRAND}"/>
  <rect x="184" y="226" width="96" height="14" rx="7" fill="${BRAND}"/>
  <rect x="184" y="290" width="144" height="20" rx="10" fill="${BRAND}" opacity="0.15"/>
  <rect x="184" y="290" width="72" height="20" rx="10" fill="${BRAND}"/>
</svg>`.trim();
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const svg = Buffer.from(buildSvg());

  const targets = [
    { file: "icon-192.png", size: 192 },
    { file: "icon-512.png", size: 512 },
    { file: "icon-maskable-512.png", size: 512, maskablePad: true },
  ];

  for (const t of targets) {
    let pipeline = sharp(svg, { density: 384 }).resize(t.size, t.size);
    if (t.maskablePad) {
      // Maskable icons need generous safe-area padding (~20%) around the glyph.
      const inner = Math.round(t.size * 0.7);
      pipeline = sharp(svg, { density: 384 })
        .resize(inner, inner)
        .extend({
          top: Math.round((t.size - inner) / 2),
          bottom: Math.round((t.size - inner) / 2),
          left: Math.round((t.size - inner) / 2),
          right: Math.round((t.size - inner) / 2),
          background: BRAND,
        });
    }
    await pipeline.png().toFile(path.join(OUT_DIR, t.file));
    console.log(`wrote ${t.file}`);
  }

  // Next.js App Router file conventions — auto-detected, no metadata wiring needed.
  await sharp(svg, { density: 384 })
    .resize(48, 48)
    .png()
    .toFile(path.join(process.cwd(), "src", "app", "icon.png"));
  console.log("wrote src/app/icon.png (favicon convention)");

  await sharp(svg, { density: 384 })
    .resize(180, 180)
    .png()
    .toFile(path.join(process.cwd(), "src", "app", "apple-icon.png"));
  console.log("wrote src/app/apple-icon.png (apple-touch-icon convention)");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

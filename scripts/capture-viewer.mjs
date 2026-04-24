import puppeteer from "puppeteer-core";

const url = process.argv[2] ?? "http://127.0.0.1:5174/";
const out = process.argv[3] ?? "/tmp/hit_exolimb_viewer.png";

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/google-chrome",
  headless: "new",
  args: [
    "--no-sandbox",
    "--disable-gpu",
    "--use-gl=swiftshader",
    "--enable-unsafe-swiftshader",
    "--window-size=1600,1000",
  ],
});

const page = await browser.newPage();
page.on("console", (message) => {
  console.log(`[browser:${message.type()}] ${message.text()}`);
});
page.on("pageerror", (error) => {
  console.error(`[browser:pageerror] ${error.stack ?? error.message}`);
});

await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForFunction(
  () => document.querySelector("#frame-count")?.textContent !== "0",
  { timeout: 30000 },
);
await new Promise((resolve) => setTimeout(resolve, 1500));
await page.screenshot({ path: out, fullPage: false });
await browser.close();
console.log(out);

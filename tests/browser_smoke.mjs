// End-to-end check in this PC's headless Chrome/Edge over the DevTools protocol: opens the page through serve.py
// (token proxy) and from file://, loads real models from Hugging Face at desktop and phone widths, and fails on
// load errors, horizontal page scroll or a wrong gated-repo path. Screenshots go to logs/. Needs network access.
// Run: node tests/browser_smoke.mjs
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const Root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LogDir = path.join(Root, "logs");
const FileUrl = pathToFileURL(path.join(Root, "index.html")).href;
const HfHome = process.env.HF_HOME || path.join(os.homedir(), ".cache", "huggingface");
const HasToken = Boolean(process.env.HF_TOKEN) || fs.existsSync(path.join(HfHome, "token"));
// file: true opens index.html directly (no token); the rest go through serve.py.
const Cases = [
  { query: "Qwen/Qwen3-8B", width: 1920 },
  { query: "deepseek-ai/DeepSeek-V3", width: 1920 },
  { query: "google/gemma-3-27b-it", width: 375, expectMirror: true },
  { query: "gpt-oss-20b", width: 1280, expectRepo: "openai/gpt-oss-20b" },
  { query: "Qwen/Qwen3-Next-80B-A3B-Instruct", width: 375 },
  { query: "meta-llama/Llama-3.1-8B", width: 1280, expectMirror: !HasToken },
  { query: "meta-llama/Llama-3.1-8B", width: 1280, expectMirror: true, file: true },
];
const LoadTimeoutMs = 30000;

function findBrowser() {
  const roots = [process.env.ProgramFiles, process.env["ProgramFiles(x86)"], process.env.LocalAppData].filter(Boolean);
  for (const tail of ["Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"]) {
    for (const root of roots) {
      const candidate = path.join(root, tail);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  throw new Error("no Chrome or Edge found");
}

async function waitFor(check, timeout_ms, label) {
  const deadline = Date.now() + timeout_ms;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  throw new Error(`timed out: ${label}`);
}

async function startBrowser() {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "llmarch-smoke-"));
  const browser = spawn(findBrowser(), [
    "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
    "--no-first-run", "--no-default-browser-check", "--window-size=1280,900", "about:blank",
  ], { stdio: "ignore" });
  const port_file = path.join(profile, "DevToolsActivePort");
  const port = await waitFor(() => fs.existsSync(port_file) && fs.readFileSync(port_file, "utf8").split("\n")[0].trim(), 15000, "DevToolsActivePort");
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  return { browser, profile, socket };
}

function cdp(socket) {
  let next_id = 0;
  const waiting = new Map();
  const errors = [];
  socket.onmessage = message => {
    const data = JSON.parse(message.data);
    if (data.id && waiting.has(data.id)) {
      const { resolve, reject } = waiting.get(data.id);
      waiting.delete(data.id);
      if (data.error) reject(new Error(data.error.message));
      else resolve(data.result);
    } else if (data.method === "Runtime.exceptionThrown") {
      errors.push(data.params.exceptionDetails.exception?.description || data.params.exceptionDetails.text);
    }
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++next_id;
    waiting.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  return { send, errors };
}

async function evaluate(send, expression) {
  const res = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  return res.result.value;
}

function startServer() {
  const python = process.platform === "win32" ? "py" : "python3";
  const args = process.platform === "win32" ? ["-3", "serve.py", "--serve-only"] : ["serve.py", "--serve-only"];
  const server = spawn(python, args, { cwd: Root, stdio: ["ignore", "pipe", "inherit"] });
  return new Promise((resolve, reject) => {
    server.stdout.on("data", chunk => {
      const match = /serving at (\S+)/.exec(chunk.toString());
      if (match) resolve({ server, url: match[1] });
    });
    server.on("exit", code => reject(new Error(`serve.py exited with ${code}`)));
  });
}

async function runCase(send, errors, item, serverUrl) {
  const mobile = item.width < 600;
  await send("Emulation.setDeviceMetricsOverride", { width: item.width, height: 900, deviceScaleFactor: 1, mobile });
  // A bare hash change would keep the previous model on screen and satisfy the wait below at once.
  await send("Page.navigate", { url: "about:blank" });
  await send("Page.navigate", { url: `${item.file ? FileUrl : serverUrl}#${item.query}` });
  const state = await waitFor(() => evaluate(send, `(() => {
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    if (status && status.dataset.error === '1') return { error: status.textContent };
    if (result && !result.hidden && document.querySelector('#diagram svg')) return { ok: true };
    return null;
  })()`), LoadTimeoutMs, item.query);
  if (state.error) throw new Error(`${item.query}: ${state.error}`);
  const info = await evaluate(send, `({
    repo: document.getElementById('name').textContent,
    sub: document.getElementById('sub').textContent,
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    svgText: document.querySelector('#diagram svg').textContent.length,
    specRows: document.querySelectorAll('#spec dt').length,
    heading: document.querySelector('#detail h3').textContent,
  })`);
  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const name = `${item.query.replace(/[^A-Za-z0-9.-]+/g, "_")}_${item.width}${item.file ? "_file" : ""}.png`;
  fs.writeFileSync(path.join(LogDir, name), Buffer.from(shot.data, "base64"));
  const problems = [];
  const mirrored = info.sub.includes("config");
  if (info.scrollWidth > info.clientWidth) problems.push(`page scrolls sideways (${info.scrollWidth} > ${info.clientWidth})`);
  if (item.expectRepo && info.repo !== item.expectRepo) problems.push(`resolved to ${info.repo}`);
  if (item.expectMirror && !mirrored) problems.push("gated repo did not fall back to a mirror");
  if (item.expectMirror === false && mirrored) problems.push("token did not open the gated original");
  if (info.specRows < 8) problems.push(`only ${info.specRows} spec rows`);
  if (info.heading !== "Layer composition") problems.push(`default language is not English: ${info.heading}`);
  if (errors.length) problems.push(`page exceptions: ${errors.splice(0).join(" | ")}`);
  return { name, problems };
}

// Picks a real config through the file input, the same path as the upload button.
async function runUpload(send, errors, serverUrl, profile) {
  const fixture = JSON.parse(fs.readFileSync(path.join(Root, "tests", "fixtures", "qwen-qwen3-30b-a3b.json"), "utf8"));
  const upload = path.join(profile, "config.json");
  fs.writeFileSync(upload, JSON.stringify(fixture.config));
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: "about:blank" });
  await send("Page.navigate", { url: serverUrl });
  await waitFor(() => evaluate(send, "document.readyState === 'complete' && !!document.getElementById('file')"), 10000, "page load");
  const { root } = await send("DOM.getDocument");
  const { nodeId } = await send("DOM.querySelector", { nodeId: root.nodeId, selector: "#file" });
  await send("DOM.setFileInputFiles", { nodeId, files: [upload] });
  const info = await waitFor(() => evaluate(send, `(() => {
    const result = document.getElementById('result');
    if (result.hidden || !document.querySelector('#diagram svg')) return null;
    return { sub: document.getElementById('sub').textContent, moe: document.querySelector('#diagram svg').textContent.includes('MoE layer') };
  })()`), 10000, "upload render");
  const problems = [];
  if (!info.sub.includes("config.json")) problems.push(`source line: ${info.sub}`);
  if (!info.moe) problems.push("MoE inset missing for an MoE config");
  if (errors.length) problems.push(`page exceptions: ${errors.splice(0).join(" | ")}`);
  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  fs.writeFileSync(path.join(LogDir, "upload_1280.png"), Buffer.from(shot.data, "base64"));
  return problems;
}

// Korean is opt-in: the toggle relabels the shown model in place and the choice survives a reload.
async function runLanguage(send, errors, serverUrl) {
  const heading = () => evaluate(send, "document.querySelector('#detail h3').textContent");
  const click = code => evaluate(send, `document.querySelector('#lang button[data-lang="${code}"]').click()`);
  await send("Page.navigate", { url: "about:blank" });
  await send("Page.navigate", { url: `${serverUrl}#Qwen/Qwen3-8B` });
  await waitFor(() => evaluate(send, "!document.getElementById('result').hidden"), LoadTimeoutMs, "model load");
  const problems = [];
  await click("ko");
  if ((await heading()) !== "레이어별 구성") problems.push("toggle did not switch to Korean");
  const shot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(LogDir, "lang_ko_1280.png"), Buffer.from(shot.data, "base64"));
  await send("Page.reload");
  await waitFor(() => evaluate(send, "!document.getElementById('result').hidden"), LoadTimeoutMs, "reload");
  if ((await heading()) !== "레이어별 구성") problems.push("Korean choice not kept after reload");
  await click("en");
  if ((await heading()) !== "Layer composition") problems.push("toggle did not switch back to English");
  if (errors.length) problems.push(`page exceptions: ${errors.splice(0).join(" | ")}`);
  return problems;
}

fs.mkdirSync(LogDir, { recursive: true });
const { server, url: serverUrl } = await startServer();
const { browser, profile, socket } = await startBrowser();
const { send, errors } = cdp(socket);
let failed = 0;
try {
  await send("Page.enable");
  await send("Runtime.enable");
  for (const item of Cases) {
    const label = `${item.query} @${item.width}px ${item.file ? "file://" : "serve.py"}`;
    try {
      const { name, problems } = await runCase(send, errors, item, serverUrl);
      if (problems.length) failed++;
      console.log(`${problems.length ? "FAIL" : "ok  "} ${label} -> logs/${name}${problems.length ? `: ${problems.join("; ")}` : ""}`);
    } catch (err) {
      failed++;
      console.log(`FAIL ${label}: ${err.message}`);
    }
  }
  try {
    const problems = await runUpload(send, errors, serverUrl, profile);
    if (problems.length) failed++;
    console.log(`${problems.length ? "FAIL" : "ok  "} config.json upload -> logs/upload_1280.png${problems.length ? `: ${problems.join("; ")}` : ""}`);
  } catch (err) {
    failed++;
    console.log(`FAIL config.json upload: ${err.message}`);
  }
  try {
    const problems = await runLanguage(send, errors, serverUrl);
    if (problems.length) failed++;
    console.log(`${problems.length ? "FAIL" : "ok  "} language toggle -> logs/lang_ko_1280.png${problems.length ? `: ${problems.join("; ")}` : ""}`);
  } catch (err) {
    failed++;
    console.log(`FAIL language toggle: ${err.message}`);
  }
} finally {
  socket.close();
  browser.kill();
  server.kill();
  // Chrome can hold profile files for a moment after kill (EPERM on Windows); retry instead of failing the run.
  setTimeout(() => fs.rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 }), 500);
}
console.log(`${Cases.length + 2 - failed}/${Cases.length + 2} passed`);
process.exitCode = failed ? 1 : 0;

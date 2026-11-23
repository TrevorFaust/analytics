// Wakes https://trevorfaust-analytics.streamlit.app/ and opens a live session.
// A download of the outer page returns 200 even while the app is asleep.
// The same file lives in Second Brain as scripts/keep-analytics-awake.mjs.

import fs from "node:fs";
import https from "node:https";
import crypto from "node:crypto";
import { Agent } from "undici";

const ORIGIN = "https://trevorfaust-analytics.streamlit.app";
const RUNNING = 5;
const SHUTDOWN = 12;
const BOOT_MS = 3 * 60 * 1000;
const SESSION_MS = 20 * 1000;

const jar = new Map();
const agent = new Agent({ keepAliveTimeout: 10, keepAliveMaxTimeout: 10 });

function stamp() {
  const date = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const hours = pad(Math.floor(Math.abs(offset) / 60));
  const mins = pad(Math.abs(offset) % 60);
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}` +
    `${sign}${hours}:${mins}`
  );
}

let finished = false;

async function finish(line, code) {
  if (finished) return;
  finished = true;
  const text = `${stamp()} ${line}`;
  const log = process.env.ANALYTICS_PING_LOG;
  if (log) fs.appendFileSync(log, text + "\n");
  console.log(text);
  process.exitCode = code;
  await agent.close().catch(() => {});
}

function remember(response) {
  const cookies =
    typeof response.headers.getSetCookie === "function"
      ? response.headers.getSetCookie()
      : [];
  for (const cookie of cookies) {
    const pair = cookie.split(";")[0];
    const eq = pair.indexOf("=");
    if (eq > 0) jar.set(pair.slice(0, eq).trim(), pair.slice(eq + 1).trim());
  }
}

function cookieHeader() {
  return [...jar.entries()].map(([name, value]) => `${name}=${value}`).join("; ");
}

async function call(path, options = {}) {
  const headers = new Headers(options.headers);
  headers.set("accept", headers.get("accept") ?? "application/json");
  headers.set(
    "user-agent",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
  );
  const cookie = cookieHeader();
  if (cookie) headers.set("cookie", cookie);
  const response = await fetch(ORIGIN + path, {
    method: options.method ?? "GET",
    headers,
    redirect: "manual",
    dispatcher: agent,
  });
  remember(response);
  return response;
}

async function readStatus() {
  const response = await call("/api/v2/app/status");
  const csrf = (response.headers.get("x-csrf-token") ?? "").trim();
  if (response.status === 404) return { missing: true, csrf };
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`status ${response.status} ${text.slice(0, 180)}`);
  }
  const body = JSON.parse(text);
  return { missing: false, csrf, status: body.status };
}

async function resume(csrf) {
  const response = await call("/api/v2/app/resume", {
    method: "POST",
    headers: {
      "x-csrf-token": csrf,
      "x-streamlit-machine-id": crypto.randomUUID(),
      origin: ORIGIN,
      referer: `${ORIGIN}/`,
    },
  });
  if (response.status === 404) return false;
  if (!response.ok && response.status !== 204) {
    const text = await response.text();
    throw new Error(`resume ${response.status} ${text.slice(0, 180)}`);
  }
  return true;
}

async function waitUntil(label, ready, timeoutMs) {
  const start = Date.now();
  let last = "waiting";
  while (Date.now() - start < timeoutMs) {
    last = await ready();
    if (last === true) return true;
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
  throw new Error(`${label} timed out (${last})`);
}

function holdSession() {
  const url = new URL(ORIGIN);
  const key = crypto.randomBytes(16).toString("base64");
  return new Promise((resolve) => {
    const request = https.request({
      host: url.host,
      path: "/~/+/_stcore/stream",
      method: "GET",
      headers: {
        Connection: "Upgrade",
        Upgrade: "websocket",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": key,
        "Sec-WebSocket-Protocol": "streamlit",
        Origin: ORIGIN,
        Host: url.host,
      },
    });
    const timer = setTimeout(() => {
      request.destroy();
      resolve("timeout");
    }, 15000);
    request.on("upgrade", (_response, socket) => {
      clearTimeout(timer);
      setTimeout(() => {
        socket.end();
        resolve("open");
      }, SESSION_MS);
    });
    request.on("response", (response) => {
      clearTimeout(timer);
      response.resume();
      resolve(`http-${response.statusCode}`);
    });
    request.on("error", () => {
      clearTimeout(timer);
      resolve("error");
    });
    request.end();
  });
}

async function main() {
try {
  const first = await readStatus();
  if (first.missing) {
    await finish("NOT_FOUND The public app lookup says this app does not exist.", 1);
    return;
  }

  let woke = false;
  if (first.status === SHUTDOWN) {
    const accepted = await resume(first.csrf);
    if (!accepted) {
      await finish("NOT_FOUND The wake call could not find the app.", 1);
      return;
    }
    woke = true;
  }

  await waitUntil("boot", async () => {
    const current = await readStatus();
    if (current.missing) return "missing";
    return current.status === RUNNING ? true : `status ${current.status}`;
  }, BOOT_MS);

  await waitUntil("health", async () => {
    const response = await call("/~/+/_stcore/health", {
      headers: { accept: "text/plain" },
    });
    const text = (await response.text()).trim();
    return response.ok && text === "ok" ? true : `${response.status} ${text.slice(0, 40)}`;
  }, BOOT_MS);

  const session = await holdSession();
  const current = await readStatus();
  const label = woke ? "WOKE" : "AWAKE";
  await finish(`${label} status=${current.status} health=ok session=${session}`, 0);
} catch (error) {
  await finish(`FAIL ${error instanceof Error ? error.message : String(error)}`, 1);
}
}

await main();

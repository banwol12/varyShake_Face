import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import puppeteer from 'puppeteer-core';
import { WebSocketServer } from 'ws';
import os from 'os';
import { exec, spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 5173;

const SAVED_URLS_FILE = path.join(__dirname, 'public', 'saved_urls.json');

// Initialize saved_urls.json if it doesn't exist
if (!fs.existsSync(path.dirname(SAVED_URLS_FILE))) {
  fs.mkdirSync(path.dirname(SAVED_URLS_FILE), { recursive: true });
}
if (!fs.existsSync(SAVED_URLS_FILE)) {
  fs.writeFileSync(SAVED_URLS_FILE, JSON.stringify([]));
}

// Helper to get saved URLs
function getSavedUrls() {
  try {
    return JSON.parse(fs.readFileSync(SAVED_URLS_FILE, 'utf8'));
  } catch (e) {
    return [];
  }
}

// Helper to add a saved URL
function addSavedUrl(url) {
  try {
    const urls = getSavedUrls();
    if (!urls.includes(url)) {
      urls.push(url);
      fs.writeFileSync(SAVED_URLS_FILE, JSON.stringify(urls, null, 2));
    }
  } catch (e) {
    console.error("Failed to write to saved_urls.json:", e.message);
  }
}

app.use(express.json({ limit: '50mb' }));

// Serve static assets from public/ folder at the root level (e.g. /models/ -> public/models/)
app.use(express.static(path.join(__dirname, 'public')));
// Serve index.html, main.js, style.css etc. from root
app.use(express.static(path.join(__dirname)));

// Route for homepage
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Proxy endpoint to load external images without CORS blocks in the browser
app.get('/api/proxy-image', async (req, res) => {
  const { url } = req.query;
  if (!url) {
    return res.status(400).send('URL is required.');
  }

  // SSRF Protection: Block requests to private/internal network addresses
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();
    const blockedPatterns = [
      /^localhost$/i,
      /^127\./,
      /^10\./,
      /^172\.(1[6-9]|2[0-9]|3[0-1])\./,
      /^192\.168\./,
      /^0\./,
      /^\[::1\]$/,
      /^\[fc/i,
      /^\[fd/i,
    ];
    if (blockedPatterns.some(p => p.test(hostname))) {
      return res.status(403).send('Access to internal network addresses is not allowed.');
    }
  } catch (e) {
    return res.status(400).send('Invalid URL.');
  }

  let targetUrl = url;
  try {
    let response = await fetch(targetUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      }
    });

    // Fallback logic for YouTube maxresdefault.jpg (some videos don't have max-res default thumbnail)
    if (!response.ok && targetUrl.includes('ytimg.com/vi/') && targetUrl.includes('maxresdefault.jpg')) {
      const fallbackUrl = targetUrl.replace('maxresdefault.jpg', 'hqdefault.jpg');
      console.log(`  [Proxy] maxresdefault failed for YouTube thumbnail, trying fallback: ${fallbackUrl}`);
      response = await fetch(fallbackUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
      });
    }

    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }

    const contentType = response.headers.get('content-type') || 'image/jpeg';
    const buffer = await response.arrayBuffer();

    res.set('Content-Type', contentType);
    res.set('Access-Control-Allow-Origin', '*'); // Prevent CORS blocks
    res.send(Buffer.from(buffer));
  } catch (err) {
    console.error('Proxy error for URL:', targetUrl, '-', err.message);
    res.status(500).send(`Error proxying image: ${err.message}`);
  }
});

// Global map to track active downloads to prevent duplicate spawns for the same URL
const activeDownloads = new Map();

// Endpoint to prepare (download) YouTube video locally in the background to avoid connection abort lockups
app.get('/api/youtube-prepare', (req, res) => {
  const { url } = req.query;
  if (!url) {
    return res.status(400).json({ error: 'YouTube URL is required.' });
  }

  // Preprocess URL: Support watch, shorts, mobile, youtu.be, embed, etc.
  let cleanUrl = url;
  const videoIdMatch = url.match(/(?:youtube\.com\/(?:watch\?.*v=|shorts\/|embed\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  if (videoIdMatch && videoIdMatch[1]) {
    cleanUrl = `https://www.youtube.com/watch?v=${videoIdMatch[1]}`;
  }

  console.log(`[YouTube Prepare] Request to download: ${cleanUrl} (Original: ${url})`);
  const tempFile = path.join(__dirname, 'public', 'youtube_temp.mp4');

  // Clean up any stale temp or part files from previous failed downloads
  try {
    const publicDir = path.join(__dirname, 'public');
    fs.readdirSync(publicDir).forEach(file => {
      if (file.startsWith('youtube_temp')) {
        try { fs.unlinkSync(path.join(publicDir, file)); } catch (e) {}
      }
    });
  } catch (e) {}

  // If already downloading the same URL, join the active queue
  if (activeDownloads.has(cleanUrl)) {
    console.log(`[YouTube Prepare] Already downloading this URL, joining queue...`);
    activeDownloads.get(cleanUrl).push({ res });
    return;
  }

  // Set up waiters list
  const waiters = [{ res }];
  activeDownloads.set(cleanUrl, waiters);

  let stderrBuffer = '';

  const ytdlp = spawn('yt-dlp', [
    '-f', 'bv*[height<=1080]+ba/bv*[height<=1080]/b[height<=1080]/bestvideo[height<=1080]/best',
    '-o', tempFile,
    '--no-part',           // Download directly to target file without creating .part extension
    '--no-continue',       // Force fresh download, do not attempt range resume
    '--force-overwrites',  // Overwrite existing file
    '--no-warnings',
    '--no-playlist',       // Don't download playlist, only the single video
    cleanUrl
  ]);

  ytdlp.stdout.on('data', (data) => {
    const output = data.toString();
    const chunks = output.split(/[\r\n]+/);
    for (const chunk of chunks) {
      const trimmed = chunk.trim();
      if (!trimmed) continue;
      if (trimmed.includes('[download]')) {
        process.stdout.write(`\r[YouTube Download] ${trimmed}                                \r`);
      } else {
        console.log(`\n[YouTube Download] ${trimmed}`);
      }
    }
  });

  ytdlp.stderr.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) {
      if (msg.includes('[download]')) {
        process.stdout.write(`\r[YouTube Download] ${msg}                                \r`);
      } else {
        console.log(`\n[YouTube Download] ${msg}`);
      }
      stderrBuffer += msg + '\n';
    }
  });

  ytdlp.on('error', (err) => {
    console.error('[YouTube Prepare] Spawn error during download:', err.message);
    const activeWaiters = activeDownloads.get(cleanUrl) || [];
    activeDownloads.delete(cleanUrl);
    activeWaiters.forEach(w => {
      if (!w.res.headersSent) {
        w.res.status(500).json({ error: `Failed to launch yt-dlp: ${err.message}` });
      }
    });
  });

  ytdlp.on('close', (code) => {
    const activeWaiters = activeDownloads.get(cleanUrl) || [];
    activeDownloads.delete(cleanUrl);

    if (code !== 0 && !stderrBuffer.includes('WARNING:')) {
      console.error(`[YouTube Prepare] yt-dlp exited with code ${code}. Stderr: ${stderrBuffer}`);
      let errorMsg = `yt-dlp download failed (code ${code})`;
      if (stderrBuffer.includes('ERROR:')) {
        const errorLines = stderrBuffer.split('\n').filter(l => l.includes('ERROR:'));
        if (errorLines.length > 0) {
          errorMsg = errorLines[0].replace(/^ERROR:\s*/, '');
        }
      }
      activeWaiters.forEach(w => {
        if (!w.res.headersSent) {
          w.res.status(500).json({ error: errorMsg });
        }
      });
    } else {
      try {
        // Dynamically find the largest downloaded file matching youtube_temp in public/
        let targetFile = null;
        let maxSizeBytes = 0;
        const publicDir = path.join(__dirname, 'public');
        
        if (fs.existsSync(publicDir)) {
          const files = fs.readdirSync(publicDir);
          for (const file of files) {
            if (file.startsWith('youtube_temp')) {
              const fullPath = path.join(publicDir, file);
              try {
                const stats = fs.statSync(fullPath);
                if (stats.isFile() && stats.size > maxSizeBytes) {
                  maxSizeBytes = stats.size;
                  targetFile = file;
                }
              } catch (e) {}
            }
          }
        }

        if (targetFile && maxSizeBytes > 0) {
          console.log(`[YouTube Prepare] Download complete. Serving target file: /${targetFile} (${(maxSizeBytes / (1024 * 1024)).toFixed(2)} MB)`);
          activeWaiters.forEach(w => {
            if (!w.res.headersSent) {
              w.res.json({ success: true, videoUrl: `/${targetFile}` });
            }
          });
        } else {
          throw new Error("Downloaded video file not found on disk.");
        }
      } catch (e) {
        console.error("[YouTube Prepare] Post-download error:", e.message);
        activeWaiters.forEach(w => {
          if (!w.res.headersSent) {
            w.res.status(500).json({ error: `File handling error: ${e.message}` });
          }
        });
      }
    }
  });
});

function getChromePath() {
  const platform = os.platform();
  if (platform === 'win32') {
    const paths = [
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
      path.join(os.homedir(), 'AppData\\Local\\Google\\Chrome\\Application\\chrome.exe')
    ];
    for (const p of paths) {
      if (fs.existsSync(p)) return p;
    }
    // Fallback to local appdata env var
    const localAppData = process.env.LOCALAPPDATA;
    if (localAppData) {
      const p = path.join(localAppData, 'Google\\Chrome\\Application\\chrome.exe');
      if (fs.existsSync(p)) return p;
    }
    return 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  } else if (platform === 'darwin') {
    return '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  }
  return 'google-chrome';
}

// Endpoint to fetch candidate image URLs from Bing and DuckDuckGo search in parallel
app.get('/api/candidates', async (req, res) => {
  const { query } = req.query;
  if (!query) {
    return res.status(400).json({ error: 'Query is required.' });
  }

  console.log(`Scraping search candidate URLs for: "${query}"`);
  const urls = new Set();

  // 1. Scrape DuckDuckGo Images
  const fetchDDG = async () => {
    try {
      const mainUrl = `https://duckduckgo.com/?q=${encodeURIComponent(query)}`;
      const response = await fetch(mainUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
      });
      if (!response.ok) {
        console.log(`  [DuckDuckGo] Main page HTTP error: ${response.statusText}`);
        return;
      }

      const html = await response.text();
      const match = html.match(/vqd=([\d-]+)/) || html.match(/vqd=["']([\d-]+)["']/);
      let vqd = null;
      if (match) {
        vqd = match[1];
      } else {
        const match2 = html.match(/vqd\s*:\s*["']([\d-]+)["']/);
        if (match2) vqd = match2[1];
      }

      if (!vqd) {
        console.log(`  [DuckDuckGo] Could not extract VQD token.`);
        return;
      }

      const imgUrl = `https://duckduckgo.com/i.js?o=json&q=${encodeURIComponent(query)}&vqd=${vqd}&f=,,,&p=1`;
      const imgResponse = await fetch(imgUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Accept': 'application/json',
          'Referer': 'https://duckduckgo.com/'
        }
      });

      if (imgResponse.ok) {
        const data = await imgResponse.json();
        if (data.results) {
          let count = 0;
          for (const item of data.results) {
            if (item.image && item.image.startsWith('http')) {
              urls.add(item.image);
              count++;
            }
          }
          console.log(`  [DuckDuckGo] Found ${count} candidates.`);
        }
      } else {
        console.log(`  [DuckDuckGo] Images JSON HTTP error: ${imgResponse.statusText}`);
      }
    } catch (e) {
      console.error(`  [DuckDuckGo] Failed to fetch:`, e.message);
    }
  };

  // 2. Scrape Google Images via Puppeteer
  const fetchGoogle = async () => {
    let browser;
    try {
      browser = await puppeteer.launch({
        executablePath: getChromePath(),
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
      });
      const page = await browser.newPage();
      await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
      
      const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}&tbm=isch`;
      await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 10000 });
      
      // Wait for at least some images to load
      await page.waitForSelector('img', { timeout: 3000 }).catch(() => {});
      
      const googleUrls = await page.evaluate(() => {
        const found = [];
        // Extract high-res image URLs from imgres links
        const links = Array.from(document.querySelectorAll('a'));
        for (const link of links) {
          const href = link.href;
          if (href && href.includes('imgurl=')) {
            try {
              const urlParams = new URLSearchParams(href.split('?')[1]);
              const imgUrl = urlParams.get('imgurl');
              if (imgUrl && imgUrl.startsWith('http')) {
                found.push(imgUrl);
              }
            } catch (e) {}
          }
        }
        
        // Extract gstatic thumbnail image URLs as fallback
        const imgs = Array.from(document.querySelectorAll('img'));
        for (const img of imgs) {
          const src = img.src || img.getAttribute('data-src') || img.getAttribute('data-iurl');
          if (src && src.includes('encrypted-tbn0.gstatic.com')) {
            found.push(src);
          }
        }
        return found;
      });
      
      let count = 0;
      for (const u of googleUrls) {
        urls.add(u);
        count++;
      }
      console.log(`  [Google] Found ${count} candidates.`);
    } catch (e) {
      console.error(`  [Google] Failed to fetch:`, e.message);
    } finally {
      if (browser) {
        try {
          await browser.close();
        } catch (e) {}
      }
    }
  };

  try {
    // Run searches in parallel (DuckDuckGo and Google only)
    await Promise.all([fetchDDG(), fetchGoogle()]);
    const mergedUrls = Array.from(urls);
    console.log(`  [Total] Found ${mergedUrls.length} unique candidates.`);
    res.json({ urls: mergedUrls });
  } catch (err) {
    console.error('Scraping error:', err.message);
    res.status(500).json({ error: `Failed to scrape images: ${err.message}` });
  }
});

// Endpoint to retrieve already processed/saved image URLs
app.get('/api/saved-urls', (req, res) => {
  res.json({ urls: getSavedUrls() });
});

// Endpoint to save a verified image to the member's directory
app.post('/api/save-image', async (req, res) => {
  const { memberId, imageUrl, filename } = req.body;

  if (!memberId || !imageUrl || !filename) {
    return res.status(400).json({ error: 'Missing required parameters: memberId, imageUrl, filename' });
  }

  try {
    const destDir = path.join(__dirname, 'public', 'members', memberId);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }

    let buffer;
    // Check if it is a base64 Data URL uploaded directly from the browser
    if (imageUrl.startsWith('data:')) {
      const base64Data = imageUrl.split(';base64,').pop();
      buffer = Buffer.from(base64Data, 'base64');
    } else {
      // Download image from proxy URL
      const proxyUrl = `http://localhost:${PORT}/api/proxy-image?url=${encodeURIComponent(imageUrl)}`;
      const response = await fetch(proxyUrl);
      
      if (!response.ok) {
        throw new Error(`Failed to download verified image buffer via proxy: ${response.statusText}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      buffer = Buffer.from(arrayBuffer);
    }

    const filePath = path.join(destDir, filename);
    fs.writeFileSync(filePath, buffer);

    // Only add to saved URLs if it's an external web URL
    if (!imageUrl.startsWith('data:')) {
      addSavedUrl(imageUrl); // Save URL to prevent duplicate crawling
    }

    console.log(`[SAVED] Verified image added to: public/members/${memberId}/${filename}`);
    res.json({ success: true, path: `/members/${memberId}/${filename}` });
  } catch (err) {
    console.error('Error saving image:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Endpoint to regenerate members.json database mapping
app.post('/api/regenerate-db', (req, res) => {
  console.log('Regenerating members.json database...');
  try {
    const membersDir = path.join(__dirname, 'public', 'members');
    const allowed = new Set(['johnny', 'taeyong', 'yuta', 'doyoung', 'jaehyun', 'jungwoo', 'haechan']);
    const membersData = [];

    if (!fs.existsSync(membersDir)) {
      return res.status(404).json({ error: 'Members directory not found.' });
    }

    const folders = fs.readdirSync(membersDir);
    for (const folder of folders) {
      if (!allowed.has(folder)) continue;
      const folderPath = path.join(membersDir, folder);
      if (!fs.statSync(folderPath).isDirectory()) continue;

      const files = fs.readdirSync(folderPath)
        .filter(f => !f.startsWith('_') && !f.startsWith('.') && f.toLowerCase().match(/\.(jpg|jpeg|png|webp)$/))
        .sort(); // Sort files alphabetically, excluding _trash/ contents

      if (files.length > 0) {
        const names = {
          johnny: { name: 'Johnny', kor: '쟈니' },
          taeyong: { name: 'Taeyong', kor: '태용' },
          yuta: { name: 'Yuta', kor: '유타' },
          doyoung: { name: 'Doyoung', kor: '도영' },
          jaehyun: { name: 'Jaehyun', kor: '재현' },
          jungwoo: { name: 'Jungwoo', kor: '정우' },
          haechan: { name: 'Haechan', kor: '해찬' }
        };

        membersData.push({
          id: folder,
          name: names[folder].name,
          korName: names[folder].kor,
          images: files.map(f => `/members/${folder}/${f}`)
        });
        console.log(`  ${names[folder].name}: Found ${files.length} images.`);
      }
    }

    fs.writeFileSync(
      path.join(__dirname, 'public', 'members.json'),
      JSON.stringify({ members: membersData }, null, 2),
      'utf-8'
    );

    console.log('Database regenerated successfully.');
    res.json({ success: true });
  } catch (err) {
    console.error('Database regeneration error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Endpoint to batch-move discarded images into _trash folders during DB rebuild
app.post('/api/trash-images', (req, res) => {
  const { images } = req.body; // Array of { memberId, filename, reason }
  if (!images || !Array.isArray(images) || images.length === 0) {
    return res.json({ success: true, moved: 0, errors: 0 });
  }

  let movedCount = 0;
  const errors = [];

  for (const item of images) {
    const { memberId, filename } = item;
    if (!memberId || !filename) continue;

    // Security: prevent path traversal
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) continue;

    const memberDir = path.join(__dirname, 'public', 'members', memberId);
    const trashDir = path.join(memberDir, '_trash');
    const srcPath = path.join(memberDir, filename);
    const destPath = path.join(trashDir, filename);

    try {
      if (!fs.existsSync(srcPath)) continue;
      if (!fs.existsSync(trashDir)) {
        fs.mkdirSync(trashDir, { recursive: true });
      }
      fs.renameSync(srcPath, destPath);
      movedCount++;
    } catch (err) {
      errors.push(`${memberId}/${filename}: ${err.message}`);
    }
  }

  console.log(`[TRASH] Moved ${movedCount}/${images.length} discarded images to _trash folders.`);
  if (errors.length > 0) {
    console.warn(`[TRASH] ${errors.length} move errors:`, errors.slice(0, 5));
  }

  res.json({ success: true, moved: movedCount, errors: errors.length });
});

// Endpoint to save pre-calculated face descriptors to descriptors.json
app.post('/api/save-descriptors', (req, res) => {
  try {
    const { descriptors } = req.body;
    if (!descriptors) {
      return res.status(400).json({ error: 'Missing descriptors data.' });
    }

    const descriptorsPath = path.join(__dirname, 'public', 'descriptors.json');
    fs.writeFileSync(descriptorsPath, JSON.stringify(descriptors), 'utf-8');
    console.log(`[DB] Saved ${descriptors.length} member descriptors to descriptors.json`);

    // Notify Python 512D Service to reload descriptors
    fetch('http://localhost:5001/reload').catch(() => {});

    res.json({ success: true });
  } catch (err) {
    console.error('Error saving descriptors:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Proxy endpoint for 512-D InsightFace ArcFace recognition
app.post('/api/recognize-512d', async (req, res) => {
  try {
    const pyResponse = await fetch('http://localhost:5001/recognize-512d', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    if (!pyResponse.ok) throw new Error(`Python service status ${pyResponse.status}`);
    const data = await pyResponse.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Cross-platform Python executable detection
function getPythonCommand() {
  if (os.platform() === 'win32') {
    return 'python';
  }
  // On macOS/Linux, prefer python3
  try {
    const result = require('child_process').execSync('which python3', { stdio: 'pipe' });
    if (result) return 'python3';
  } catch (e) {}
  return 'python';
}

let py512Process = null;
function startPython512DService() {
  const pythonCmd = getPythonCommand();
  console.log(`[*] Spawning 512-D InsightFace ArcFace Python Service (using ${pythonCmd})...`);
  py512Process = spawn(pythonCmd, [path.join(__dirname, 'python_512d_service.py')], {
    stdio: 'inherit',
    cwd: __dirname
  });
  py512Process.on('error', (err) => {
    console.error('[!] Python 512D Service error:', err.message);
  });

  // Health check: wait for Python service to become ready
  let healthCheckAttempts = 0;
  const maxAttempts = 10;
  const healthCheck = setInterval(async () => {
    healthCheckAttempts++;
    try {
      const resp = await fetch('http://localhost:5001/');
      if (resp.ok) {
        const data = await resp.json();
        console.log(`[OK] Python 512D Service health check passed: ${data.engine || 'active'}`);
        clearInterval(healthCheck);
      }
    } catch (e) {
      if (healthCheckAttempts >= maxAttempts) {
        console.warn('[!] Python 512D Service did not respond after 10 attempts. Face recognition may use browser fallback.');
        clearInterval(healthCheck);
      }
    }
  }, 2000);
}

const server = app.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`   NCT 127 FACE ID EXPRESS SERVER ACTIVE             `);
  console.log(`   Localhost: http://localhost:${PORT}               `);
  console.log(`====================================================`);
  startPython512DService();
});

// Setup WebSocket server for TouchDesigner integration
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws) => {
  console.log('[WS] Client connected. Active clients:', wss.clients.size);
  broadcastClientCount();

  ws.on('message', (message) => {
    try {
      const messageString = message.toString();
      const data = JSON.parse(messageString);
      
      // If it is face tracking data, broadcast it to all other connected clients
      if (data && data.type === 'face_tracking') {
        wss.clients.forEach((client) => {
          if (client !== ws && client.readyState === 1) { // 1 = OPEN
            client.send(messageString);
          }
        });
      }
    } catch (err) {
      console.error('[WS] Error processing message:', err.message);
    }
  });

  ws.on('close', () => {
    console.log('[WS] Client disconnected. Active clients:', wss.clients.size);
    broadcastClientCount();
  });
});

function broadcastClientCount() {
  const countMsg = JSON.stringify({ type: 'client_count', count: wss.clients.size });
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(countMsg);
    }
  });
}

// Upgrade HTTP connection to WebSocket
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});

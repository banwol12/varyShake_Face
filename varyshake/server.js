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

app.use(express.json({ limit: '10mb' }));

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
      if (file.startsWith('youtube_temp.mp4')) {
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

  // Flexible format selection: works with or without ffmpeg
  const ytdlp = spawn('yt-dlp', [
    '-f', 'best[ext=mp4]/best[height<=1080]/bestvideo[height<=1080]+bestaudio/best',
    '-o', tempFile,
    '--no-part',           // Download directly to target file without creating .part extension
    '--no-continue',       // Force fresh download, do not attempt range resume
    '--force-overwrites',  // Overwrite existing file
    '--no-warnings',
    '--no-playlist',       // Don't download playlist, only the single video
    cleanUrl
  ]);

  ytdlp.stderr.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) {
      console.log(`[YouTube Download] ${msg}`);
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

    if (code !== 0) {
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
        if (fs.existsSync(tempFile)) {
          console.log(`[YouTube Prepare] Download complete. Serving file: ${tempFile}`);
          activeWaiters.forEach(w => {
            if (!w.res.headersSent) {
              w.res.json({ success: true, videoUrl: '/youtube_temp.mp4' });
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
        .filter(f => f.toLowerCase().match(/\.(jpg|jpeg|png|webp)$/))
        .sort(); // Sort files alphabetically

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

// Endpoint to save pre-calculated face descriptors to descriptors.json on the server
app.post('/api/save-descriptors', (req, res) => {
  const { descriptors } = req.body;
  if (!descriptors) {
    return res.status(400).json({ error: 'Descriptors data is required.' });
  }

  console.log('Saving pre-calculated face descriptors database...');
  try {
    const filePath = path.join(__dirname, 'public', 'descriptors.json');
    fs.writeFileSync(filePath, JSON.stringify(descriptors, null, 2), 'utf-8');
    console.log('  Descriptors database saved successfully to public/descriptors.json.');
    res.json({ success: true });
  } catch (err) {
    console.error('Error saving descriptors:', err.message);
    res.status(500).json({ error: err.message });
  }
});

const server = app.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`   NCT 127 FACE ID EXPRESS SERVER ACTIVE             `);
  console.log(`   Localhost: http://localhost:${PORT}               `);
  console.log(`====================================================`);
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

// Active members of NCT 127
const MEMBER_INFO = {
  johnny: { name: "Johnny", kor: "쟈니" },
  taeyong: { name: "Taeyong", kor: "태용" },
  yuta: { name: "Yuta", kor: "유타" },
  doyoung: { name: "Doyoung", kor: "도영" },
  jaehyun: { name: "Jaehyun", kor: "재현" },
  jungwoo: { name: "Jungwoo", kor: "정우" },
  haechan: { name: "Haechan", kor: "해찬" }
};

// Custom matching thresholds per member (InsightFace 512D ArcFace Cosine Distance: 0.42 threshold)
const MEMBER_THRESHOLDS = {
  johnny: 0.42,   // ~58%+ similarity confidence
  taeyong: 0.42,  // ~58%+ similarity confidence
  yuta: 0.42,     // ~58%+ similarity confidence
  doyoung: 0.42,  // ~58%+ similarity confidence
  jaehyun: 0.42,  // ~58%+ similarity confidence
  jungwoo: 0.42,  // ~58%+ similarity confidence
  haechan: 0.42   // ~58%+ similarity confidence
};

// Global application state
let faceMatcher = null;
let faceMatcherNoLimit = null;
let activeDetector = 'ssd';
let matchThreshold = 0.42; // 512D ArcFace cosine distance cutoff (concert-optimized)
let stream = null;

// TouchDesigner integration state
let ws = null;
let isTdStreaming = false;
let tdConnectedCount = 0;

let isCameraActive = false;
let labeledDescriptors = [];
let databaseStats = {}; // Map of member_id -> { total, kept, discarded }
let databaseImages = {}; // Map of member_id -> array of image elements

// Track management for temporal webcam voting
let faceTracks = []; // Array of { id, lastBox, history: [], lastSeen: timestamp }
let trackCounter = 0;

// DOM Elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('overlay-canvas');
const loaderOverlay = document.getElementById('loader-overlay');
const loaderStatus = document.getElementById('loader-status');
const progressBar = document.getElementById('progress-bar');
const loaderLogs = document.getElementById('loader-logs');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const detectorSelect = document.getElementById('detector-select');
const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnRelearn = document.getElementById('btn-relearn');
const btnStartCamera = document.getElementById('btn-start-camera');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const memberCatalogList = document.getElementById('member-catalog-list');
const dbDescriptorCount = document.getElementById('db-descriptor-count');
const systemStatusText = document.getElementById('system-status-text');

// New Video & Harvester Elements
const feedSourceSelect = document.getElementById('feed-source-select');
const videoFileGroup = document.getElementById('video-file-group');
const videoFileInput = document.getElementById('video-file-input');
const btnChooseVideo = document.getElementById('btn-choose-video');
const selectedVideoName = document.getElementById('selected-video-name');
const videoHarvesterGroup = document.getElementById('video-harvester-group');
const harvestTargetSelect = document.getElementById('harvest-target-select');
const btnToggleHarvest = document.getElementById('btn-toggle-harvest');
const chkForceTarget = document.getElementById('chk-force-target');
const youtubeUrlGroup = document.getElementById('youtube-url-group');
const youtubeUrlInput = document.getElementById('youtube-url-input');
const btnLoadYoutube = document.getElementById('btn-load-youtube');

// Harvester state
let isHarvestingActive = false;
let lastHarvestTimes = {}; // Map of memberId -> timestamp (for rate limiting)

// TouchDesigner Link DOM Elements
const btnToggleTd = document.getElementById('btn-toggle-td');
const tdStatusBadge = document.getElementById('td-status-badge');
const tdClientCount = document.getElementById('td-client-count');
const btnCopyTdScript = document.getElementById('btn-copy-td-script');

// Helper to write to loader log terminal
function logToLoader(message, type = 'info') {
  const div = document.createElement('div');
  div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  if (type === 'success') div.className = 'log-success';
  if (type === 'error') div.className = 'log-error';
  loaderLogs.appendChild(div);
  loaderLogs.scrollTop = loaderLogs.scrollHeight;
}

// Helper to set loader progress
function setProgress(percent, statusText) {
  progressBar.style.width = `${percent}%`;
  loaderStatus.textContent = statusText;
}

// 1. Initialize Neural Network Models
async function loadModels() {
  setProgress(5, "Loading face detection models...");
  logToLoader("Loading SSD Mobilenet V1 model...");
  await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');

  setProgress(15, "Loading fast detection models...");
  logToLoader("Loading Tiny Face Detector model...");
  await faceapi.nets.tinyFaceDetector.loadFromUri('/models');

  setProgress(25, "Loading landmark models...");
  logToLoader("Loading Face Landmark 68 model...");
  await faceapi.nets.faceLandmark68Net.loadFromUri('/models');

  setProgress(35, "Loading facial extraction models...");
  logToLoader("Loading Face Recognition model...");
  await faceapi.nets.faceRecognitionNet.loadFromUri('/models');

  logToLoader("All models loaded successfully!", "success");
}

// 2. Load reference library (either from Server, LocalStorage Cache or extract from Scratch)
async function buildReferenceLibrary(forceRebuild = false) {
  labeledDescriptors = [];
  databaseStats = {};
  databaseImages = {};

  if (!forceRebuild) {
    // 1. Try loading from Server descriptors.json first (instant load, breaks 5MB localStorage limit)
    logToLoader("Checking Server database for pre-calculated descriptors...");
    try {
      const serverResponse = await fetch('/descriptors.json');
      if (serverResponse.ok) {
        const parsed = await serverResponse.json();
        labeledDescriptors = parsed.map(item => new faceapi.LabeledFaceDescriptors(
          item.label,
          item.descriptors.map(d => new Float32Array(d))
        ));

        logToLoader("✓ Loaded pre-calculated face descriptors from Server database!", "success");

        // Load members.json just to populate the UI catalogs and stats
        try {
          const response = await fetch('/members.json');
          if (response.ok) {
            const db = await response.json();
            db.members.forEach(m => {
              databaseStats[m.id] = { total: m.images.length, kept: m.images.length, discarded: 0 };
              databaseImages[m.id] = m.images;
            });
            renderMemberCatalog(db.members, true);
          }
        } catch (e) { }

        let totalDescriptors = labeledDescriptors.reduce((sum, ld) => sum + ld.descriptors.length, 0);
        dbDescriptorCount.textContent = totalDescriptors;
        faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
        faceMatcherNoLimit = new faceapi.FaceMatcher(labeledDescriptors, 1.0);
        return; // Caching loaded successfully from server, exit early!
      }
    } catch (err) {
      logToLoader("Server descriptors file not found. Checking localStorage...", "info");
    }

    // 2. Try loading from LocalStorage as fallback
    logToLoader("Checking LocalStorage cache for face descriptors...");
    const cachedData = localStorage.getItem('nct127_face_descriptors_v5');

    if (cachedData) {
      try {
        const parsed = JSON.parse(cachedData);
        labeledDescriptors = parsed.map(item => new faceapi.LabeledFaceDescriptors(
          item.label,
          item.descriptors.map(d => new Float32Array(d))
        ));

        logToLoader("✓ Loaded pre-calculated face descriptors from cache!", "success");

        // Load members.json just to populate the UI catalogs and stats
        try {
          const response = await fetch('/members.json');
          if (response.ok) {
            const db = await response.json();
            db.members.forEach(m => {
              databaseStats[m.id] = { total: m.images.length, kept: m.images.length, discarded: 0 };
              databaseImages[m.id] = m.images;
            });
            renderMemberCatalog(db.members, true);
          }
        } catch (e) { }

        let totalDescriptors = labeledDescriptors.reduce((sum, ld) => sum + ld.descriptors.length, 0);
        dbDescriptorCount.textContent = totalDescriptors;
        faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
        faceMatcherNoLimit = new faceapi.FaceMatcher(labeledDescriptors, 1.0);
        return; // Caching loaded successfully from localStorage, exit early!
      } catch (err) {
        logToLoader(`Failed to parse cache: ${err.message}. Rebuilding database...`, "error");
        localStorage.removeItem('nct127_face_descriptors_v5');
      }
    }
  } else {
    logToLoader("Force Rebuild requested. Processing reference images from scratch...", "info");
  }

  // Scratch training
  setProgress(40, "Scanning reference image database...");
  logToLoader("Scanning local folders for NCT 127 photos...");

  let membersList = [];
  try {
    const response = await fetch('/members.json');
    if (!response.ok) throw new Error("Could not load database file members.json");
    const db = await response.json();
    membersList = db.members;
    logToLoader(`Successfully loaded database list with ${membersList.length} members.`, "success");
  } catch (err) {
    logToLoader(`Failed to fetch members.json: ${err.message}`, "error");
    // Fallback list
    membersList = Object.keys(MEMBER_INFO).map(id => ({
      id: id,
      name: MEMBER_INFO[id].name,
      korName: MEMBER_INFO[id].kor,
      images: Array.from({ length: 40 }, (_, i) => `/members/${id}/image_${i + 1}.jpg`)
    }));
  }

  const totalSteps = membersList.length;
  let successDescriptorsCount = 0;
  const discardedImages = []; // Collect filtered-out images to move to _trash

  for (let idx = 0; idx < membersList.length; idx++) {
    const member = membersList[idx];
    const memberId = member.id;
    const engName = member.name;

    setProgress(
      Math.floor(40 + (idx / totalSteps) * 55),
      `Extracting face metrics: ${engName} (${idx + 1}/${totalSteps})...`
    );
    logToLoader(`Processing ${member.images.length} images for ${engName}...`);

    const descriptors = [];
    databaseImages[memberId] = [];
    databaseStats[memberId] = { total: member.images.length, kept: 0, discarded: 0 };

    for (let imgUrl of member.images) {
      try {
        // Fetch and load image
        const img = await faceapi.fetchImage(imgUrl);

        // Detect ALL faces in this reference image to check for group photos
        const detections = await faceapi.detectAllFaces(img)
          .withFaceLandmarks()
          .withFaceDescriptors();

        if (detections.length === 1) {
          const score = detections[0].detection.score;
          if (score > 0.90) {
            descriptors.push(detections[0].descriptor);
            databaseImages[memberId].push(imgUrl);
            databaseStats[memberId].kept++;
          } else {
            databaseStats[memberId].discarded++;
            const fname = imgUrl.split('/').pop();
            discardedImages.push({ memberId, filename: fname, reason: `Detection confidence too low (${Math.round(score * 100)}%)` });
            logToLoader(`  ✗ Discarded ${fname}: Detection confidence too low (${Math.round(score * 100)}%)`);
          }
        } else if (detections.length > 1) {
          databaseStats[memberId].discarded++;
          const fname = imgUrl.split('/').pop();
          discardedImages.push({ memberId, filename: fname, reason: `Group photo (${detections.length} faces)` });
          logToLoader(`  ✗ Discarded ${fname}: Group photo detected (${detections.length} faces)`, "error");
        } else {
          databaseStats[memberId].discarded++;
          const fname = imgUrl.split('/').pop();
          discardedImages.push({ memberId, filename: fname, reason: 'No face detected' });
          logToLoader(`  ✗ Discarded ${fname}: No face detected`, "error");
        }
      } catch (err) {
        databaseStats[memberId].discarded++;
        const fname = imgUrl.split('/').pop();
        discardedImages.push({ memberId, filename: fname, reason: `Load error: ${err.message}` });
      }
    }

    if (descriptors.length > 0) {
      labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(memberId, descriptors));
      successDescriptorsCount += descriptors.length;
      logToLoader(`✓ Completed: Kept ${descriptors.length}/${member.images.length} images for ${engName}.`, "success");
    } else {
      logToLoader(`⚠️ Warning: No valid face descriptors found for ${engName}!`, "error");
    }
  }

  // Move discarded images to _trash folders on the server
  if (discardedImages.length > 0) {
    setProgress(90, `Moving ${discardedImages.length} discarded images to trash...`);
    logToLoader(`Moving ${discardedImages.length} discarded images to _trash folders...`);
    try {
      const trashResp = await fetch('/api/trash-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: discardedImages })
      });
      if (trashResp.ok) {
        const trashData = await trashResp.json();
        logToLoader(`✓ Moved ${trashData.moved} images to _trash folders. (${trashData.errors} errors)`, "success");
      }
    } catch (e) {
      logToLoader(`Failed to move discarded images: ${e.message}`, "error");
    }

    // Regenerate members.json so trashed images are excluded from future loads
    try {
      await fetch('/api/regenerate-db', { method: 'POST' });
      logToLoader("✓ members.json regenerated (trashed images excluded).", "success");
    } catch (e) {
      logToLoader("Failed to regenerate members.json after trashing.", "error");
    }
  }

  setProgress(95, "Saving descriptors database...");
  logToLoader(`Database processed. Total faces trained: ${successDescriptorsCount}`, "success");

  if (labeledDescriptors.length > 0) {
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
    faceMatcherNoLimit = new faceapi.FaceMatcher(labeledDescriptors, 1.0);
    dbDescriptorCount.textContent = successDescriptorsCount;

    const cacheArray = labeledDescriptors.map(ld => ({
      label: ld.label,
      descriptors: ld.descriptors.map(d => Array.from(d))
    }));

    // Save to Server descriptors.json
    try {
      await fetch('/api/save-descriptors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ descriptors: cacheArray })
      });
      logToLoader("✓ Saved pre-calculated descriptors to Server successfully!", "success");
    } catch (e) {
      logToLoader("Failed to save descriptors to server.");
    }

    // Save to LocalStorage cache as fallback
    try {
      localStorage.setItem('nct127_face_descriptors_v5', JSON.stringify(cacheArray));
      logToLoader("Face descriptors cached in LocalStorage for fallback.", "success");
    } catch (e) {
      logToLoader("LocalStorage cache full, server database file will be used.");
    }
  } else {
    logToLoader("CRITICAL: No face descriptors were loaded! Face recognition will not work.", "error");
  }

  renderMemberCatalog(membersList, false);
}

// 3. Render Database List on Right Panel
function renderMemberCatalog(membersList, isCached = false) {
  memberCatalogList.innerHTML = '';

  membersList.forEach(m => {
    const memberId = m.id;
    const engName = m.name;
    const korName = m.korName;
    const successfulImages = databaseImages[memberId] || [];
    const stats = databaseStats[memberId] || { total: 0, kept: 0, discarded: 0 };

    const card = document.createElement('div');
    card.className = 'member-card';
    card.id = `catalog-card-${memberId}`;

    const avatarSrc = successfulImages.length > 0 ? successfulImages[0] : 'data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 fill=%22%23222%22/><text x=%2250%25%22 y=%2255%25%22 font-size=%2212%22 font-family=%22Orbitron%22 fill=%22%23666%22 text-anchor=%22middle%22>NCT</text></svg>';

    let thumbsHTML = '';
    successfulImages.slice(0, 8).forEach(imgUrl => {
      thumbsHTML += `<img src="${imgUrl}" class="member-thumb" alt="Reference Thumb" onerror="this.style.display='none'">`;
    });

    // Status text shows filtering counts
    const statusMsg = isCached
      ? `Cached reference data`
      : `${stats.kept} kept / ${stats.discarded} filtered (Total ${stats.total})`;

    card.innerHTML = `
      <div class="member-avatar-container">
        <img class="member-avatar" src="${avatarSrc}" alt="${engName}">
      </div>
      <div class="member-info">
        <div class="member-name">
          <span>${engName}<span class="member-kor-name">${korName}</span></span>
          <span class="member-badge" id="badge-${memberId}">STANDBY</span>
        </div>
        <div class="member-status" id="status-${memberId}">
          ${statusMsg}
        </div>
        <div class="member-thumbs">
          ${thumbsHTML}
        </div>
      </div>
    `;

    memberCatalogList.appendChild(card);
  });

  document.getElementById('db-member-count').textContent = membersList.length;
}

// 4. Highlight matched member in the database catalog list
function highlightCatalogMember(label) {
  const card = document.getElementById(`catalog-card-${label}`);
  const badge = document.getElementById(`badge-${label}`);
  const statusEl = document.getElementById(`status-${label}`);

  if (card && badge && statusEl) {
    card.classList.add('recognized');
    badge.textContent = 'IDENTIFIED';

    // Automatically revert highlight after a short period if not matched again
    if (card.timeoutId) clearTimeout(card.timeoutId);
    card.timeoutId = setTimeout(() => {
      card.classList.remove('recognized');
      badge.textContent = 'STANDBY';
    }, 1200);
  }
}

// Cosine Distance Calculator with L2 Normalization (Supports 128-D, 512-D and any vector dim)
function calcCosineDistance(vecA, vecB) {
  if (!vecA || !vecB) return 1.0;
  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  const len = Math.min(vecA.length, vecB.length);
  for (let i = 0; i < len; i++) {
    const a = vecA[i];
    const b = vecB[i];
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 1.0;
  const cosSim = dot / (normA * normB);
  return 1.0 - Math.max(-1.0, Math.min(1.0, cosSim));
}

// Top-K NN Matcher supporting Cosine Distance & Variable Vector Dimensions
function findKnnMatch(queryDescriptor, topK = 3) {
  if (!labeledDescriptors || labeledDescriptors.length === 0) {
    return { label: 'unknown', distance: 1.0 };
  }

  let bestLabel = 'unknown';
  let minKnnDist = Infinity;

  for (let ld of labeledDescriptors) {
    const label = ld.label;
    const distances = [];

    for (let desc of ld.descriptors) {
      // Calculate Cosine Distance
      const dist = calcCosineDistance(queryDescriptor, desc);
      distances.push(dist);
    }

    distances.sort((a, b) => a - b);
    const k = Math.min(topK, distances.length);
    if (k === 0) continue;

    const topKDistances = distances.slice(0, k);
    const meanDist = topKDistances.reduce((a, b) => a + b, 0) / k;

    if (meanDist < minKnnDist) {
      minKnnDist = meanDist;
      bestLabel = label;
    }
  }
  return { label: bestLabel, distance: minKnnDist };
}

// High-Speed 512-D InsightFace ArcFace Batch Recognition Client (Multi-Face Parallel)
async function recognizeFaces512DBatch(videoEl, boxes) {
  try {
    if (!videoEl || videoEl.videoWidth === 0 || videoEl.videoHeight === 0 || !boxes || boxes.length === 0) return [];

    const scaleX = videoEl.videoWidth / (displaySize.width || videoEl.clientWidth || 1);
    const scaleY = videoEl.videoHeight / (displaySize.height || videoEl.clientHeight || 1);

    const b64Images = boxes.map(box => {
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = 160;
      cropCanvas.height = 160;
      const cropCtx = cropCanvas.getContext('2d');

      const realX = box.x * scaleX;
      const realY = box.y * scaleY;
      const realW = box.width * scaleX;
      const realH = box.height * scaleY;

      const padW = realW * 0.15;
      const padH = realH * 0.15;
      const sx = Math.max(0, realX - padW);
      const sy = Math.max(0, realY - padH);
      const sw = Math.min(videoEl.videoWidth - sx, realW + padW * 2);
      const sh = Math.min(videoEl.videoHeight - sy, realH + padH * 2);

      cropCtx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, 160, 160);
      return cropCanvas.toDataURL('image/jpeg', 0.85);
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500); // Generous 2.5s timeout for batch

    const resp = await fetch('/api/recognize-512d-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: b64Images, topK: 3 }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (resp.ok) {
      const data = await resp.json();
      return data.results || [];
    }
  } catch (e) {
    // Exception fallback
  }
  return [];
}

// 5. Temporal Webcam Voting & Tracking algorithm
function trackAndSmoothDetections(detections) {
  const now = Date.now();
  const maxDistance = 140; // max pixel distance to associate boxes between frames
  const maxAge = 800; // time in ms to keep dead tracks (reduced for faster cleanup)

  // Clean dead tracks
  faceTracks = faceTracks.filter(t => now - t.lastSeen < maxAge);

  // Associate current detections to existing tracks (prevent duplicate track assignment)
  const assignedTrackIds = new Set();
  const trackedResults = [];

  detections.forEach(det => {
    const box = det.detection.box;
    const center = { x: box.x + box.width / 2, y: box.y + box.height / 2 };

    // Find closest track based on distance
    let bestTrack = null;
    let minD = Infinity;

    faceTracks.forEach(t => {
      if (assignedTrackIds.has(t.id)) return; // Skip already assigned tracks
      const prevCenter = { x: t.lastBox.x + t.lastBox.width / 2, y: t.lastBox.y + t.lastBox.height / 2 };
      const d = Math.hypot(center.x - prevCenter.x, center.y - prevCenter.y);
      if (d < minD && d < maxDistance) {
        minD = d;
        bestTrack = t;
      }
    });

    if (bestTrack) {
      assignedTrackIds.add(bestTrack.id);
    }

    // Use 512-D ArcFace Match if available, else Fallback to KNN
    let rawLabel = 'unknown';
    let distance = 1.0;

    if (det.match512 && det.match512.matched) {
      rawLabel = det.match512.label;
      distance = det.match512.distance;
    } else {
      const bestMatch = findKnnMatch(det.descriptor, 3);
      rawLabel = bestMatch.label;
      distance = bestMatch.distance;
    }

    const similarityPercent = Math.round(Math.max(0, 1 - distance) * 100);
    const targetThreshold = (rawLabel !== 'unknown' && MEMBER_THRESHOLDS[rawLabel]) ? MEMBER_THRESHOLDS[rawLabel] : matchThreshold;
    const resolvedLabel = (rawLabel !== 'unknown' && distance <= targetThreshold) ? rawLabel : 'unknown';

    if (bestTrack) {
      bestTrack.lastSeen = now;
      bestTrack.targetBox = { x: box.x, y: box.y, width: box.width, height: box.height };

      // Only set targetBox here — the 60fps renderLoop handles all smoothing
      if (!bestTrack.smoothBox) {
        bestTrack.smoothBox = { ...bestTrack.targetBox };
      }
      bestTrack.lastBox = bestTrack.smoothBox;

      bestTrack.history.push(resolvedLabel);
      if (bestTrack.history.length > 8) bestTrack.history.shift(); // Keep last 8 frames for faster identity lock

      // Calculate majority vote winner
      const votes = {};
      bestTrack.history.forEach(lbl => { votes[lbl] = (votes[lbl] || 0) + 1; });

      let maxVotes = 0;
      let winnerLabel = 'unknown';
      for (let lbl in votes) {
        if (votes[lbl] > maxVotes) {
          maxVotes = votes[lbl];
          winnerLabel = lbl;
        }
      }

      // Require stability: majority of recent history, minimum 3 votes for faster lock-on
      const finalLabel = maxVotes >= Math.min(3, bestTrack.history.length) ? winnerLabel : 'unknown';

      trackedResults.push({
        detection: { ...det.detection, box: bestTrack.smoothBox },
        descriptor: det.descriptor,
        label: finalLabel,
        distance: distance
      });
    } else {
      // Create a brand new track with smooth initial state
      trackCounter++;
      const initBox = { x: box.x, y: box.y, width: box.width, height: box.height };
      const newTrack = {
        id: trackCounter,
        lastBox: initBox,
        smoothBox: initBox,
        targetBox: initBox,
        history: [resolvedLabel],
        lastSeen: now
      };
      faceTracks.push(newTrack);

      trackedResults.push({
        detection: { ...det.detection, box: initBox },
        descriptor: det.descriptor,
        label: resolvedLabel, // use instant label on first frame
        distance: distance
      });
    }
  });

  return trackedResults;
}

// 6. Webcam Management
async function startCamera() {
  if (isCameraActive) return;

  logToLoader("Accessing webcam...");
  systemStatusText.textContent = "STATUS: INITIALIZING CAMERA";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        facingMode: 'user'
      }
    });

    webcam.srcObject = stream;
    isCameraActive = true;
    document.querySelector('.video-container').classList.add('mirrored');
    cameraPlaceholder.style.display = 'none';
    btnToggleCam.textContent = 'DISABLE CAMERA';
    document.querySelector('.scanner-line').style.display = 'block';
    systemStatusText.textContent = "STATUS: LIVE BIOMETRICS SCAN";
    logToLoader("Webcam feed active.", "success");

    // Start detection loop once video metadata is ready
    const startCamPlayback = () => {
      canvas.width = webcam.videoWidth;
      canvas.height = webcam.videoHeight;
      runRecognitionLoop();
    };

    if (webcam.readyState >= 1) {
      startCamPlayback();
    } else {
      webcam.onloadedmetadata = startCamPlayback;
    }
  } catch (err) {
    logToLoader(`Webcam access failed: ${err.message}`, "error");
    systemStatusText.textContent = "STATUS: ERROR (WEBCAM OFFLINE)";
    console.warn(`Could not start webcam: ${err.message}`);
  }
}

function stopCamera() {
  if (!isCameraActive) return;

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  webcam.srcObject = null;
  isCameraActive = false;
  cameraPlaceholder.style.display = 'flex';
  btnToggleCam.textContent = 'ENABLE CAMERA';
  document.querySelector('.scanner-line').style.display = 'none';
  systemStatusText.textContent = "STATUS: CAMERA OFFLINE";

  // Clear canvas overlay
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  logToLoader("Webcam feed stopped.");
}

// 7. Real-time Face Recognition loop (Separated: AI Detection async + Rendering 60fps)

// Shared state between AI detection and rendering loops
let latestSmoothedResults = [];
let isAiDetectionRunning = false;

async function runRecognitionLoop() {
  console.log("[DEBUG] runRecognitionLoop() called, isCameraActive:", isCameraActive);
  if (!isCameraActive) return;

  const displaySize = { width: webcam.videoWidth, height: webcam.videoHeight };
  console.log("[DEBUG] displaySize:", displaySize);
  faceapi.matchDimensions(canvas, displaySize);

  const ctx = canvas.getContext('2d');

  let lastDetectionStartTime = 0;

  // === AI Detection Loop (runs as fast as the model allows, ~15-30fps) ===
  async function aiDetectionLoop() {
    if (!isCameraActive) return;

    try {
      const now = Date.now();
      // Watchdog: If lock has been held for over 1500ms, force reset it to prevent freezing
      if (isAiDetectionRunning) {
        if (now - lastDetectionStartTime > 1500) {
          console.warn("[WATCHDOG] AI detection loop was stuck, force resetting lock!");
          isAiDetectionRunning = false;
        } else {
          setTimeout(aiDetectionLoop, 16);
          return;
        }
      }
      isAiDetectionRunning = true;
      lastDetectionStartTime = now;

      let detections = [];

      if (activeDetector === 'ssd') {
        detections = await faceapi.detectAllFaces(
          webcam,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.35 })
        ).withFaceLandmarks().withFaceDescriptors();
      } else {
        detections = await faceapi.detectAllFaces(
          webcam,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.30 })
        ).withFaceLandmarks().withFaceDescriptors();
      }

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      if (resizedDetections.length > 0) {
        // High-Speed Batch 512-D Query for ALL detected faces in 1 single HTTP request
        const boxes = resizedDetections.map(d => d.detection.box);
        const match512Results = await recognizeFaces512DBatch(webcam, boxes);

        resizedDetections.forEach((det, i) => {
          if (match512Results && match512Results[i]) {
            det.match512 = match512Results[i];
          }
        });
        latestSmoothedResults = trackAndSmoothDetections(resizedDetections);
      } else {
        latestSmoothedResults = trackAndSmoothDetections([]);
      }

      isAiDetectionRunning = false;
    } catch (err) {
      console.error("Error in AI detection loop:", err);
      isAiDetectionRunning = false;
    }

    // Schedule next AI detection
    if (isCameraActive) {
      setTimeout(aiDetectionLoop, 0);
    }
  }

  // === 60FPS Render Loop (independent, smooth interpolation) ===
  function renderLoop() {
    if (!isCameraActive) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Interpolate all active tracks toward their target positions for buttery smooth rendering
    const now = Date.now();
    const lerpAlpha = 0.45; // Increased for snappier box tracking (was 0.30)

    faceTracks.forEach(track => {
      if (now - track.lastSeen > 800) return; // Skip dead tracks (reduced from 1200ms)
      if (!track.smoothBox || !track.targetBox) return;

      // Lerp smooth box toward target box each render frame
      track.smoothBox = {
        x: track.smoothBox.x + (track.targetBox.x - track.smoothBox.x) * lerpAlpha,
        y: track.smoothBox.y + (track.targetBox.y - track.smoothBox.y) * lerpAlpha,
        width: track.smoothBox.width + (track.targetBox.width - track.smoothBox.width) * lerpAlpha,
        height: track.smoothBox.height + (track.targetBox.height - track.smoothBox.height) * lerpAlpha
      };
      track.lastBox = track.smoothBox;
    });

    // Draw all current results with interpolated positions
    const smoothedResults = latestSmoothedResults;

    if (smoothedResults.length > 0) {
      smoothedResults.forEach(result => {
        // Find this result's track to get the smoothed box
        const resultCenter = {
          x: result.detection.box.x + result.detection.box.width / 2,
          y: result.detection.box.y + result.detection.box.height / 2
        };
        let renderBox = result.detection.box;

        // Find matching track with closest center
        let bestTrack = null;
        let bestDist = Infinity;
        faceTracks.forEach(t => {
          if (!t.smoothBox) return;
          const tc = { x: t.smoothBox.x + t.smoothBox.width / 2, y: t.smoothBox.y + t.smoothBox.height / 2 };
          const d = Math.hypot(resultCenter.x - tc.x, resultCenter.y - tc.y);
          if (d < bestDist && d < 200) {
            bestDist = d;
            bestTrack = t;
          }
        });

        if (bestTrack && bestTrack.smoothBox) {
          renderBox = bestTrack.smoothBox;
        }

        const label = result.label;
        const distance = result.distance;
        const isUnknown = label === 'unknown';

        let displayLabel = '';
        let color = '';

        if (isUnknown) {
          const closest = faceMatcherNoLimit.findBestMatch(result.descriptor);
          const closestName = MEMBER_INFO[closest.label]?.name || closest.label;
          const closestDistance = closest.distance;
          const matchPercent = Math.round((1 - closestDistance) * 100);
          displayLabel = `NO MATCH (BEST: ${closestName.toUpperCase()} [${matchPercent}%])`;
          color = '#ff3333';
        } else {
          const confidence = Math.round((1 - distance) * 100);
          displayLabel = `${MEMBER_INFO[label]?.name.toUpperCase() || label.toUpperCase()} [${confidence}%]`;
          color = '#c9fc00';
        }

        // Draw Glowing Cyber Bracket Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;

        const x = renderBox.x;
        const y = renderBox.y;
        const w = renderBox.width;
        const h = renderBox.height;
        const bracketLength = Math.min(w, h) * 0.15;

        ctx.beginPath();
        ctx.moveTo(x + bracketLength, y); ctx.lineTo(x, y); ctx.lineTo(x, y + bracketLength);
        ctx.moveTo(x + w - bracketLength, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bracketLength);
        ctx.moveTo(x, y + h - bracketLength); ctx.lineTo(x, y + h); ctx.lineTo(x + bracketLength, y + h);
        ctx.moveTo(x + w - bracketLength, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - bracketLength);
        ctx.stroke();

        ctx.shadowBlur = 0;

        ctx.fillStyle = isUnknown ? 'rgba(255, 51, 51, 0.05)' : 'rgba(201, 252, 0, 0.05)';
        ctx.fillRect(x, y, w, h);

        ctx.font = 'bold 11px "Orbitron", monospace';
        const textWidth = ctx.measureText(displayLabel).width;

        ctx.fillStyle = 'rgba(6, 6, 8, 0.85)';
        ctx.fillRect(x, y - 22, textWidth + 16, 18);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y - 22, textWidth + 16, 18);

        ctx.fillStyle = isUnknown ? '#ff5555' : '#c9fc00';
        ctx.fillText(displayLabel, x + 8, y - 9);

        if (!isUnknown) {
          highlightCatalogMember(label);
        }

        // Real-time Video Face Harvesting (STRICT BACKDANCER SHIELD)
        if (isHarvestingActive) {
          const harvestNow = Date.now();
          const harvestTarget = harvestTargetSelect.value;
          const isForceHarvest = chkForceTarget && chkForceTarget.checked && harvestTarget !== 'all';

          if (isForceHarvest) {
            // FORCE SINGLE TARGET MODE: Verify that the detected face actually matches the target member!
            // Calculate distance to the chosen target member's anchor vectors
            let isTrueMatch = false;
            let targetDist = 1.0;

            if (det.match512 && det.match512.matched && det.match512.label === harvestTarget) {
              isTrueMatch = true;
              targetDist = det.match512.distance;
            } else {
              // Fallback check against descriptor matcher
              const bestMatch = findKnnMatch(result.descriptor, 3);
              if (bestMatch.label === harvestTarget && bestMatch.distance <= 0.38) {
                isTrueMatch = true;
                targetDist = bestMatch.distance;
              }
            }

            if (isTrueMatch && targetDist <= 0.38) {
              const lastHarvest = lastHarvestTimes[harvestTarget] || 0;
              const cooldown = 2000;
              if (harvestNow - lastHarvest > cooldown) {
                lastHarvestTimes[harvestTarget] = harvestNow;
                logToLoader(`[HARVEST] Verified face for ${harvestTarget.toUpperCase()} saved. (dist: ${targetDist.toFixed(3)})`, "success");
                harvestFace(harvestTarget, renderBox, targetDist, result.descriptor);
              }
            } else {
              // Ignore background dancers/non-target people in 1-person mode
            }
          } else {
            // MULTI-MEMBER MODE: Only harvest if identity is VERIFIED (NEVER harvest unknown/backdancers)
            if (!isUnknown && label && label !== 'unknown') {
              const lastHarvest = lastHarvestTimes[label] || 0;
              const cooldown = 2000;
              const isTarget = (harvestTarget === 'all' || harvestTarget === label);
              const isPassingCriteria = (distance <= 0.35); // Strict threshold for auto-saving

              if (isTarget && (harvestNow - lastHarvest > cooldown)) {
                if (isPassingCriteria) {
                  lastHarvestTimes[label] = harvestNow;
                  harvestFace(label, renderBox, distance, result.descriptor);
                }
              }
            }
          }
        }
      });
    }

    // Stream to TouchDesigner via WebSocket at 60fps with interpolated coordinates
    if (isTdStreaming && ws && ws.readyState === WebSocket.OPEN) {
      const tdPayload = {
        type: 'face_tracking',
        timestamp: Date.now(),
        width: webcam.videoWidth,
        height: webcam.videoHeight,
        facesCount: smoothedResults.length,
        faces: []
      };

      smoothedResults.forEach(result => {
        // Use interpolated track box for TD too
        const resultCenter = {
          x: result.detection.box.x + result.detection.box.width / 2,
          y: result.detection.box.y + result.detection.box.height / 2
        };
        let box = result.detection.box;
        faceTracks.forEach(t => {
          if (!t.smoothBox) return;
          const tc = { x: t.smoothBox.x + t.smoothBox.width / 2, y: t.smoothBox.y + t.smoothBox.height / 2 };
          if (Math.hypot(resultCenter.x - tc.x, resultCenter.y - tc.y) < 200) {
            box = t.smoothBox;
          }
        });

        const label = result.label;
        const distance = result.distance;
        const isUnknown = label === 'unknown';

        const normX = box.x / webcam.videoWidth;
        const normY = box.y / webcam.videoHeight;
        const normWidth = box.width / webcam.videoWidth;
        const normHeight = box.height / webcam.videoHeight;

        const centerX = box.x + box.width / 2;
        const centerY = box.y + box.height / 2;
        const normCenterX = centerX / webcam.videoWidth;
        const normCenterY = centerY / webcam.videoHeight;

        const rawLandmarks = result.detection.landmarks?.positions || [];
        const landmarks = rawLandmarks.map(pt => ({
          x: Math.round(pt.x),
          y: Math.round(pt.y),
          normX: parseFloat((pt.x / webcam.videoWidth).toFixed(4)),
          normY: parseFloat((pt.y / webcam.videoHeight).toFixed(4))
        }));

        const memberInfo = MEMBER_INFO[label] || { name: 'Unknown', kor: '\uBBF8\uD655\uC778' };

        tdPayload.faces.push({
          id: label,
          name: memberInfo.name,
          korName: memberInfo.kor,
          confidence: isUnknown ? 0 : parseFloat((1 - distance).toFixed(4)),
          isUnknown: isUnknown,
          box: {
            x: Math.round(box.x),
            y: Math.round(box.y),
            width: Math.round(box.width),
            height: Math.round(box.height),
            normX: parseFloat(normX.toFixed(4)),
            normY: parseFloat(normY.toFixed(4)),
            normWidth: parseFloat(normWidth.toFixed(4)),
            normHeight: parseFloat(normHeight.toFixed(4))
          },
          center: {
            x: Math.round(centerX),
            y: Math.round(centerY),
            normX: parseFloat(normCenterX.toFixed(4)),
            normY: parseFloat(normCenterY.toFixed(4))
          },
          landmarks: landmarks
        });
      });

      ws.send(JSON.stringify(tdPayload));
    }

    // Schedule next render frame at 60fps
    requestAnimationFrame(renderLoop);
  }

  // Start both loops independently
  aiDetectionLoop();
  requestAnimationFrame(renderLoop);
}

// 8. Event Listeners & UI Controls
thresholdSlider.addEventListener('input', (e) => {
  matchThreshold = parseFloat(e.target.value);
  thresholdVal.textContent = matchThreshold.toFixed(2);

  if (labeledDescriptors.length > 0) {
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
    faceMatcherNoLimit = new faceapi.FaceMatcher(labeledDescriptors, 1.0);
  }
});

detectorSelect.addEventListener('change', (e) => {
  activeDetector = e.target.value;
  logToLoader(`Face detection model changed to: ${activeDetector === 'ssd' ? 'SSD Mobilenet' : 'Tiny Face Detector'}`);
});

btnToggleCam.addEventListener('click', () => {
  if (isCameraActive) {
    stopActiveFeed();
  } else {
    startCamera();
  }
});

btnStartCamera.addEventListener('click', startCamera);

btnRelearn.addEventListener('click', async () => {
  const confirmReload = confirm("Are you sure you want to rebuild the biometric database? This will clear the cache and re-process all downloaded reference images.");
  if (!confirmReload) return;

  stopCamera();
  localStorage.removeItem('nct127_face_descriptors_v5'); // Clear descriptors cache
  localStorage.removeItem('nct127_boost_completed'); // Clear booster completion flag
  loaderOverlay.classList.remove('fade-out');
  setProgress(10, "Clearing cache and rebuilding db...");

  try {
    await buildReferenceLibrary(true); // Force scratch rebuild to overwrite server JSON file
    setProgress(100, "Ready!");
    setTimeout(() => {
      loaderOverlay.classList.add('fade-out');
      startCamera();
    }, 800);
  } catch (err) {
    logToLoader(`Failed to rebuild reference library: ${err.message}`, "error");
    setProgress(100, "Error during rebuild!");
  }
});

// New Video & Harvester Elements Event Listeners
feedSourceSelect.addEventListener('change', (e) => {
  const mode = e.target.value;
  stopActiveFeed();

  if (mode === 'webcam') {
    videoFileGroup.classList.add('hidden');
    youtubeUrlGroup.classList.add('hidden');
    videoHarvesterGroup.classList.add('hidden');
    btnToggleCam.style.display = 'inline-block';
    btnStartCamera.style.display = 'inline-block';
  } else if (mode === 'video') {
    videoFileGroup.classList.remove('hidden');
    youtubeUrlGroup.classList.add('hidden');
    videoHarvesterGroup.classList.remove('hidden');
    btnToggleCam.style.display = 'none';
    btnStartCamera.style.display = 'none';

    // Auto-load test.mp4 as a default demo video
    loadVideoFeed('/test.mp4', 'test.mp4 (Default Demo Video)');
  } else if (mode === 'youtube') {
    videoFileGroup.classList.add('hidden');
    youtubeUrlGroup.classList.remove('hidden');
    videoHarvesterGroup.classList.remove('hidden');
    btnToggleCam.style.display = 'none';
    btnStartCamera.style.display = 'none';
  }
});

// Harvester Target Select Change Listener to disable chkForceTarget if 'all' is selected
harvestTargetSelect.addEventListener('change', (e) => {
  if (chkForceTarget) {
    if (e.target.value === 'all') {
      chkForceTarget.checked = false;
      chkForceTarget.disabled = true;
    } else {
      chkForceTarget.disabled = false;
    }
  }
});

btnChooseVideo.addEventListener('click', () => {
  videoFileInput.click();
});

videoFileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fileUrl = URL.createObjectURL(file);
  loadVideoFeed(fileUrl, file.name);
});

btnLoadYoutube.addEventListener('click', async () => {
  const url = youtubeUrlInput.value.trim();
  if (!url) {
    alert("Please enter a YouTube video URL.");
    return;
  }

  // Extract video ID for clean naming
  let videoId = 'YouTube Video';
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
  const match = url.match(regExp);
  if (match && match[2].length === 11) {
    videoId = match[2];
  }

  logToLoader(`Preparing YouTube video (downloading 1080p Full HD locally)...`, "info");
  btnLoadYoutube.disabled = true;
  btnLoadYoutube.textContent = "DOWNLOADING...";

  try {
    const response = await fetch(`/api/youtube-prepare?url=${encodeURIComponent(url)}`);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({ error: 'Unknown server error' }));
      throw new Error(errData.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    logToLoader(`✓ YouTube video prepared successfully! Starting playback...`, "success");
    loadVideoFeed(data.videoUrl, `YouTube ID: ${videoId}`);
  } catch (err) {
    logToLoader(`Failed to prepare YouTube video: ${err.message}`, "error");
    alert(`Could not load YouTube video: ${err.message}`);
  } finally {
    btnLoadYoutube.disabled = false;
    btnLoadYoutube.textContent = "LOAD VIDEO";
  }
});

function loadVideoFeed(srcUrl, name) {
  stopActiveFeed();
  selectedVideoName.textContent = name;
  document.querySelector('.video-container').classList.remove('mirrored');

  webcam.srcObject = null;
  webcam.src = srcUrl;
  webcam.loop = true; // Loop video for continuous testing
  isCameraActive = true;
  cameraPlaceholder.style.display = 'none';
  document.querySelector('.scanner-line').style.display = 'block';
  systemStatusText.textContent = "STATUS: VIDEO BIOMETRICS SCAN";

  logToLoader(`Loaded video: ${name}`, "success");

  // Automatically activate the face harvester for convenience
  setTimeout(() => {
    toggleHarvesting(true);
  }, 1000);

  const startPlayback = () => {
    // If the browser hasn't populated videoWidth/videoHeight yet, wait and retry shortly
    if (webcam.videoWidth === 0 || webcam.videoHeight === 0) {
      console.log("[DEBUG] videoWidth/videoHeight is 0, retrying startPlayback in 100ms...");
      setTimeout(startPlayback, 100);
      return;
    }

    console.log(`[DEBUG] Video size confirmed: ${webcam.videoWidth}x${webcam.videoHeight}`);
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    webcam.play()
      .then(() => {
        console.log("[DEBUG] Video playback started successfully.");
        runRecognitionLoop();
      })
      .catch(e => {
        console.warn("Video play failed:", e.message);
        // Fallback: run loop anyway
        runRecognitionLoop();
      });
  };

  // Listen to both loadedmetadata and canplay to cover different browser behaviors
  webcam.onloadedmetadata = startPlayback;
  webcam.oncanplay = startPlayback;

  // Also check if already ready
  if (webcam.readyState >= 2 && webcam.videoWidth > 0) {
    startPlayback();
  } else {
    // Fail-safe timeout to force attempt if events missed
    setTimeout(startPlayback, 500);
  }
}

btnToggleHarvest.addEventListener('click', () => {
  toggleHarvesting(!isHarvestingActive);
});

function toggleHarvesting(active) {
  isHarvestingActive = active;
  if (isHarvestingActive) {
    btnToggleHarvest.textContent = 'STOP HARVESTING';
    btnToggleHarvest.classList.add('active');
    logToLoader(`Video Face Harvester started. Target: ${harvestTargetSelect.value.toUpperCase()}`, "success");
  } else {
    btnToggleHarvest.textContent = 'START HARVESTING';
    btnToggleHarvest.classList.remove('active');
    logToLoader("Video Face Harvester stopped.");
  }
}

function stopActiveFeed() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  webcam.srcObject = null;
  webcam.src = '';
  isCameraActive = false;

  cameraPlaceholder.style.display = 'flex';
  btnToggleCam.textContent = 'ENABLE FEED';
  document.querySelector('.scanner-line').style.display = 'none';
  systemStatusText.textContent = "STATUS: FEED OFFLINE";

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (isHarvestingActive) {
    toggleHarvesting(false);
  }
}

function harvestFace(memberId, box, distance, descriptor) {
  try {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');

    // Crop with 15% padding
    const paddingX = box.width * 0.15;
    const paddingY = box.height * 0.15;

    let cropX = Math.max(0, box.x - paddingX);
    let cropY = Math.max(0, box.y - paddingY);
    let cropW = Math.min(webcam.videoWidth - cropX, box.width + paddingX * 2);
    let cropH = Math.min(webcam.videoHeight - cropY, box.height + paddingY * 2);

    tempCanvas.width = 300;
    tempCanvas.height = 300;

    tempCtx.drawImage(
      webcam,
      cropX, cropY, cropW, cropH,
      0, 0, 300, 300
    );

    let dataUrl;
    try {
      dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
    } catch (corsErr) {
      console.warn('[HARVEST] Canvas tainted by CORS — cannot crop face from this video source:', corsErr.message);
      logToLoader(`[HARVEST] CORS error: Cannot crop from this video source. Use a local file instead.`, "error");
      return;
    }

    let maxAutoIdx = 99;
    if (databaseImages[memberId]) {
      databaseImages[memberId].forEach(imgPath => {
        const match = imgPath.match(/\/auto_(\d+)\./);
        if (match) {
          const idx = parseInt(match[1], 10);
          if (!isNaN(idx) && idx > maxAutoIdx) maxAutoIdx = idx;
        }
      });
    }
    const autoIndex = maxAutoIdx + 1;
    const filename = `auto_${autoIndex}.jpg`;
    const similarity = Math.round((1 - distance) * 100);

    logToLoader(`[HARVEST] Harvesting face for ${memberId.toUpperCase()} (${similarity}%)...`, "info");

    fetch('/api/save-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        memberId: memberId,
        imageUrl: dataUrl,
        filename: filename
      })
    })
      .then(async (response) => {
        if (response.ok) {
          logToLoader(`[HARVEST] ✓ Saved auto_${autoIndex}.jpg for ${memberId.toUpperCase()}!`, "success");

          // 1. Add descriptor directly to memory
          const memberLd = labeledDescriptors.find(ld => ld.label === memberId);
          if (memberLd && descriptor) {
            memberLd.descriptors.push(descriptor);
          }

          // 1.5. Add saved image path to local databaseImages catalog
          if (!databaseImages[memberId]) databaseImages[memberId] = [];
          const savedPath = `/members/${memberId}/${filename}`;
          if (!databaseImages[memberId].includes(savedPath)) {
            databaseImages[memberId].push(savedPath);
          }

          // 2. Update the active FaceMatchers live without page reloads
          faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
          faceMatcherNoLimit = new faceapi.FaceMatcher(labeledDescriptors, 1.0);

          // 3. Update overall face descriptors count in UI
          let totalDescriptors = labeledDescriptors.reduce((sum, ld) => sum + ld.descriptors.length, 0);
          dbDescriptorCount.textContent = totalDescriptors;

          // 4. Save updated descriptors database to Server descriptors.json
          const cacheArray = labeledDescriptors.map(ld => ({
            label: ld.label,
            descriptors: ld.descriptors.map(d => Array.from(d))
          }));

          fetch('/api/save-descriptors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ descriptors: cacheArray })
          }).catch(e => console.error("Failed to save descriptors to server:", e));

          // 5. Update localStorage cache
          try {
            localStorage.setItem('nct127_face_descriptors_v5', JSON.stringify(cacheArray));
          } catch (e) { }

          // 6. Regenerate database mapping and update UI
          fetch('/api/regenerate-db', { method: 'POST' })
            .then(async (r) => {
              if (r.ok) {
                const memberListResp = await fetch('/members.json');
                if (memberListResp.ok) {
                  const db = await memberListResp.json();
                  db.members.forEach(m => {
                    databaseStats[m.id] = { total: m.images.length, kept: m.images.length, discarded: 0 };
                    databaseImages[m.id] = m.images;
                  });
                  renderMemberCatalog(db.members, false);
                }
              }
            });
        } else {
          logToLoader(`[HARVEST] Failed to save image for ${memberId.toUpperCase()}`, "error");
        }
      })
      .catch((e) => {
        console.error("Harvester upload error:", e);
      });
  } catch (err) {
    console.error("Failed to crop face frame:", err);
  }
}

// 9. TouchDesigner Integration Connection Logic
function initTdWebSocket() {
  if (ws) {
    ws.close();
    ws = null;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // Default to localhost:5173 to support local files (file://) or frontend page served from other ports (e.g. Live Server)
  let wsUrl = `${protocol}//localhost:5173`;
  if (window.location.protocol !== 'file:' && window.location.hostname) {
    wsUrl = `${protocol}//${window.location.hostname}:5173`;
  }

  // Update UI with URL
  const tdWsUrlEl = document.getElementById('td-ws-url');
  if (tdWsUrlEl) {
    tdWsUrlEl.textContent = wsUrl;
  }

  logToLoader(`Opening TouchDesigner WebSocket stream at ${wsUrl}...`);
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] Connected to Face ID WebSocket Server');
    if (tdStatusBadge) {
      tdStatusBadge.textContent = 'ACTIVE';
      tdStatusBadge.className = 'td-status-badge active';
    }
    if (btnToggleTd) {
      btnToggleTd.textContent = 'STOP TD STREAM';
      btnToggleTd.classList.remove('primary-btn');
      btnToggleTd.classList.add('secondary-btn');
    }
    logToLoader("TouchDesigner WebSocket stream active.", "success");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data && data.type === 'client_count') {
        tdConnectedCount = data.count;
        if (tdClientCount) {
          // Subtract 1 since the browser itself is 1 client
          tdClientCount.textContent = Math.max(0, tdConnectedCount - 1);
        }
      }
    } catch (e) {
      // Ignore
    }
  };

  ws.onclose = () => {
    console.log('[WS] Disconnected from WebSocket Server');
    if (tdStatusBadge) {
      tdStatusBadge.textContent = 'OFFLINE';
      tdStatusBadge.className = 'td-status-badge disconnected';
    }
    if (btnToggleTd) {
      btnToggleTd.textContent = 'START TD STREAM';
      btnToggleTd.classList.add('primary-btn');
      btnToggleTd.classList.remove('secondary-btn');
    }
    if (tdClientCount) {
      tdClientCount.textContent = '0';
    }

    // Auto-reconnect if streaming is active
    if (isTdStreaming) {
      setTimeout(initTdWebSocket, 3000);
    }
  };

  ws.onerror = (err) => {
    console.error('[WS] WebSocket Error:', err);
  };
}

function stopTdWebSocket() {
  isTdStreaming = false;
  if (ws) {
    ws.close();
    ws = null;
  }
  if (tdStatusBadge) {
    tdStatusBadge.textContent = 'OFFLINE';
    tdStatusBadge.className = 'td-status-badge disconnected';
  }
  if (btnToggleTd) {
    btnToggleTd.textContent = 'START TD STREAM';
    btnToggleTd.classList.add('primary-btn');
    btnToggleTd.classList.remove('secondary-btn');
  }
  if (tdClientCount) {
    tdClientCount.textContent = '0';
  }
  logToLoader("TouchDesigner WebSocket stream stopped.");
}

// Bind TouchDesigner toggle button
if (btnToggleTd) {
  btnToggleTd.addEventListener('click', () => {
    if (isTdStreaming) {
      stopTdWebSocket();
    } else {
      isTdStreaming = true;
      initTdWebSocket();
    }
  });
}

// Bind TD Script Copy Button
if (btnCopyTdScript) {
  btnCopyTdScript.addEventListener('click', () => {
    const pythonScript = `import json

# 이 스크립트를 TouchDesigner의 "Web Socket DAT" 콜백 스크립트에 그대로 붙여넣으세요.
# 이 스크립트는 수신된 얼굴 인식 좌표 데이터를 실시간으로 파싱하여 같은 컨테이너 내의 "face_data"라는 이름의 Table DAT에 씁니다.
# 미리 "Table DAT"을 생성하고 이름을 "face_data"로 설정해주어야 정상 작동합니다.

def onConnect(dat):
	print("Connected to Face ID server.")
	return

def onDisconnect(dat):
	print("Disconnected from Face ID server.")
	return

def onReceiveText(dat, rowIndex, message):
	try:
		data = json.loads(message)
		
		# 클라이언트 수 카운트 메시지는 무시
		if data.get('type') == 'client_count':
			return

		# 데이터를 기록할 대상 Table DAT 찾기
		table = op('face_data')
		if not table:
			return
			
		# 테이블 초기화 및 헤더 작성
		table.clear()
		table.appendRow(['id', 'name', 'confidence', 'x', 'y', 'w', 'h', 'centerX', 'centerY'])
		
		faces = data.get('faces', [])
		for face in faces:
			box = face.get('box', {})
			center = face.get('center', {})
			table.appendRow([
				face.get('id', ''),
				face.get('name', ''),
				round(face.get('confidence', 0), 4),
				round(box.get('normX', 0), 4),
				round(box.get('normY', 0), 4),
				round(box.get('normWidth', 0), 4),
				round(box.get('normHeight', 0), 4),
				round(center.get('normX', 0), 4),
				round(center.get('normY', 0), 4)
			])
	except Exception as e:
		print("Error parsing WebSocket message:", e)
	return

def onReceiveBinary(dat, contents):
	return
`;
    navigator.clipboard.writeText(pythonScript)
      .then(() => {
        const originalText = btnCopyTdScript.textContent;
        btnCopyTdScript.textContent = 'COPIED TO CLIPBOARD!';
        btnCopyTdScript.style.backgroundColor = 'var(--primary)';
        btnCopyTdScript.style.color = '#000';
        setTimeout(() => {
          btnCopyTdScript.textContent = originalText;
          btnCopyTdScript.style.backgroundColor = '';
          btnCopyTdScript.style.color = '';
        }, 2000);
      })
      .catch(err => {
        console.error('Failed to copy TD script:', err);
        alert('Script copied to console. Open inspector (F12) to copy it manually.');
        console.log(pythonScript);
      });
  });
}

// Initialize Application
async function init() {
  // Disable force checkbox initially since 'all' is selected by default
  if (chkForceTarget) {
    chkForceTarget.disabled = true;
  }
  try {
    await loadModels();
    await buildReferenceLibrary();

    setProgress(100, "All systems operational!");
    logToLoader("System fully initialized. Ready for operations.", "success");

    setTimeout(() => {
      loaderOverlay.classList.add('fade-out');
      startCamera();

      // Automatically trigger silent database expansion on startup
      startAutoExpansion(true);
    }, 1200);
  } catch (err) {
    logToLoader(`System initialization failed: ${err.message}`, "error");
    setProgress(100, "Initialization failed!");
    console.error(err);
  }
}

// Booster DOM Elements
const btnStartBoost = document.getElementById('btn-start-boost');
const boosterStatusContainer = document.getElementById('booster-status-container');
const boostCurrentMember = document.getElementById('boost-current-member');
const boostCurrentProgress = document.getElementById('boost-current-progress');
const boostProgressBar = document.getElementById('boost-progress-bar');
const boosterLogs = document.getElementById('booster-logs');

// Active booster state
let isBoostingActive = false;
let processedUrls = new Set();
const QUERY_VARIATIONS = ['직캠', '무대', '콘서트', '가요대전', '음악방송', '팬미팅', '프로필', '화보', '퍼포먼스'];

// Helper to write to booster logs
function logToBooster(message, type = 'info') {
  if (!boosterLogs) return;
  const div = document.createElement('div');
  div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  if (type === 'success') div.className = 'log-line success';
  else if (type === 'error') div.className = 'log-line error';
  else if (type === 'system') div.className = 'log-line system';
  else div.className = 'log-line';
  boosterLogs.appendChild(div);
  boosterLogs.scrollTop = boosterLogs.scrollHeight;
}

// Biometric Euclidean distance calculation (Supports dynamic 128D / 512D vectors)
function getEuclideanDistance(arr1, arr2) {
  if (!arr1 || !arr2 || arr1.length === 0 || arr2.length === 0) return 1.0;
  // If dimensions match, compute full Euclidean distance
  if (arr1.length === arr2.length) {
    let sum = 0;
    for (let i = 0; i < arr1.length; i++) {
      const diff = arr1[i] - arr2[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }
  // If dimensions differ (e.g., 128D webcam descriptor vs 512D server descriptor)
  const len = Math.min(arr1.length, arr2.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const diff = arr1[i] - arr2[i];
    sum += diff * diff;
  }
  // Scale distance proportionally for dimensional mismatch
  return Math.sqrt(sum * (Math.max(arr1.length, arr2.length) / len));
}

// Booster implementation running in the browser (supports interactive or background silent mode)
async function startAutoExpansion(silent = false) {
  // If already running, just print status and reveal the panel if not silent
  if (isBoostingActive) {
    if (!silent && boosterStatusContainer) {
      boosterStatusContainer.classList.remove('hidden');
      logToBooster("Re-attached to active background booster loop.", "system");
    }
    return;
  }

  isBoostingActive = true;

  if (btnStartBoost) {
    btnStartBoost.disabled = true;
    btnStartBoost.textContent = "AUTO-BOOST ACTIVE...";
  }
  if (boosterStatusContainer) {
    boosterStatusContainer.classList.remove('hidden');
  }
  if (boosterLogs) {
    boosterLogs.innerHTML = "";
  }

  logToBooster(`Starting Continuous Biometric Accuracy Booster (Mode: ${silent ? 'Silent Background' : 'Interactive'})...`, "system");
  logToBooster("Verification loops continuously in the background while keeping the live camera feed active.", "system");

  // Load already processed URLs from the server to prevent duplicates
  try {
    const urlsResp = await fetch('/api/saved-urls');
    if (urlsResp.ok) {
      const urlsData = await urlsResp.json();
      processedUrls = new Set(urlsData.urls);
      logToBooster(`Loaded ${processedUrls.size} already processed URLs from server cache.`, "system");
    }
  } catch (e) {
    console.error("Failed to load processed URLs:", e);
  }

  let membersList = [];
  try {
    const response = await fetch('/members.json');
    if (!response.ok) throw new Error("Could not load database file.");
    const data = await response.json();
    membersList = data.members;
  } catch (err) {
    logToBooster(`Error loading member list: ${err.message}`, "error");
    isBoostingActive = false;
    if (btnStartBoost) {
      btnStartBoost.disabled = false;
      btnStartBoost.textContent = "START AUTO-EXPANSION";
    }
    return;
  }

  let queryVariationIndex = 0;

  // Outer continuous learning loop (Indefinite background operation)
  while (true) {
    const currentQuerySuffix = QUERY_VARIATIONS[queryVariationIndex];
    logToBooster(`==============================================`, "system");
    logToBooster(`[ROUND START] Searching variation: "${currentQuerySuffix}"`, "system");
    logToBooster(`Scraping candidates for all members in parallel...`);

    // 1. Fetch search candidates for ALL members in parallel for the current variation
    const candidatesMap = {};
    const candidatePromises = membersList.map(async (member) => {
      const memberId = member.id;
      const engName = member.name;
      const korName = member.korName || engName;
      const query = `NCT ${korName} ${currentQuerySuffix}`;

      try {
        const resp = await fetch(`/api/candidates?query=${encodeURIComponent(query)}`);
        if (resp.ok) {
          const data = await resp.json();
          if (data.urls) {
            // Filter out URLs we've already processed to save network and computation!
            candidatesMap[memberId] = data.urls.filter(url => !processedUrls.has(url));
            console.log(`Candidates for ${korName} (${currentQuerySuffix}): ${data.urls.length} found, ${candidatesMap[memberId].length} new.`);
          }
        }
      } catch (e) {
        console.error(`Failed to fetch candidates for ${korName}:`, e);
      }
      if (!candidatesMap[memberId]) {
        candidatesMap[memberId] = [];
      }
    });

    await Promise.all(candidatePromises);

    // 2. Perform Round-Robin verification across all members (ensures balanced data collection)
    let roundRobinActive = true;
    const urlIndices = {};
    const successCounts = {};
    membersList.forEach(m => {
      urlIndices[m.id] = 0;
      successCounts[m.id] = 0;
    });

    const maxAdditionsPerRound = 10; // Limit per member per variation round to maintain balance

    while (roundRobinActive) {
      roundRobinActive = false;

      for (let idx = 0; idx < membersList.length; idx++) {
        const member = membersList[idx];
        const memberId = member.id;
        const engName = member.name;
        const korName = member.korName || engName;

        // Skip if this member has hit their per-round limit
        if (successCounts[memberId] >= maxAdditionsPerRound) {
          continue;
        }

        const candidates = candidatesMap[memberId];
        const currentUrlIdx = urlIndices[memberId];

        // Skip if this member has no more new candidates
        if (!candidates || currentUrlIdx >= candidates.length) {
          continue;
        }

        // We have at least one active member with remaining candidates
        roundRobinActive = true;

        // UI Updates
        if (boostCurrentMember) {
          boostCurrentMember.textContent = `${korName} (${currentQuerySuffix})`;
        }

        // Calculate progress across all members in this round
        let totalCandidatesCount = 0;
        let processedCandidatesCount = 0;
        membersList.forEach(m => {
          totalCandidatesCount += candidatesMap[m.id].length;
          processedCandidatesCount += urlIndices[m.id];
        });
        const percent = totalCandidatesCount > 0 ? Math.round((processedCandidatesCount / totalCandidatesCount) * 100) : 100;
        if (boostCurrentProgress) boostCurrentProgress.textContent = `${percent}%`;
        if (boostProgressBar) boostProgressBar.style.width = `${percent}%`;

        const candidateUrl = candidates[currentUrlIdx];
        urlIndices[memberId]++; // Advance candidate index

        // Double check duplicate check set
        if (processedUrls.has(candidateUrl)) {
          continue;
        }

        // Pre-emptively add to processedUrls to prevent duplicate scans
        processedUrls.add(candidateUrl);

        // Run biometric comparison
        try {
          // Find anchor templates
          const memberLd = labeledDescriptors.find(ld => ld.label === memberId);
          if (!memberLd || memberLd.descriptors.length === 0) {
            continue;
          }

          const proxyUrl = `/api/proxy-image?url=${encodeURIComponent(candidateUrl)}`;
          const img = await faceapi.fetchImage(proxyUrl);

          const detection = await faceapi.detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.85 }))
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detection) {
            const distances = memberLd.descriptors.map(anchor => getEuclideanDistance(detection.descriptor, anchor));
            const minDistance = Math.min(...distances);
            const similarity = Math.round((1 - minDistance) * 100);

            // Collect only with 73% or more similarity (minDistance <= 0.27) to allow more training data
            if (minDistance <= 0.27) {
              // Duplicate check: If distance is too close to an existing template (distance < 0.06),
              // it represents the exact same image content. We skip it to avoid duplicates.
              const isDuplicate = distances.some(dist => dist < 0.06);
              if (isDuplicate) {
                console.log(`[${korName}] Candidate discarded: duplicate image detected (distance: ${minDistance.toFixed(3)}).`);
                continue;
              }

              let maxAutoIdx = 99;
              if (databaseImages[memberId]) {
                databaseImages[memberId].forEach(imgPath => {
                  const match = imgPath.match(/\/auto_(\d+)\./);
                  if (match) {
                    const idx = parseInt(match[1], 10);
                    if (!isNaN(idx) && idx > maxAutoIdx) maxAutoIdx = idx;
                  }
                });
              }
              const autoIndex = maxAutoIdx + 1;

              logToBooster(`[${korName}] ✓ VERIFIED! Similarity: ${similarity}% (Distance: ${minDistance.toFixed(3)}). Saving auto_${autoIndex}.jpg...`, "success");

              const saveResp = await fetch('/api/save-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  memberId: memberId,
                  imageUrl: candidateUrl,
                  filename: `auto_${autoIndex}.jpg`
                })
              });

              if (saveResp.ok) {
                // 1. Add descriptor directly to memory
                memberLd.descriptors.push(detection.descriptor);

                // 2. Add image path to local catalog record
                if (!databaseImages[memberId]) databaseImages[memberId] = [];
                const savedPath = `/members/${memberId}/auto_${autoIndex}.jpg`;
                databaseImages[memberId].push(savedPath);

                // 3. Update local catalog statistics
                if (databaseStats[memberId]) {
                  databaseStats[memberId].total++;
                  databaseStats[memberId].kept++;
                }

                // 4. Update the active FaceMatchers live without page reloads
                faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
                faceMatcherNoLimit = new faceapi.FaceMatcher(labeledDescriptors, 1.0);

                // 5. Update overall face descriptors count in UI
                let totalDescriptors = labeledDescriptors.reduce((sum, ld) => sum + ld.descriptors.length, 0);
                dbDescriptorCount.textContent = totalDescriptors;

                // 6. Re-render catalog cards to display new thumbnails
                renderMemberCatalog(membersList, false);

                // 7. Save updated descriptors database to Server descriptors.json
                const cacheArray = labeledDescriptors.map(ld => ({
                  label: ld.label,
                  descriptors: ld.descriptors.map(d => Array.from(d))
                }));
                fetch('/api/save-descriptors', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ descriptors: cacheArray })
                }).catch(e => console.error("Failed to save descriptors to server:", e));

                successCounts[memberId]++;
              }
            } else {
              console.log(`[${korName}] Candidate discarded: similarity too low (${similarity}%).`);
            }
          }
        } catch (err) {
          console.warn(`Error processing candidate for ${korName}:`, err.message);
        }

        // Yield to UI rendering loop to keep webcam feed perfectly smooth (1.5 seconds)
        await new Promise(r => setTimeout(r, 1500));
      }
    }

    logToBooster(`[ROUND COMPLETE] Finished search variation: "${currentQuerySuffix}"`, "system");
    logToBooster("Updating active database index on server...", "system");

    try {
      const regenResp = await fetch('/api/regenerate-db', { method: 'POST' });
      if (regenResp.ok) {
        // Update local cache
        const cacheArray = labeledDescriptors.map(ld => ({
          label: ld.label,
          descriptors: ld.descriptors.map(d => Array.from(d))
        }));
        localStorage.setItem('nct127_face_descriptors_v5', JSON.stringify(cacheArray));
        localStorage.setItem('nct127_boost_completed', 'true');

        // Save to Server descriptors.json
        try {
          await fetch('/api/save-descriptors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ descriptors: cacheArray })
          });
          logToBooster("✓ Server database index, descriptors.json, and LocalStorage synced.", "success");
        } catch (e) {
          console.error("Failed to save descriptors to server:", e);
          logToBooster("✓ Server database index and LocalStorage synced.", "success");
        }
      }
    } catch (err) {
      console.error("Failed to regenerate database index:", err);
    }

    // Cycle to next search variation
    queryVariationIndex = (queryVariationIndex + 1) % QUERY_VARIATIONS.length;

    logToBooster(`Sleeping for 30 seconds before launching next variation...`, "system");

    // Smooth countdown in logs
    for (let c = 30; c > 0; c -= 10) {
      if (boostCurrentMember) boostCurrentMember.textContent = `SLEEP (${c}s)`;
      await new Promise(r => setTimeout(r, 10000));
    }
  }
}

// Bind booster event (default is interactive mode)
if (btnStartBoost) {
  btnStartBoost.addEventListener('click', () => startAutoExpansion(false));
}

// Kick off initialization
if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

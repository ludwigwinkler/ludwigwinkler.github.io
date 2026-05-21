---
layout: post
title:  "Quantum Physics"
category: blog
date:   2026-05-18
excerpt: "TBD"
highlighter: rouge
# image: "/blog/blogthumbnails/quantum.png"
---
{% include mathjax3.html %}

<div id="qp-atoms-wave" style="width:100%;max-width:760px;margin:1.5em auto;border:1px solid #ddd;border-radius:8px;background:#ffffff;color:#222;font-family:sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;">
    <span style="font-size:0.9em;letter-spacing:0.05em;text-transform:uppercase;color:#556;">From particles to waves</span>
    <div>
      <button id="qp-play" style="background:#1f6feb;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;">Play</button>
      <button id="qp-reset" style="background:#6e7681;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;margin-left:4px;">Reset</button>
    </div>
  </div>
  <canvas id="qp-canvas" width="760" height="380" style="width:100%;height:auto;display:block;background:#ffffff;"></canvas>
  <div style="padding:6px 12px;font-size:0.85em;color:#556;">
    <span id="qp-stage-label">Stage 1 — classical particles</span>
    <input id="qp-scrub" type="range" min="0" max="1000" value="0" style="width:100%;margin-top:6px;">
  </div>
</div>

<script>
(function(){
  const canvas = document.getElementById('qp-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const playBtn = document.getElementById('qp-play');
  const resetBtn = document.getElementById('qp-reset');
  const scrub = document.getElementById('qp-scrub');
  const label = document.getElementById('qp-stage-label');

  let t = 0;              // 0..1 progress
  let playing = false;
  let raf = null;
  let lastTs = 0;

  const atoms = [
    { x0: W*0.32, y0: H*0.55, color: '#7cc4ff', label: 'A' },
    { x0: W*0.68, y0: H*0.55, color: '#ffb37c', label: 'B' }
  ];

  function lerp(a,b,u){ return a + (b-a)*u; }
  function smooth(u){ return u<0?0:u>1?1:u*u*(3-2*u); }

  function stageInfo(p){
    // p in [0,1]; three stages: 0-0.33 particles, 0.33-0.66 zoom, 0.66-1 wave
    if (p < 0.33) return { name:'Stage 1 — classical particles', s:0, local:p/0.33 };
    if (p < 0.66) return { name:'Stage 2 — zooming in', s:1, local:(p-0.33)/0.33 };
    return { name:'Stage 3 — wave functions', s:2, local:(p-0.66)/0.34 };
  }

  function drawBackground(){
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0,0,W,H);
  }

  function drawAtom(cx, cy, r, color, ringAlpha){
    // electron cloud / ring
    if (ringAlpha > 0){
      ctx.strokeStyle = `rgba(60,70,90,${0.45*ringAlpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.ellipse(cx, cy, r*2.2, r*0.9, 0, 0, Math.PI*2); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(cx, cy, r*2.2, r*0.9, Math.PI/3, 0, Math.PI*2); ctx.stroke();
      // electron dot
      const ang = performance.now()*0.003;
      const ex = cx + Math.cos(ang)*r*2.2;
      const ey = cy + Math.sin(ang)*r*0.9;
      ctx.fillStyle = `rgba(30,50,90,${ringAlpha})`;
      ctx.beginPath(); ctx.arc(ex, ey, 2.5, 0, Math.PI*2); ctx.fill();
    }
    // nucleus
    const grad = ctx.createRadialGradient(cx,cy,1,cx,cy,r);
    grad.addColorStop(0, color);
    grad.addColorStop(0.6, color);
    grad.addColorStop(1, 'rgba(255,255,255,0.0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.fill();
  }

  function drawWavePacket(cx, cy, amp, sigma, k, color, envAlpha, oscAlpha){
    // envAlpha drives the probability cloud / envelope; oscAlpha drives the visible oscillations.
    const span = 220;
    // baseline axis (fades in with envelope)
    if (envAlpha > 0){
      ctx.strokeStyle = `rgba(0,0,0,${0.2*envAlpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx-span, cy); ctx.lineTo(cx+span, cy); ctx.stroke();
    }

    // |psi|^2 cloud (probability density) — appears first
    if (envAlpha > 0){
      ctx.fillStyle = hexToRgba(color, 0.22*envAlpha);
      ctx.beginPath();
      ctx.moveTo(cx-span, cy);
      for (let i=-span; i<=span; i+=2){
        const env = Math.exp(-(i*i)/(2*sigma*sigma));
        const y = cy - amp*env*env;
        ctx.lineTo(cx+i, y);
      }
      ctx.lineTo(cx+span, cy);
      ctx.closePath();
      ctx.fill();
    }

    // Real part oscillation — comes in later, amplitude ramps with oscAlpha
    if (oscAlpha > 0){
      const phase = performance.now()*0.004;
      ctx.lineWidth = 2;
      ctx.strokeStyle = hexToRgba(color, oscAlpha);
      ctx.beginPath();
      for (let i=-span; i<=span; i+=2){
        const env = Math.exp(-(i*i)/(2*sigma*sigma));
        const y = cy - amp*oscAlpha*env*Math.cos(k*i - phase);
        if (i===-span) ctx.moveTo(cx+i, y); else ctx.lineTo(cx+i, y);
      }
      ctx.stroke();
    }
  }

  function hexToRgba(hex, a){
    const h = hex.replace('#','');
    const r = parseInt(h.substring(0,2),16);
    const g = parseInt(h.substring(2,4),16);
    const b = parseInt(h.substring(4,6),16);
    return `rgba(${r},${g},${b},${a})`;
  }

  function drawZoomFrame(alpha){
    ctx.strokeStyle = `rgba(0,0,0,${0.35*alpha})`;
    ctx.setLineDash([6,4]);
    ctx.lineWidth = 1;
    ctx.strokeRect(W*0.15, H*0.25, W*0.7, H*0.55);
    ctx.setLineDash([]);
  }

  function render(){
    drawBackground();
    const info = stageInfo(t);
    label.textContent = info.name;

    // Compute interpolated parameters
    // Stage 0: small atoms at original positions, no rings yet
    // Stage 1: zoom in -> atoms grow, ring appears
    // Stage 2: staggered morph — atom expands & dissolves, |ψ|² cloud emerges first, then oscillations
    let atomR, ringA, atomFade, envA, oscA, zoomA;
    if (info.s === 0){
      atomR = lerp(6, 14, smooth(info.local));
      ringA = lerp(0, 0.4, smooth(info.local));
      atomFade = 1;
      envA = 0; oscA = 0;
      zoomA = lerp(0, 0.6, smooth(info.local));
    } else if (info.s === 1){
      atomR = lerp(14, 48, smooth(info.local));
      ringA = lerp(0.4, 0.9, smooth(info.local));
      atomFade = 1;
      envA = 0; oscA = 0;
      zoomA = lerp(0.6, 1.0, smooth(info.local));
    } else {
      const u = info.local;
      // atom expands a touch further, then softens
      atomR = lerp(48, 70, smooth(u));
      ringA = lerp(0.9, 0.0, smooth(Math.min(1, u*1.4)));
      // atom fades out across roughly the first 65% of the stage
      atomFade = 1 - smooth(Math.min(1, u/0.65));
      // envelope/cloud comes in early (overlaps with atom dissolving)
      envA = smooth(Math.min(1, u/0.55));
      // oscillations come in later, after the cloud is mostly there
      oscA = smooth(Math.max(0, (u-0.45)/0.55));
      zoomA = lerp(1.0, 0.25, smooth(u));
    }

    atoms.forEach((a,i) => {
      // Soft-glow expansion underneath atom while it's dissolving (helps bridge into the cloud)
      if (atomFade > 0 && info.s === 2){
        const glowR = atomR * (1 + 1.2*(1-atomFade));
        const g = ctx.createRadialGradient(a.x0, a.y0, 1, a.x0, a.y0, glowR);
        g.addColorStop(0, hexToRgba(a.color, 0.35*atomFade));
        g.addColorStop(1, hexToRgba(a.color, 0));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(a.x0, a.y0, glowR, 0, Math.PI*2); ctx.fill();
      }

      // Atom representation fades out
      if (atomFade > 0){
        ctx.globalAlpha = atomFade;
        drawAtom(a.x0, a.y0, atomR, a.color, ringA*atomFade);
        ctx.globalAlpha = 1;
      }

      // Wave packet — envelope first, oscillations later
      if (envA > 0 || oscA > 0){
        const amp = 70;
        const sigma = 60;
        const k = 0.12;
        drawWavePacket(a.x0, a.y0, amp, sigma, k, a.color, envA, oscA);
      }

      // Labels (crossfade between "Atom" and "ψ(x)")
      ctx.font = '12px sans-serif';
      const labelY = a.y0 + Math.max(atomR, 60) + 22;
      if (atomFade > 0){
        ctx.fillStyle = `rgba(40,40,40,${0.85*atomFade})`;
        ctx.fillText(`Atom ${a.label}`, a.x0-22, labelY);
      }
      if (envA > 0){
        ctx.fillStyle = `rgba(40,40,40,${0.85*envA})`;
        ctx.fillText(`ψ_${a.label}(x)`, a.x0-22, labelY + (atomFade > 0 ? 14 : 0));
      }
    });

    // Header text
    ctx.fillStyle = 'rgba(60,70,90,0.9)';
    ctx.font = '13px sans-serif';
    ctx.fillText('Two atoms → zoom in → quantum wave functions', 16, 24);

    scrub.value = Math.round(t*1000);
  }

  function tick(ts){
    if (!lastTs) lastTs = ts;
    const dt = (ts - lastTs)/1000;
    lastTs = ts;
    if (playing){
      t += dt * 0.18; // ~5.5s full sweep
      if (t >= 1){ t = 1; playing = false; playBtn.textContent = 'Play'; }
    }
    render();
    raf = requestAnimationFrame(tick);
  }

  playBtn.addEventListener('click', () => {
    if (t >= 1) t = 0;
    playing = !playing;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    lastTs = 0;
  });
  resetBtn.addEventListener('click', () => {
    t = 0; playing = false; playBtn.textContent = 'Play';
  });
  scrub.addEventListener('input', () => {
    t = parseInt(scrub.value,10)/1000;
    playing = false; playBtn.textContent = 'Play';
  });

  raf = requestAnimationFrame(tick);
})();
</script>


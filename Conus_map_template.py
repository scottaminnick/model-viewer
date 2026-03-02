CONUS_MAP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Model Viewer — RAP13 CONUS</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --ac:#f0883e; }
  *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
         height:100dvh; display:flex; flex-direction:column; overflow:hidden; }
  #header { background:var(--panel); border-bottom:1px solid var(--border);
    padding:0.45rem 0.75rem; display:flex; align-items:center;
    gap:0.75rem; flex-wrap:wrap; flex-shrink:0; }
  #header .title { font-weight:700; font-size:0.95rem; white-space:nowrap; }
  #header .subtitle { font-size:0.72rem; color:var(--muted); }
  .nav-link { font-size:0.75rem; color:var(--ac); text-decoration:none;
              padding:0.25rem 0.5rem; border:1px solid var(--ac);
              border-radius:4px; white-space:nowrap; }
  .nav-link:hover { background:rgba(240,136,62,0.1); }
  select { background:var(--bg); color:var(--text); border:1px solid var(--border);
           border-radius:5px; font-size:0.78rem; padding:0.28rem 0.5rem; cursor:pointer; }
  input[type=range] { background:var(--bg); }
  .ctrl-group { display:flex; align-items:center; gap:0.4rem; }
  .ctrl-label { font-size:0.68rem; color:var(--muted); white-space:nowrap; }
  #hour-bar { display:flex; align-items:center; gap:0.3rem; padding:0.3rem 0.75rem;
    background:var(--panel); border-bottom:1px solid var(--border);
    flex-shrink:0; flex-wrap:wrap; }
  .hbtn { font-size:0.68rem; font-weight:600; padding:0.2rem 0.45rem;
    border-radius:4px; border:1px solid var(--border); background:var(--bg);
    color:var(--muted); cursor:pointer; transition:background 0.15s,color 0.15s;
    position:relative; }
  .hbtn.available { color:var(--text); border-color:#444; }
  .hbtn.active { background:var(--ac); color:#000; border-color:var(--ac); }
  .hbtn.unavail { opacity:0.35; cursor:not-allowed; }
  .dot-badge { position:absolute; top:-3px; right:-3px;
               width:6px; height:6px; border-radius:50%; }
  .dot-green { background:#2ecc71; } .dot-grey { background:#555; }
  #progress-bar { height:3px; background:var(--border); flex:1;
                  border-radius:2px; min-width:60px; }
  #progress-fill { height:100%; background:var(--ac); border-radius:2px;
                   transition:width 0.4s; width:0%; }
  #main { flex:1; display:flex; min-height:0; }
  #map  { flex:1; position:relative; }
  #sidebar { width:220px; background:var(--panel); border-left:1px solid var(--border);
             display:flex; flex-direction:column; flex-shrink:0; overflow-y:auto; }
  #legend { padding:0.75rem; }
  .leg-title { font-size:0.72rem; font-weight:700; color:var(--muted);
               margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.05em; }
  .leg-row { display:flex; align-items:center; gap:0.55rem; margin:0.22rem 0;
             font-size:0.72rem; }
  .leg-swatch { width:22px; height:12px; border-radius:2px; flex-shrink:0; }
  #cursor-box { padding:0.6rem 0.75rem; border-top:1px solid var(--border);
                font-size:0.72rem; min-height:80px; }
  .cursor-title { font-size:0.65rem; color:var(--muted); text-transform:uppercase;
                  letter-spacing:0.05em; margin-bottom:0.4rem; }
  #cursor-val { font-size:1.4rem; font-weight:700; color:var(--ac); }
  #cursor-pos { font-size:0.65rem; color:var(--muted); margin-top:0.2rem; }
  #meta { padding:0.5rem 0.75rem; font-size:0.68rem; color:var(--muted);
          border-top:1px solid var(--border); }
  #meta b { color:var(--text); }
  #loading-overlay { position:absolute; inset:0; z-index:2000;
    background:rgba(13,17,23,0.88); display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:1rem; transition:opacity 0.3s; }
  #loading-overlay.hidden { opacity:0; pointer-events:none; }
  .spinner { width:42px; height:42px; border:3px solid var(--border);
             border-top-color:var(--ac); border-radius:50%;
             animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #load-msg { font-size:0.8rem; color:var(--muted); text-align:center; max-width:260px; }
  #error-bar { display:none; background:#5a1a1a; color:#f9a8a8;
               padding:0.4rem 0.75rem; font-size:0.78rem;
               border-bottom:1px solid #8b2020; flex-shrink:0; }
  .leaflet-control-layers { background:var(--panel)!important;
    border:1px solid var(--border)!important; color:var(--text)!important; font-size:0.75rem; }
  .leaflet-control-layers label { color:var(--text)!important; }
  /* Cursor crosshair tooltip */
  #cursor-tooltip { position:absolute; pointer-events:none; z-index:1500;
    background:rgba(13,17,23,0.90); border:1px solid var(--border);
    border-radius:5px; padding:0.3rem 0.55rem; font-size:0.72rem;
    white-space:nowrap; display:none; }
</style>
</head>
<body>

<div id="header">
  <span class="title">🌎 RAP13 CONUS</span>
  <span class="subtitle">Wind Gusts — Surface</span>
  <div class="ctrl-group">
    <span class="ctrl-label">CYCLE</span>
    <select id="cycle-sel" onchange="onCycleChange()"><option value="">—</option></select>
  </div>
  <div class="ctrl-group" style="margin-left:auto;">
    <span class="ctrl-label">OPACITY</span>
    <input type="range" id="opacity-slider" min="10" max="100" step="5" value="70"
      style="width:80px;" oninput="updateOpacity(this.value)"/>
    <span id="opacity-val" style="font-size:0.72rem;color:var(--muted);width:28px;">70%</span>
  </div>
  <a class="nav-link" href="/map/hrrr">← HRRR Colorado</a>
</div>

<div id="error-bar"></div>

<div id="hour-bar">
  <span class="ctrl-label">HOUR →</span>
  <div id="progress-bar"><div id="progress-fill"></div></div>
  <span id="cycle-pct" style="font-size:0.68rem;color:var(--muted);white-space:nowrap;"></span>
</div>

<div id="main">
  <div id="map">
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div id="load-msg">Loading RAP13…<br>
        <small style="color:var(--muted)">~30 s first load</small>
      </div>
    </div>
    <div id="cursor-tooltip"></div>
  </div>

  <div id="sidebar">
    <div id="legend">
      <div class="leg-title">Wind Gust (kt)</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#4575b4"></div>&lt; 5 kt — Calm</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#74add1"></div>5–10 kt — Light</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#abd9e9"></div>10–15 kt — Breezy</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#e0f3f8"></div>15–20 kt — Moderate</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#fee090"></div>20–25 kt — Fresh</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#fc8d59"></div>25–35 kt — Strong</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#d73027"></div>35–50 kt — Very Strong</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#a50026"></div>&ge; 50 kt — Extreme</div>
      <div style="margin-top:0.7rem;font-size:0.62rem;color:var(--muted);">
        RAP13 13km grid, stride=2<br>Move cursor over map for value
      </div>
    </div>

    <div id="cursor-box">
      <div class="cursor-title">Cursor Sample</div>
      <div id="cursor-val">—</div>
      <div id="cursor-pos"></div>
    </div>

    <div id="meta">
      <div>Model: <b>RAP13</b></div>
      <div>Valid: <b id="meta-valid">—</b></div>
      <div>Points: <b id="meta-pts">—</b></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// ── Color scale (8-band, diverging blue→red) ─────────────────────────────────
var BANDS = [
  { max:  5, color:'#4575b4' },
  { max: 10, color:'#74add1' },
  { max: 15, color:'#abd9e9' },
  { max: 20, color:'#e0f3f8' },
  { max: 25, color:'#fee090' },
  { max: 35, color:'#fc8d59' },
  { max: 50, color:'#d73027' },
  { max: Infinity, color:'#a50026' },
];
function gustColor(kt){
  for(var i=0;i<BANDS.length;i++) if(kt<BANDS[i].max) return BANDS[i].color;
  return '#a50026';
}

// ── State vars ────────────────────────────────────────────────────────────────
var currentCycle=null, currentFxx=1, currentOpacity=0.70;
var cycleStatus={}, dataLayer=null, pointsFlat=[];

// ── Map setup ─────────────────────────────────────────────────────────────────
var map=L.map('map',{center:[39.5,-98.0],zoom:4,preferCanvas:true});

L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Tiles &copy; Esri',maxZoom:10}
).addTo(map);

// ── Boundary layers ───────────────────────────────────────────────────────────
var stateStyle={ color:'#8b949e', weight:1.0, fill:false, opacity:0.7 };
var artccStyle={ color:'#f0883e', weight:1.5, fill:false, opacity:0.85,
                 dashArray:'6 4' };

var statesLayer=null, artccLayer=null;

// US States GeoJSON
fetch('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
  .then(function(r){return r.json();})
  .then(function(gj){
    statesLayer=L.geoJSON(gj,{style:stateStyle});
    layerControl.addOverlay(statesLayer,'⬜ States');
    statesLayer.addTo(map);   // on by default
  }).catch(function(e){console.warn('States GeoJSON failed',e);});

// ARTCC boundaries — FAA open data via ArcGIS
fetch('https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/Air_Route_Traffic_Control_Centers/FeatureServer/0/query?outFields=NAME,IDENT&where=1%3D1&f=geojson')
  .then(function(r){return r.json();})
  .then(function(gj){
    artccLayer=L.geoJSON(gj,{
      style:artccStyle,
      onEachFeature:function(feat,layer){
        var name=(feat.properties&&(feat.properties.NAME||feat.properties.IDENT))||'ARTCC';
        layer.bindTooltip(name,{sticky:true,className:'artcc-tip',
          direction:'center',opacity:0.9});
      }
    });
    layerControl.addOverlay(artccLayer,'🔶 ARTCCs');
    artccLayer.addTo(map);    // on by default
  }).catch(function(e){console.warn('ARTCC GeoJSON failed',e);});

// Layer control (populated after GeoJSON loads)
var layerControl=L.control.layers(null,{},
  {collapsed:false,position:'topright'}).addTo(map);

// Roads overlay
var roadsLayer=L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
  {attribution:'',maxZoom:10,opacity:0.35});
layerControl.addOverlay(roadsLayer,'≡ Roads');

// ── Cursor sampling ───────────────────────────────────────────────────────────
// Build simple grid index: bucket points by ~1° cell for fast lookup
var _idx={}, _cellSz=1.0;
function _idxKey(lat,lon){
  return Math.floor(lat/_cellSz)+'|'+Math.floor(lon/_cellSz);
}
function buildIndex(pts){
  _idx={};
  for(var i=0;i<pts.length;i++){
    var p=pts[i], k=_idxKey(p.lat,p.lon);
    if(!_idx[k]) _idx[k]=[];
    _idx[k].push(i);
  }
}
function nearestPoint(lat,lon){
  if(!pointsFlat.length) return null;
  var best=null, bestD=1e9;
  // Check 3x3 neighbourhood of cells
  for(var dy=-1;dy<=1;dy++){
    for(var dx=-1;dx<=1;dx++){
      var k=(Math.floor(lat/_cellSz)+dy)+'|'+(Math.floor(lon/_cellSz)+dx);
      var cell=_idx[k];
      if(!cell) continue;
      for(var i=0;i<cell.length;i++){
        var p=pointsFlat[cell[i]];
        var d=(p.lat-lat)*(p.lat-lat)+(p.lon-lon)*(p.lon-lon);
        if(d<bestD){bestD=d;best=p;}
      }
    }
  }
  return bestD<4 ? best : null;   // only return if within ~2° (~200km)
}

var tooltip=document.getElementById('cursor-tooltip');
map.on('mousemove',function(e){
  var p=nearestPoint(e.latlng.lat,e.latlng.lng);
  if(p){
    var color=gustColor(p.gust_kt);
    // Sidebar panel
    document.getElementById('cursor-val').textContent=p.gust_kt.toFixed(0)+' kt';
    document.getElementById('cursor-val').style.color=color;
    document.getElementById('cursor-pos').textContent=
      p.lat.toFixed(2)+'°N  '+Math.abs(p.lon).toFixed(2)+'°W';
    // Floating tooltip near cursor
    tooltip.style.display='block';
    var cp=map.latLngToContainerPoint(e.latlng);
    tooltip.style.left=(cp.x+18)+'px';
    tooltip.style.top=(cp.y-12)+'px';
    tooltip.innerHTML='<span style="color:'+color+';font-weight:700;">'+
      p.gust_kt.toFixed(0)+' kt</span>';
  } else {
    tooltip.style.display='none';
    document.getElementById('cursor-val').textContent='—';
    document.getElementById('cursor-val').style.color='var(--ac)';
    document.getElementById('cursor-pos').textContent='';
  }
});
map.on('mouseout',function(){
  tooltip.style.display='none';
  document.getElementById('cursor-val').textContent='—';
  document.getElementById('cursor-pos').textContent='';
});

// ── Opacity ───────────────────────────────────────────────────────────────────
function updateOpacity(val){
  currentOpacity=val/100;
  document.getElementById('opacity-val').textContent=val+'%';
  if(dataLayer) dataLayer.eachLayer(function(l){
    l.setStyle({fillOpacity:currentOpacity});
  });
}

// ── Cycle status ──────────────────────────────────────────────────────────────
async function fetchStatus(){
  try{
    var s=await(await fetch('/api/rap/status')).json();
    cycleStatus={};
    (s.cycles||[]).forEach(function(c){cycleStatus[c.cycle_utc]=c;});
    var sel=document.getElementById('cycle-sel'), prev=sel.value;
    sel.innerHTML='';
    Object.keys(cycleStatus).sort().reverse().forEach(function(c){
      var opt=document.createElement('option'); opt.value=c;
      opt.textContent='RAP '+new Date(c).toUTCString().slice(5,22)+'Z';
      sel.appendChild(opt);
    });
    if(prev&&cycleStatus[prev]) sel.value=prev;
    else if(!currentCycle&&sel.options.length){
      sel.value=sel.options[0].value; currentCycle=sel.value;
    }
    buildHourButtons();
    var cs=cycleStatus[currentCycle];
    if(cs){
      document.getElementById('progress-fill').style.width=cs.pct_complete+'%';
      document.getElementById('cycle-pct').textContent=cs.pct_complete+'% ready';
    }
  }catch(e){console.warn('status fetch failed',e);}
}

function onCycleChange(){
  currentCycle=document.getElementById('cycle-sel').value;
  buildHourButtons(); loadData();
}

function buildHourButtons(){
  document.querySelectorAll('.hbtn').forEach(function(b){b.remove();});
  var cs=cycleStatus[currentCycle], avail=cs?cs.available_hours:[];
  var cached=cs?(cs.cached_hours.gusts||[]):[];
  var bar=document.getElementById('hour-bar'), prog=document.getElementById('progress-bar');
  for(var fxx=1;fxx<=18;fxx++){(function(f){
    var btn=document.createElement('button');
    btn.className='hbtn';
    btn.textContent='F'+String(f).padStart(2,'0'); btn.dataset.fxx=f;
    var dot=document.createElement('span'); dot.className='dot-badge';
    dot.classList.add(cached.includes(f)?'dot-green':'dot-grey');
    btn.appendChild(dot);
    if(avail.includes(f)){
      btn.classList.add('available');
      btn.onclick=function(){selectHour(f);};
    } else { btn.classList.add('unavail'); btn.disabled=true; }
    if(f===currentFxx) btn.classList.add('active');
    bar.insertBefore(btn,prog);
  })(fxx);}
}

function selectHour(fxx){
  currentFxx=fxx;
  document.querySelectorAll('.hbtn').forEach(function(b){
    b.classList.toggle('active',parseInt(b.dataset.fxx)===fxx);
  });
  loadData();
}

// ── Data load + render ────────────────────────────────────────────────────────
async function loadData(){
  if(!currentCycle) return;
  document.getElementById('loading-overlay').classList.remove('hidden');
  document.getElementById('error-bar').style.display='none';
  if(dataLayer){ map.removeLayer(dataLayer); dataLayer=null; pointsFlat=[]; }

  try{
    var url='/api/rap/conus?fxx='+currentFxx+'&cycle_utc='+encodeURIComponent(currentCycle);
    var resp=await fetch(url);
    if(!resp.ok) throw new Error((await resp.text()).slice(0,300));
    var data=await resp.json();

    pointsFlat=data.points;
    buildIndex(pointsFlat);

    // Cell size: overlap by 15% to eliminate projection gaps
    var cell=data.cell_size_deg||0.234;
    var halfLat=cell*0.58, halfLon=cell*0.72;

    var renderer=L.canvas({padding:0.5}), rects=[];
    data.points.forEach(function(p){
      var color=gustColor(p.gust_kt);
      var rect=L.rectangle(
        [[p.lat-halfLat, p.lon-halfLon],[p.lat+halfLat, p.lon+halfLon]],
        {renderer:renderer, color:'none', fillColor:color,
         fillOpacity:currentOpacity, weight:0, stroke:false}
      );
      rects.push(rect);
    });

    dataLayer=L.layerGroup(rects).addTo(map);

    // Bring boundary layers to top so they're visible over the data
    if(statesLayer && map.hasLayer(statesLayer)) statesLayer.bringToFront();
    if(artccLayer  && map.hasLayer(artccLayer))  artccLayer.bringToFront();

    document.getElementById('meta-valid').textContent=data.valid_utc||'—';
    document.getElementById('meta-pts').textContent=
      (data.point_count||rects.length).toLocaleString();
  }catch(e){
    var eb=document.getElementById('error-bar');
    eb.textContent=e.message; eb.style.display='block';
  }finally{
    document.getElementById('loading-overlay').classList.add('hidden');
  }
}

fetchStatus().then(function(){if(currentCycle) loadData();});
setInterval(fetchStatus,300000);
</script>
</body>
</html>"""

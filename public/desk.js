"use strict";
const data = window.deskSnapshot;
const byId = id => document.getElementById(id);
const money = n => Number.isFinite(n) ? '$' + n.toFixed(2) : '—';
const pct = n => Number.isFinite(n) ? n.toFixed(2) + '%' : '—';
const signed = (n, unit = '%') => Number.isFinite(n) ? (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(2) + unit : '—';
const tone = n => n >= 0 ? 'up' : 'down';
const safe = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names = {SPY:'S&P 500',QQQ:'Nasdaq 100',IWM:'Russell 2000',NVDA:'NVIDIA',DIA:'Dow Jones',RSP:'S&P 500 Equal Weight',GLD:'Gold',TLT:'Long Treasuries',SMH:'Semiconductors'};
const titles = {overview:'Market Overview',charts:'Single Ticker',gamma:'Gamma Exposure',volatility:'Volatility Pricing',bias:'Bias Matrix',sector:'Index & Sector Flow',mag7:'Magnificent 7 Flow',ratios:'Synthetic Ratio Rotations'};
const state = {symbol:'SPY', range:63, priceMode:'Candles', ema:true, vol:30, gammaMode:'Net GEX', gammaRange:'Relevant', gammaTable:'Relevant only', gammaDate:''};
const W=800,H=301,L=10,R=730,T=19,B=272;
const priceRows = () => data.prices[state.symbol];
const gammaRows = () => data.gamma[state.symbol] || [];
const currentGamma = new Map();
const latest = () => gammaRows().at(-1);
const activeGamma = () => state.gammaDate==='live' ? currentGamma.get(state.symbol)||latest() : gammaRows().find(row => row.date === state.gammaDate) || latest();
const daily = rows => (rows.at(-1)[4] / rows.at(-2)[4] - 1) * 100;
const dates = new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',timeZone:'UTC'});
const short = date => dates.format(new Date(date));
const route = () => Object.hasOwn(titles, location.hash.slice(1)) ? location.hash.slice(1) : 'overview';

function artwork() {
  let paths = '';
  for (let row = 0; row < 27; row++) {
    let line = '';
    for (let x = 0; x <= 840; x += 12) {
      const height = 38 * Math.sin(x / 112 + row * .082) + 16 * Math.cos(x / 57 - row * .094);
      const taper = 1 + .45 * Math.sin(x / 173);
      const y = 69 + row * 5.5 + height * taper + .12 * x;
      line += (x ? 'L' : 'M') + x + ',' + y.toFixed(1);
    }
    paths += '<path d="' + line + '" stroke="url(#wire)" opacity="' + (.23 + row * .024) + '"/>';
  }
  return '<svg class="hero-art" aria-hidden="true" viewBox="0 0 800 290" preserveAspectRatio="xMidYMid slice"><defs><linearGradient id="wire"><stop stop-color="var(--wire-start)" stop-opacity="0"/><stop offset=".52" stop-color="var(--wire-mid)"/><stop offset="1" stop-color="var(--wire-end)"/></linearGradient></defs><g fill="none">' + paths + '</g></svg>';
}
function hero(label, title, subtitle, date, detail = false) {
  return '<div class="' + (detail ? 'detail-head ' : '') + 'in"><section class="hero"><div class="hero-light" aria-hidden="true"></div>' + artwork() + '<div><p class="eyebrow">' + label + '</p><h1>' + title + '</h1><p>' + subtitle + '</p></div><div class="hero-meta"><strong>' + safe(date) + '</strong><span>SESSION DATE</span></div></section></div>';
}
function control(key, choices, labels = choices) {
  return choices.map((v,i) => '<button data-key="' + key + '" data-value="' + v + '" aria-pressed="' + (state[key] === v) + '">' + labels[i] + '</button>').join('');
}
function stat(label,value,cls='') {
  return '<div class="stat"><span>' + label + '</span><strong class="' + cls + (/^[+−$\d]/.test(value) ? ' numeric' : '') + '">' + value + '</strong></div>';
}
function card(kicker,title,body,controls='', classes='') {
  return '<section class="card ' + classes + '"><header class="card-head"><div><p class="eyebrow">' + kicker + '</p><h2>' + title + '</h2></div><div class="controls">' + controls + '</div></header>' + body + '</section>';
}
function spark(rows, ch) {
  const values = rows.map(r => r[4]), min = Math.min(...values), max = Math.max(...values);
  const points = values.map((v,i) => [i*72/(values.length-1||1),24-(v-min)/(max-min||1)*21].join(',')).join(' ');
  return '<svg aria-hidden="true" viewBox="0 0 72 26"><polyline points="' + points + '" fill="none" stroke="' + (ch>=0?'var(--green)':'var(--red)') + '" stroke-width="1.25" opacity=".85"/></svg>';
}
const coreAssets = [
  ['SPY','S&P 500','Large-cap core benchmark'],
  ['QQQ','Nasdaq 100','Tech growth benchmark'],
  ['IWM','Russell 2000','Small-cap value benchmark'],
  ['GLD','Gold Trust','Safe haven and debasement hedge'],
  ['TLT','20Y Treasury','Duration and real yields'],
  ['IBIT','Bitcoin Trust','High-beta liquidity'],
  ['USO','Crude Oil','Energy and inflation input costs']
];
const megaCaps = [
  ['NVDA','Nvidia','Accelerated computing and AI hardware'],
  ['AAPL','Apple','Consumer ecosystem and devices'],
  ['MSFT','Microsoft','Enterprise cloud and AI'],
  ['AMZN','Amazon','E-commerce and AWS cloud'],
  ['GOOGL','Alphabet','Search, YouTube and Gemini'],
  ['META','Meta Platforms','Social media and AI'],
  ['TSLA','Tesla','Autonomy and electric mobility']
];
const breadthPairs = [
  ['SPY/RSP','Market Breadth','Cap-weighted vs. equal-weight'],
  ['QQQ/RSP','Tech Breadth','Tech beta vs. broad market'],
  ['IWM/QQQ','Small vs. Tech','Small-cap vs. mega-cap tech']
];
function overviewSection(title, subtitle, markup, type) {
  return '<section class="overview-section in"><header class="overview-heading"><h2>'+title+'</h2><span>'+subtitle+'</span></header><div class="'+type+'">'+markup+'</div></section>';
}
function overviewQuote([ticker,name,description], group) {
  const m = data.overview.assets[ticker];
  if (!m) return '<article class="card"><h3>'+ticker+'</h3><p>Snapshot unavailable</p></article>';
  const metric = (label,value) => '<span><span>'+label+'</span><b class="overview-num '+tone(value)+'">'+signed(value)+'</b></span>';
  const footer = (group!=='macro'?metric('ATH DD',m.drawdown):'')+metric('5D',m.week)+(group!=='index'?metric('20D',m.month):'');
  const id='quote-'+ticker;
  return '<article class="overview-flip"><div class="overview-flip-inner">'
    +'<button class="card overview-quote overview-front" data-flip="open" aria-controls="'+id+'" aria-expanded="false" aria-label="'+ticker+' '+money(m.close)+', '+signed(m.day)+' today. Show details">'
    +'<span class="overview-quote-head"><b>'+ticker+'</b><span class="overview-num '+tone(m.day)+'">'+signed(m.day)+'</span></span>'
    +'<span class="overview-name">'+name+'</span><span class="overview-price"><strong class="overview-num">'+money(m.close)+'</strong>'+spark(remoteSparks[ticker]?.map(close=>[null,null,null,null,close])||data.prices[ticker].slice(-6),m.week)+'</span></button>'
    +'<div class="card overview-quote overview-back" id="'+id+'" inert><div class="overview-quote-head"><b>'+ticker+'</b><button class="flip-close" data-flip="close" aria-label="Close '+ticker+' details">↶</button></div>'
    +'<div class="overview-metrics">'+footer+'</div>'
    +(group==='mega'?'<div class="overview-signal">'+safe(m.status)+'</div>':'<div class="overview-description">'+description+'</div>')
    +'<button class="flip-chart" data-chart-symbol="'+ticker+'">Price chart ↗</button></div></div></article>';
}
function breadthCard([pair,title,subtitle]) {
  const m = data.overview.ratios[pair];
  if (!m) return card(pair,title,'<p>Ratio snapshot unavailable</p>');
  const meter = Number.isFinite(m.z) ? '<div class="breadth-scale"><div class="breadth-track"><span style="--position:'+Math.max(1,Math.min(99,(m.z+3)/6*100))+'%"></span></div><div><span>−3σ</span><span>0σ</span><span>+3σ</span></div></div>' : '';
  return card(pair,title,'<p>'+subtitle+'</p><div class="breadth-value"><strong class="overview-num">'+m.close.toFixed(2)+'</strong><span class="overview-num amber">'+signed(m.z,'σ')+'</span></div>'
    +meter+'<div class="breadth-status"><strong>'+safe(m.state)+'</strong><span class="overview-num '+tone(m.week)+'">5D '+signed(m.week)+'</span></div><p>'+safe(m.action)+'</p>');
}
function marketOverview() {
  return overviewSection('Core Equities & Macro Benchmarks','Session closes and momentum',coreAssets.map((entry,i)=>overviewQuote(entry,i<3?'index':'macro')).join(''),'overview-quotes')
    +overviewSection('The Magnificent 7','Momentum and technical signals',megaCaps.map(entry=>overviewQuote(entry,'mega')).join(''),'overview-quotes')
    +overviewSection('Quantitative Regime & Breadth Ratios','50-day z-score and relative momentum',breadthPairs.map(breadthCard).join(''),'overview-breadth');
}
function plotFrame(label, grid, marks, xLabels, hover = false, stretch = false) {
  return '<div class="chart"><svg class="draw" ' + (hover ? 'tabindex="0" ' : '') + 'role="img" aria-label="' + label + '" viewBox="0 0 '+W+' '+H+'"' + (stretch ? ' preserveAspectRatio="none"' : '') + '>' + grid + marks + xLabels + (hover ? '<line id="cross" class="hover-line" y1="'+T+'" y2="'+B+'" visibility="hidden"/><circle id="cross-dot" r="4" fill="var(--amber)" visibility="hidden"/>' : '') + '</svg><div class="tooltip" hidden></div></div>';
}
function scale(values) {
  const lo=Math.min(...values), hi=Math.max(...values), pad=(hi-lo)*.08||1;
  const min=lo-pad,max=hi+pad;
  return {y:v=>B-(v-min)/(max-min)*(B-T), ticks:Array.from({length:5},(_,i)=>max-i*(max-min)/4)};
}
function grid(y,ticks,format,showLines=true) {
  return ticks.map(v=>(showLines?'<line x1="'+L+'" x2="'+R+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="var(--grid-line)" stroke-dasharray="2 4"/>':'')+'<text x="'+(R+8)+'" y="'+(y(v)+4)+'">'+format(v)+'</text>').join('');
}
function axis(rows,x,format) {
  const indices=[0,Math.floor((rows.length-1)/3),Math.floor((rows.length-1)*2/3),rows.length-1];
  return indices.map((i,j)=>'<text x="'+x(i)+'" y="292" text-anchor="'+(j===0?'start':j===3?'end':'middle')+'">'+format(rows[i])+'</text>').join('');
}
function line(values,x,y) {
  return values.map((v,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(v).toFixed(2)).join('');
}
let hoverSeries;
function priceChart() {
  const all=priceRows(), r=all.slice(-state.range), last=r.at(-1);
  const ema=all.reduce((out,row,i)=>{out.push(i ? out[i-1]+2/21*(row[4]-out[i-1]) : row[4]);return out;},[]).slice(-r.length);
  const closes=r.map(row=>row[4]), values=state.priceMode==='Candles' ? r.flatMap(row=>[row[2],row[3]]) : closes;
  const {y,ticks}=scale(state.ema ? values.concat(ema) : values);
  const step=(R-L)/r.length, x=i=>L+step*(i+.5), bw=Math.max(1,step*.61);
  const maxVolume=Math.max(...r.map(row=>row[5]),1);
  let marks=r.map((row,i)=>'<rect x="'+(x(i)-bw/2)+'" y="'+(B-row[5]/maxVolume*31)+'" width="'+bw+'" height="'+(row[5]/maxVolume*31)+'" fill="'+(row[4]>=row[1]?'var(--green)':'var(--red)')+'" opacity=".21"/>').join('');
  if(state.priceMode==='Candles'){
    marks+=r.map((row,i)=>{
      const [,o,h,l,c]=row, color=c>=o?'var(--green)':'var(--red)';
      return '<line x1="'+x(i)+'" x2="'+x(i)+'" y1="'+y(h)+'" y2="'+y(l)+'" stroke="'+color+'"/><rect x="'+(x(i)-bw/2)+'" y="'+y(Math.max(o,c))+'" width="'+bw+'" height="'+Math.max(1,Math.abs(y(o)-y(c)))+'" rx=".5" fill="'+color+'"/>';
    }).join('');
  } else {
    const curve=line(closes,x,y);
    marks+='<defs><linearGradient id="area" x1="0%" y1="0%" x2="0%" y2="100%" gradientUnits="objectBoundingBox"><stop stop-color="var(--amber)" stop-opacity=".24"/><stop offset="1" stop-color="var(--amber)" stop-opacity="0"/></linearGradient></defs><path d="'+curve+'L'+x(r.length-1)+','+B+'L'+x(0)+','+B+'Z" fill="url(#area)"/><path d="'+curve+'" fill="none" stroke="var(--amber)" stroke-width="2"/>';
  }
  if(state.ema) marks+='<path d="'+line(ema,x,y)+'" fill="none" stroke="var(--amber)" opacity=".8" stroke-width="1.35"/>';
  hoverSeries={size:r.length,x,y:i=>y(r[i][4]),text:i=>{const [date,o,h,l,c,v]=r[i];return date+' · '+money(c)+'<br>O '+money(o)+'　H '+money(h)+'　L '+money(l)+'<br>Volume '+v.toLocaleString('en-US');}};
  return card('Daily OHLC',''+state.symbol+' price action',
    plotFrame(state.symbol+' price history. Use left and right arrow keys to inspect sessions.',grid(y,ticks,money),marks,axis(r,x,row=>short(row[0])),true,true)
    +'<div class="chart-note"><span>'+money(last[4])+' close · '+r.length+' sessions</span><span>'+(state.ema?'● EMA 20 · ':'')+'Volume</span></div>',
    control('range',[21,63,126,252],['1M','3M','6M','1Y'])+control('priceMode',['Candles','Line'])+control('ema',[true],['EMA 20']),'in');
}
function returns(rows) {
  return rows.slice(1).map((row,i)=>row[4]/rows[i][4]-1);
}
function hv(r) {
  if(r.length<2) return NaN;
  const avg=r.reduce((a,b)=>a+b,0)/r.length;
  return Math.sqrt(r.reduce((sum,v)=>sum+(v-avg)**2,0)/(r.length-1)*252)*100;
}
function cleanHv(r) {
  const median=values=>{const sorted=[...values].sort((a,b)=>a-b), mid=Math.floor(sorted.length/2);return sorted.length%2?sorted[mid]:(sorted[mid-1]+sorted[mid])/2;};
  const center=median(r), sigma=1.4826*median(r.map(v=>Math.abs(v-center))) || hv(r)/(Math.sqrt(252)*100);
  return hv(r.map(v=>Math.min(center+3*sigma,Math.max(center-3*sigma,v))));
}
function volChart() {
  const rows=priceRows(), changes=returns(rows), n=state.vol;
  const history=rows.slice(n).map((row,i)=>[row[0],hv(changes.slice(i,i+n))]).slice(-126);
  const g=latest(), iv=g?.current_iv;
  const matching=Number.isFinite(iv)&&g.date===history.at(-1)[0];
  const values=history.map(row=>row[1]),{y,ticks}=scale(matching ? values.concat(iv) : values);
  const x=i=>L+(R-L)*i/(history.length-1);
  let marks='<path d="'+line(values,x,y)+'" fill="none" stroke="var(--text)" stroke-width="2"/>';
  if(matching) marks+='<line x1="'+L+'" x2="'+R+'" y1="'+y(iv)+'" y2="'+y(iv)+'" stroke="var(--iv)" stroke-dasharray="5 5"/>';
  hoverSeries={size:history.length,x,y:i=>y(values[i]),text:i=>history[i][0]+' · '+n+'D HV '+pct(values[i])};
  return card('Annualized volatility',state.symbol+' realized volatility',
    plotFrame('Realized volatility history. Use left and right arrow keys to inspect sessions.',grid(y,ticks,pct),marks,axis(history,x,row=>short(row[0])),true)
    +'<div class="chart-note"><span class="vol-primary">● '+n+'D realized volatility</span><span>'+(matching ? '┄ Same-session IV reference' : '126 sessions')+'</span></div>',
    control('vol',[10,20,30,60,252],['10D','20D','30D','60D','1Y']),'in');
}
const gammaDistance = (strike, spot) => spot > 0 && Number.isFinite(strike) ? (strike / spot - 1) * 100 : NaN;
const gammaMoney = (amount, unit) => Number.isFinite(amount) ? (amount < 0 ? '−' : '+') + '$' + Math.abs(amount).toFixed(unit === 'B' ? 2 : 1) + unit : '—';
const contracts = value => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
const gammaLevels = g => [
  ['Spot',g.spot_price,'var(--text)'],
  ['Call wall',g.call_wall,'var(--green)'],
  ['Put wall',g.put_wall,'var(--red)'],
  ['Zero gamma',g.zero_gamma_level,'var(--amber)']
];
function savedStrikes(g) {
  return Object.entries(g?.strikes || {}).map(([key, row]) => ({...row, strike:Number(key)}))
    .filter(row => Number.isFinite(row.strike)).sort((a,b)=>a.strike-b.strike);
}
function gammaFilters(g) {
  const symbols = [...new Set([state.symbol,...Object.keys(data.gamma).filter(key=>data.gamma[key].length).sort()])];
  const tickerOptions = symbols.map(t=>'<option '+(t===state.symbol?'selected ':'')+'value="'+t+'">'+t+'</option>').join('');
  const latestOption=currentGamma.has(state.symbol)?'<option value="live" '+(state.gammaDate==='live'?'selected':'')+'>Latest calculation</option>':'';
  const sessions = latestOption+[...gammaRows()].reverse().map(row=>'<option '+(row.date===state.gammaDate||(!state.gammaDate&&row.date===g?.date)?'selected ':'')+'value="'+safe(row.date)+'">'+safe(row.date)+'</option>').join('');
  return '<div class="gamma-filters controls"><label>Underlying <select id="gamma-symbol">'+tickerOptions+'</select></label><label>Session <select id="session" '+(sessions?'':'disabled')+'>'+(sessions||'<option>Unavailable</option>')+'</select></label><button id="gamma-recalculate">Recalculate GEX</button></div>';
}
function gammaSummary(g) {
  const history = gammaRows();
  const prev = history.filter(row=>row.date<g.date).at(-1);
  const keys = ['spot_price','call_wall','put_wall','zero_gamma_level'];
  const levels = gammaLevels(g).map(([label,value,color],i)=>{
    const distance = i ? signed(gammaDistance(value,g.spot_price)) : '';
    return '<div class="gamma-kpi"><small>'+label+'</small><strong>'+money(value)+'</strong>'
      +(distance?'<span class="gamma-distance" style="color:'+color+'">'+distance+' vs. spot</span>':'')
      +(prev?'<small>Prev. '+money(prev[keys[i]])+'</small>':'')+'</div>';
  });
  const net = g.total_net_gex_billions;
  levels.push('<div class="gamma-kpi"><small>Net dealer GEX</small><strong class="'+tone(net)+'">'+gammaMoney(net,'B')+'</strong><small>'+safe(g.regime_short)+'</small>'
    +(prev?'<small>Session change '+gammaMoney(net-prev.total_net_gex_billions,'B')+'</small>':'')+'</div>');
  return '<section class="gamma-kpis in" aria-label="Key gamma levels">'+levels.join('')+'</section>';
}
const optionSnapshots = new Map();
const volSelection = {filter:'High OI benchmarks', expiry:'', front:'', back:''};
let volMessage = '';
const formatFactor = n => Number.isFinite(n) ? n.toFixed(2)+'×' : '—';
const formatMove = n => Number.isFinite(n) ? '±'+money(n) : '—';
const termTenor = row => row.Benchmark_Tenor || row.Structure || '—';
const expiryPattern = /^\d{4}-\d{2}-\d{2}$/;
const termOrder = (a,b) => a.DTE-b.DTE || a.Expiration.localeCompare(b.Expiration);
function optionSnapshot(payload) {
  const ticker=typeof payload?.ticker==='string'?payload.ticker.toUpperCase():'';
  if(!Object.hasOwn(data.prices,ticker)||!Array.isArray(payload.term_chains)) throw new Error('Choose a Volatility Pricing response for a ticker in this preview.');
  const chains=payload.term_chains.filter(r=>r&&typeof r.Expiration==='string'&&expiryPattern.test(r.Expiration)&&Number.isFinite(r.DTE)&&r.DTE>=0&&Number.isFinite(r.ATM_IV)&&r.ATM_IV>0).sort(termOrder);
  if(!chains.length) throw new Error('That response has no valid expiration chains.');
  let snapshot=optionSnapshots.get(ticker);
  if(!snapshot||snapshot.chains!==chains) snapshot={...payload,chains,smiles:new Map()};
  if(payload.smile?.expiration&&chains.some(row=>row.Expiration===payload.smile.expiration)) snapshot.smiles.set(payload.smile.expiration,payload.smile);
  optionSnapshots.set(ticker,snapshot);
  resetVolSelections();
  state.symbol=ticker;
  volSelection.expiry=snapshot.smiles.keys().next().value||chains[0].Expiration;
  volMessage='Loaded options response for '+ticker+'.';
}
function addSmileSnapshot(payload) {
  const ticker=typeof payload?.ticker==='string'?payload.ticker.toUpperCase():'';
  const snapshot=optionSnapshots.get(ticker), smile=payload?.smile;
  if(!snapshot||!smile||!snapshot.chains.some(r=>r.Expiration===smile.expiration)) throw new Error('Load the matching ticker’s term response before importing this smile.');
  resetVolSelections();
  snapshot.smiles.set(smile.expiration,smile);
  state.symbol=ticker;
  volSelection.expiry=smile.expiration;
  volMessage='Loaded '+ticker+' smile for '+smile.expiration+'.';
}
function resetVolSelections() {
  Object.assign(volSelection,{expiry:'',front:'',back:''});
  volMessage='';
}
function volControls() {
  const symbols=Object.keys(data.prices).sort();
  return '<div class="vol-toolbar controls"><label>Underlying <select id="vol-symbol">'
    +symbols.map(t=>'<option value="'+t+'" '+(state.symbol===t?'selected':'')+'>'+t+'</option>').join('')
    +'</select></label><button id="vol-refresh">Refresh options</button>'
    +'<span class="muted" role="status">'+safe(volMessage||'Saved data preview')+'</span></div>';
}
function volBaseline(snapshot) {
  const rows=priceRows(), r=returns(rows), saved=latest();
  const clean=snapshot?.clean_hv??cleanHv(r.slice(-30)), annual=snapshot?.hv_1y??hv(r.slice(-252));
  const sameSession=saved?.date===rows.at(-1)[0];
  const implied=snapshot?.chains[0]?.ATM_IV??saved?.current_iv;
  const ivStats=snapshot?.iv_stats??saved??{};
  const eligible=!!snapshot||sameSession;
  return {clean,annual,implied,ivStats,spread:eligible?implied-clean:NaN,ratio:eligible&&clean>0?implied/clean:NaN};
}
function ivRegime(stats) {
  const n=stats.iv_percentile;
  const regimes={
    cherry:['Elevated Premium (Top 20% Vol)','high'],
    amber:['Above Average Vol','above'],
    mint:['Deep Cheap Vol (Bottom 20%)','low'],
    sky:['Below Average Vol','below'],
    slate:['Neutral Volatility','neutral']
  };
  const style=Object.hasOwn(regimes,stats.style)?stats.style:
    !Number.isFinite(n)?'':n>=80?'cherry':n>=60?'amber':n<=20?'mint':n<=40?'sky':'slate';
  const [label,tone]=regimes[style]||['IV regime unavailable','neutral'];
  return {label:stats.vol_regime||label,tone};
}
function volKpis(snapshot, baseline) {
  const {clean,implied,ivStats,spread,ratio}=baseline;
  const regime=ivRegime(ivStats);
  const range=Number.isFinite(ivStats.iv_52w_low)&&Number.isFinite(ivStats.iv_52w_high)
    ?'52W '+pct(ivStats.iv_52w_low)+' – '+pct(ivStats.iv_52w_high):'52W IV range unavailable';
  const term=snapshot?.term_regime;
  const termTone=term?.style==='mint'?'low':term?.style==='cherry'?'high':'neutral';
  const tile=(label,value,detail,tone='neutral')=>'<div class="gamma-kpi vol-kpi" data-regime="'+tone+'"><small>'+label+'</small><div class="vol-kpi-value">'+value+'</div><div class="vol-kpi-details">'+detail+'</div></div>';
  const tiles=[
    tile('Spot price','<strong>'+money(snapshot?.spot_price??priceRows().at(-1)[4])+'</strong>','<small>'+state.symbol+' · '+(snapshot?'Options response':'Saved close')+'</small>'),
    tile('1-year IV percentile','<strong class="vol-regime-text">'+pct(ivStats.iv_percentile)+'</strong><small>IV rank '+(Number.isFinite(ivStats.iv_rank)?ivStats.iv_rank.toFixed(0)+' / 100':'—')+'</small>',
      '<span class="vol-regime-pill">'+safe(regime.label)+'</span><small>'+range+'</small>',regime.tone),
    tile(snapshot?'Front ATM IV':'Saved implied volatility','<strong>'+pct(implied)+'</strong>','<small>'+(snapshot?'Expiry '+snapshot.chains[0].Expiration:latest()?.date||'No IV snapshot')+'</small>'),
    tile('30D clean HV &amp; VRP','<strong>'+pct(clean)+'</strong><small class="'+(Number.isFinite(spread)?tone(spread):'')+'">'+(Number.isFinite(spread)?signed(spread,' pp')+' VRP':'VRP unavailable')+'</small>',
      '<small>IV / clean HV '+formatFactor(ratio)+'</small>'),
    tile('Term structure regime','<strong class="vol-regime-text vol-term-value">'+safe(term?.name||'—')+'</strong>',
      '<small>'+safe(term?.description||'Requires expiry-level IV')+'</small>',termTone)
  ];
  return '<section class="vol-kpis in" aria-label="Volatility metrics">'+tiles.join('')+'</section>';
}
function volUnavailable(text) {
  return '<div class="vol-empty">'+text+'</div>';
}
function volCurve(series, references, label, options={}) {
  const points=series.flatMap(item=>item.points);
  if(!points.length) return volUnavailable('No valid points in this options response.');
  const {spot,move,atm,markers=[],term=false,strikeRange}=options;
  const finiteRefs=references.filter(r=>Number.isFinite(r.value)&&r.value>0);
  const fitted=series.filter(item=>!item.scatter).flatMap(item=>item.points);
  const core=fitted.length?fitted:points;
  const anchors=[...core.map(p=>p.x),...markers.map(p=>p.x)];
  if(Number.isFinite(spot)&&spot>0) anchors.push(spot);
  const [low,high]=strikeRange||[Math.min(...anchors),Math.max(...anchors)];
  const pad=strikeRange?0:(high-low)*.045||1;
  const sx=v=>L+(R-L)*(v-low+pad)/(high-low+2*pad);
  const inRange=point=>point.x>=low&&point.x<=high;
  const visible=core.filter(inRange);
  const visibleMarkers=markers.filter(inRange);
  const vals=[...(visible.length?visible:core).map(p=>p.y),...finiteRefs.map(r=>r.value),...visibleMarkers.map(p=>p.y)];
  const lo=Math.min(...vals), hi=Math.max(...vals), headroom=Math.max((hi-lo)*.18,1.8);
  const bottom=Math.max(0,lo-headroom), top=hi+headroom;
  const y=v=>B-(v-bottom)*(B-T)/(top-bottom);
  const ticks=Array.from({length:5},(_,i)=>top-i*(top-bottom)/4);
  const id=term?'term-plot':'smile-plot';
  const xy=p=>sx(p.x).toFixed(2)+','+y(p.y).toFixed(2);
  const text=(x,y,label,anchor='start',color='var(--muted)')=>'<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" style="fill:'+color+'">'+safe(label)+'</text>';
  let under='';
  if(Number.isFinite(spot)&&Number.isFinite(move)&&move>0){
    const left=Math.max(L,sx(spot-move)), right=Math.min(R,sx(spot+move));
    if(right>left) under+='<rect x="'+left+'" y="'+T+'" width="'+(right-left)+'" height="'+(B-T)+'" fill="var(--iv)" opacity=".09"><title>One-standard-deviation expected move</title></rect>';
  }
  if(Number.isFinite(atm)){
    for(const item of series.filter(s=>!s.scatter)){
      item.points.slice(1).forEach((right,i)=>{
        const left=item.points[i], segments=[left,right];
        if((left.y-atm)*(right.y-atm)<0) segments.splice(1,0,{x:left.x+(right.x-left.x)*(atm-left.y)/(right.y-left.y),y:atm});
        segments.slice(1).forEach((b,j)=>{
          const a=segments[j], color=(a.y+b.y)/2>=atm?'var(--red)':'var(--green)';
          under+='<path d="M'+xy(a)+'L'+xy(b)+'V'+y(atm)+'H'+sx(a.x)+'Z" fill="'+color+'" opacity=".16"/>';
        });
      });
    }
  }
  let ink=finiteRefs.map(r=>'<path d="M'+L+','+y(r.value)+'H'+R+'" stroke="'+r.color+'" stroke-dasharray="'+r.dash+'" opacity=".8"/>').join('');
  if(Number.isFinite(spot)&&spot>0) ink+='<path d="M'+sx(spot)+','+T+'V'+B+'" stroke="var(--muted)" stroke-dasharray="4 4"/>';
  const dot=(p,item)=>{
    const color=p.color||item.color;
    const detail=p.detail||p.label+' · '+pct(p.y);
    return '<circle class="vol-point" cx="'+sx(p.x)+'" cy="'+y(p.y)+'" r="'+(p.radius||item.radius||3)+'" fill="'+color+'"'
      +(p.outline?' stroke="var(--text)" stroke-width=".85"':'')+(item.scatter?' opacity=".34"':' tabindex="0" aria-label="'+safe(detail)+'"')
      +' data-vol-tip="'+safe(detail)+'"><title>'+safe(detail)+'</title></circle>';
  };
  ink+=series.filter(s=>s.scatter).flatMap(item=>item.points.filter(inRange).map(p=>dot(p,item))).join('');
  ink+=series.filter(s=>!s.scatter&&s.points.length).map(item=>{
    let path='M'+xy(item.points[0]);
    item.points.slice(1).forEach((p,i)=>{
      const prev=item.points[i], mid=(sx(prev.x)+sx(p.x))/2;
      path+=term?'C'+mid+','+y(prev.y)+' '+mid+','+y(p.y)+' '+xy(p):'L'+xy(p);
    });
    return '<path d="'+path+'" fill="none" stroke="'+item.color+'" stroke-width="2.3"/>'
      +(term?item.points.map(p=>dot(p,item)).join(''):'');
  }).join('');
  ink+=visibleMarkers.map(p=>dot(p,p)+text(sx(p.x),y(p.y)-10,p.label,'middle',p.color)).join('');
  const gridLines=grid(y,ticks,v=>v.toFixed(1)+'%',true);
  const refs=finiteRefs.map((r,i)=>text(R-4,y(r.value)+(i%2?12:-5),r.label+' '+pct(r.value),'end',r.color)).join('');
  const spotText=Number.isFinite(spot)&&spot>0?text(sx(spot),T-5,'Spot '+money(spot),'middle'):'';
  const step=Math.max(1,Math.ceil(core.length/8));
  const tickStep=high-low<20?1:10;
  const firstTick=Math.ceil(low/tickStep)*tickStep;
  const strikeTicks=Array.from({length:Math.max(0,Math.floor((high-firstTick)/tickStep)+1)},(_,i)=>firstTick+i*tickStep);
  const xLabels=term
    ?core.filter((p,i)=>i%step===0).map(p=>text(sx(p.x),B+12,p.label,'middle')+text(sx(p.x),B+24,p.dte+'D','middle')).join('')
        :(strikeRange?strikeTicks:
      Array.from({length:5},(_,i)=>low+(high-low)*i/4)).map(v=>text(sx(v),B+17,'$'+Math.round(v),'middle')).join('');
  const legend=[...series.filter(s=>s.points.length),...finiteRefs];
  if(Number.isFinite(move)&&move>0) legend.push({label:'1σ expected move',color:'var(--iv)'});
  return '<div class="chart vol-study"><svg class="draw" role="group" aria-label="'+label+'" viewBox="0 0 '+W+' '+H+'">'
    +'<defs><clipPath id="'+id+'"><rect x="'+L+'" y="'+T+'" width="'+(R-L)+'" height="'+(B-T)+'"/></clipPath></defs>'
    +gridLines+'<g clip-path="url(#'+id+')">'+under+ink+'</g>'+refs+spotText+xLabels+'</svg><div class="tooltip vol-study-tip" hidden></div></div>'
    +chartKey(legend);
}
function bindVolStudyTips() {
  document.querySelectorAll('.vol-study').forEach(chart=>{
    const tip=chart.querySelector('.vol-study-tip');
    const show=(point,event)=>{
      const bounds=chart.getBoundingClientRect(), rect=point.getBoundingClientRect();
      tip.textContent=point.dataset.volTip;
      tip.hidden=false;
      const left=event?.clientX??rect.right, top=event?.clientY??rect.top;
      tip.style.left=Math.max(0,Math.min(left-bounds.left+8,bounds.width-tip.offsetWidth))+'px';
      tip.style.top=Math.max(0,Math.min(top-bounds.top+8,bounds.height-tip.offsetHeight))+'px';
    };
    const hide=()=>{tip.hidden=true;};
    chart.querySelectorAll('[data-vol-tip]').forEach(point=>{
      point.addEventListener('pointerenter',event=>show(point,event));
      point.addEventListener('pointerleave',hide);
      point.addEventListener('focus',()=>show(point));
      point.addEventListener('blur',hide);
    });
  });
}

function visibleChains(snapshot) {
  const benchmarks=snapshot.chains.filter(row=>row.Is_Benchmark);
  return volSelection.filter==='High OI benchmarks'&&benchmarks.length?benchmarks:snapshot.chains;
}
function expirySelect(chains,id,label,value) {
  return '<label>'+label+' <select id="'+id+'">'+chains.map(row=>'<option value="'+row.Expiration+'" '+(row.Expiration===value?'selected':'')+'>'+row.Expiration+' · '+row.DTE+'D</option>').join('')+'</select></label>';
}
function termStructure(snapshot,baseline) {
  if(!snapshot) return card('Term structure','ATM IV by expiration',volUnavailable('Connect the options API to plot expiration IV.'));
  const rows=visibleChains(snapshot);
  const points=rows.map((r,i)=>({x:i,y:r.ATM_IV,label:short(r.Expiration),dte:r.DTE,radius:r.Is_Benchmark?5:3.5,outline:true,
    color:r.Spans_Earnings?'var(--amber)':r.Structure==='Contango'?'var(--green)':r.Structure==='Backwardation'?'var(--red)':'var(--iv)',
    detail:[r.Expiration+' · '+r.DTE+' DTE',termTenor(r)+(r.Spans_Earnings?' · Spans earnings':''),
      'ATM IV '+pct(r.ATM_IV)+' at '+money(r.ATM_Strike),'Slope '+signed(r.Slope_Front,' pp'),
      'vs. 30D clean HV '+signed(r.ATM_IV-baseline.clean,' pp'),'vs. 1Y normal HV '+signed(r.ATM_IV-baseline.annual,' pp'),
      '1σ move '+formatMove(r.Expected_Move)+' · OI '+contracts(r.Total_OI),
      'Call bid / ask '+money(r.Call_Bid)+' / '+money(r.Call_Ask)].join('\n')
  }));
  const refs=[{label:'30D clean HV',value:baseline.clean,color:'var(--iv)',dash:'6 4'},{label:'1Y normal HV',value:baseline.annual,color:'var(--muted)',dash:'2 4'}];
  return card('Expiration maturity · DTE','ATM IV by expiration',
    volCurve([{label:'ATM implied volatility',color:'var(--text)',points}],refs,'ATM implied volatility by expiration',{term:true})
    +'<p class="vol-chart-hint">Larger dots mark benchmark expirations. Ember marks an earnings event.</p>');
}
function smileCard(snapshot) {
  const title='Volatility Smile &amp; Skew';
  if(!snapshot) return card('Strike pricing',title,volUnavailable('Connect the options API to inspect smile and skew.'));
  const expiry=volSelection.expiry||snapshot.selected_expiration||snapshot.chains[0].Expiration;
  const smile=snapshot.smiles.get(expiry);
  const selector=expirySelect(snapshot.chains,'vol-expiry','Expiration',expiry);
  if(!smile) return card('Strike pricing',title,volUnavailable('No smile is loaded for '+safe(expiry)+'.'),selector);
  const curves=[
    ['OTM puts',smile.otm_put_curve,'var(--red)',false],['OTM calls',smile.otm_call_curve,'var(--green)',false],
    ['Market put IV',smile.otm_puts_market,'var(--red)',true],['Market call IV',smile.otm_calls_market,'var(--green)',true]
  ].map(([label,curve,color,scatter])=>({
    label,color,scatter,radius:2.6,points:(Array.isArray(curve?.strikes)?curve.strikes:[])
      .map((x,i)=>({x,y:curve.iv?.[i],label:label+' · '+money(x)})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)).sort((a,b)=>a.x-b.x)
  }));
  const m=smile.metrics||{}, structural=smile.structural_markers||{};
  const markers=[['ATM',structural.atm,'var(--amber)'],['25Δ put',structural.put_25d,'var(--red)'],['25Δ call',structural.call_25d,'var(--green)']]
    .filter(([,p])=>Number.isFinite(p?.strike)&&Number.isFinite(p.iv))
    .map(([label,p,color])=>({label,x:p.strike,y:p.iv,color,radius:4.5,outline:true,detail:label+' · '+money(p.strike)+'\nIV '+pct(p.iv)}));
  const atm=structural.atm?.iv??m.atm_iv;
  const detail=[['ATM IV',pct(m.atm_iv)],['25Δ skew',signed(m.skew_25d,' pp')],['25Δ fly',signed(m.fly_25d,' pp')],['10Δ skew',signed(m.skew_10d,' pp')],['Strike slope',signed(m.skew_slope,'')],['Term slope',signed(m.term_slope,' pp')],['1σ move',formatMove(smile.exp_move)]];
  const metrics='<div class="vol-metrics smile-metrics">'+detail.map(([k,v])=>'<span>'+k+' <strong>'+v+'</strong></span>').join('')+'</div>';
  const spot=snapshot.spot_price;
  const center=[spot,structural.atm?.strike,m.atm_strike].find(v=>Number.isFinite(v)&&v>0);
  const strikeRange=center?[center*.95,center*1.05]:undefined;
  const plot=volCurve(curves,[{label:'ATM',value:atm,color:'var(--amber)',dash:'5 4'}],'Implied volatility by strike',
    {atm,spot,move:smile.exp_move,markers,strikeRange});
  return card(safe(expiry)+' · '+safe(smile.dte)+' DTE',title,metrics+plot,selector);
}
function termMatrix(snapshot) {
  if(!snapshot) return card('Expiry comparison','Multi-Expiry Contract Matrix',volUnavailable('Load an options response to compare expirations, bid/ask, expected moves, and catalysts.'));
  const headers=['Expiry','Tenor','DTE','ATM strike','ATM IV','Term slope','vs. clean HV','vs. 1Y HV','Call bid / ask','Open interest','1σ move','Catalyst','Inspect'];
  const rows=visibleChains(snapshot).map(r=>{
    const values=[safe(r.Expiration),safe(termTenor(r)),r.DTE,money(r.ATM_Strike),pct(r.ATM_IV),signed(r.Slope_Front,' pp'),signed(r.Spread_HV_Clean,' pp'),signed(r.Spread_HV_1Y,' pp'),money(r.Call_Bid)+' / '+money(r.Call_Ask),contracts(r.Total_OI),formatMove(r.Expected_Move),r.Spans_Earnings?'Earnings':'—','<button data-vol-inspect="'+r.Expiration+'" aria-label="Inspect smile for '+r.Expiration+'">Smile</button>'];
    return '<tr>'+values.map((v,i)=>i===0?'<th scope="row">'+v+'</th>':'<td>'+v+'</td>').join('')+'</tr>';
  }).join('');
  const filters=['High OI benchmarks','All expirations'].map(v=>'<button data-vol-filter="'+v+'" aria-pressed="'+(volSelection.filter===v)+'">'+v+'</button>').join('');
  return card('Expiry comparison','Multi-Expiry Contract Matrix','<div class="gamma-table-wrap" tabindex="0" role="region" aria-label="Expiration comparison"><table class="gamma-table vol-matrix"><thead><tr>'+headers.map(h=>'<th scope="col">'+h+'</th>').join('')+'</tr></thead><tbody>'+rows+'</tbody></table></div>',filters);
}
function forwardMetrics(front,back) {
  const fwd=(iv1,iv2)=>{
    if(!Number.isFinite(iv1)||!Number.isFinite(iv2)||iv1<=0||iv2<=0||back.DTE<=front.DTE) return NaN;
    const variance=(iv2**2*back.DTE-iv1**2*front.DTE)/(back.DTE-front.DTE);
    return variance>0?Math.sqrt(variance):NaN;
  };
  return {
    mid:fwd(front.ATM_IV,back.ATM_IV),
    executable:fwd(front.Call_Bid_IV,back.Call_Ask_IV),
    debit:front.Call_Bid>0&&back.Call_Ask>0?back.Call_Ask-front.Call_Bid:NaN,
    theta:Number.isFinite(front.Call_Theta)&&Number.isFinite(back.Call_Theta)?Math.abs(front.Call_Theta)-Math.abs(back.Call_Theta):NaN
  };
}
function forwardDesk(snapshot, baseline) {
  const title='Forward Volatility &amp; Relative Value';
  if(!snapshot||snapshot.chains.length<2) return card('Execution',''+title,volUnavailable('Requires two expiration chains with executable call bid and ask marks.'));
  const rows=snapshot.chains, prime=(Array.isArray(snapshot.rv_opportunities)?snapshot.rv_opportunities:[]).filter(p=>p?.is_prime&&rows.some(r=>r.Expiration===p.front_exp)&&rows.some(r=>r.Expiration===p.back_exp));
  const front=rows.slice(0,-1).find(r=>r.Expiration===volSelection.front)||rows.find(r=>r.Expiration===prime[0]?.front_exp)||rows[0];
  const later=rows.filter(r=>r.DTE>front.DTE);
  const back=later.find(r=>r.Expiration===volSelection.back)||later.find(r=>r.Expiration===prime[0]?.back_exp)||later[0];
  const validFront=rows.filter(r=>r.DTE<rows.at(-1).DTE);
  if(!back) return card('Execution',title,volUnavailable('Two distinct days-to-expiration values are required.'));
  const {mid,executable,debit,theta}=forwardMetrics(front,back);
  const metrics=[['Net debit',money(debit)],['Executable forward IV',pct(executable)],['Mid forward IV',pct(mid)],['vs. clean HV',signed(executable-baseline.clean,' pp')],['vs. 1Y normal HV',signed(executable-baseline.annual,' pp')],['Daily theta per share',signed(theta,'')]];
  const picks=prime.slice(0,5).map(p=>'<button data-vol-pair="'+p.front_exp+'/'+p.back_exp+'">'+p.front_exp+' → '+p.back_exp+'</button>').join('');
  const legs=stat('Sell '+front.Expiration,money(front.Call_Bid)+' bid · '+money(front.ATM_Strike)+' strike')+stat('Buy '+back.Expiration,money(back.Call_Ask)+' ask · '+money(back.ATM_Strike)+' strike');
  return card('Execution',title,(picks?'<div class="controls vol-pairs">'+picks+'</div>':'')+legs+'<div class="vol-metrics">'+metrics.map(([k,v])=>'<span>'+k+' <strong>'+v+'</strong></span>').join('')+'</div>',
    expirySelect(validFront,'vol-front','Short leg',front.Expiration)+expirySelect(later,'vol-back','Long leg',back.Expiration));
}
function volatilityPage(date) {
  const snapshot=optionSnapshots.get(state.symbol), baseline=volBaseline(snapshot);
  const header=hero(state.symbol+' / Options','Volatility Pricing','Compare implied volatility across strikes and expirations.',date,true);
  const fallback=snapshot?'':'<p class="vol-source-note">Expiration analytics load from the original dashboard API. Saved IV and realized volatility remain available while it is offline.</p>';
  return header+volControls()+fallback+volKpis(snapshot,baseline)+'<div class="vol-panels">'
    +termStructure(snapshot,baseline)+smileCard(snapshot)+termMatrix(snapshot)+forwardDesk(snapshot,baseline)
    +'<div class="grid vol-realized">'+volChart()+'<div class="stack in">'+volStats(true)+'</div></div></div>';
}
function bindVolatility() {
  const watchSelect=(id,fn)=>byId(id)?.addEventListener('change',event=>{fn(event.target.value);render();byId(id)?.focus({preventScroll:true});});
  watchSelect('vol-symbol',symbol=>{resetVolSelections();state.symbol=symbol;state.gammaDate='';});
  watchSelect('vol-expiry',value=>{volSelection.expiry=value;});
  watchSelect('vol-front',value=>{volSelection.front=value;volSelection.back='';});
  watchSelect('vol-back',value=>{volSelection.back=value;});
  byId('vol-refresh')?.addEventListener('click',()=>loadPageData(true));

}

function gammaChart(g, strikes) {
  const spot=g.spot_price, mode=state.gammaMode;
  const rangePct={'±3% ATM':.03,'±8% Active':.08,'±15% Macro':.15};
  const offset=state.gammaRange==='Relevant' ? Math.max(spot*.055,...gammaLevels(g).filter(([,v])=>Number.isFinite(v)&&v>0).map(([,v])=>Math.abs(v-spot)*1.35)) : spot*rangePct[state.gammaRange];
  const all=state.gammaRange==='All saved strikes';
  const span=all?[strikes[0].strike-1,strikes.at(-1).strike+1]:[Math.floor(spot-offset),Math.ceil(spot+offset)];
  const rows=strikes.filter(row=>row.strike>=span[0]&&row.strike<=span[1]);
  const toggles=control('gammaMode',['Net GEX','Call / Put','Open interest']);
  const ranges='<div class="gamma-ranges controls" aria-label="Strike range">'+control('gammaRange',['Relevant',...Object.keys(rangePct),'All saved strikes'])+'</div>';
  if(!rows.length) return card('Strike profile','Gamma Exposure',ranges+'<div class="empty">No saved strikes in this range.</div>',toggles,'gamma-plot in');
  const xValue=v=>L+(R-L)*(v-span[0])/(span[1]-span[0]);
  const x=i=>xValue(rows[i].strike);
  const oi=mode==='Open interest';
  const values=rows.map(row=>mode==='Net GEX'?[row.net_gex]:oi?[row.call_oi,row.put_oi]:[row.call_gex,-row.put_gex]);
  const {y,ticks}=scale([0,...values.flat().filter(Number.isFinite)]);
  const width=Math.max(.45,Math.min(14,(R-L)/rows.length*.48));
  let marks='<line x1="'+L+'" x2="'+R+'" y1="'+y(0)+'" y2="'+y(0)+'" stroke="#637083"/>';
  marks+=values.map((series,i)=>series.map((v,j)=>{
    if(!Number.isFinite(v)) return '';
    const color=(oi?j===0:v>=0)?'var(--green)':'var(--red)';
    return '<rect class="gamma-bar" style="--zero:'+y(0)+'px" x="'+(x(i)-width/2+(oi?(j-.5)*width:0))+'" y="'+Math.min(y(v),y(0))+'" width="'+width+'" height="'+Math.abs(y(v)-y(0))+'" rx=".5" fill="'+color+'"/>';
  }).join('')).join('');
  marks+=gammaLevels(g).filter(([,v])=>Number.isFinite(v)&&v>=span[0]&&v<=span[1]).map(([label,v,color])=>'<line data-gamma-marker="'+label+'" x1="'+xValue(v)+'" x2="'+xValue(v)+'" y1="'+T+'" y2="'+B+'" stroke="'+color+'" stroke-opacity=".85" stroke-dasharray="3 4"><title>'+label+' '+money(v)+'</title></line>').join('');
  hoverSeries={size:rows.length,x,y:i=>y(values[i][0]||0),strike:i=>rows[i].strike,text:i=>{
    const row=rows[i];
    return 'Strike '+money(row.strike)+' · '+signed(gammaDistance(row.strike,spot))+' vs. spot'
      +'<br>Net '+gammaMoney(row.net_gex,'M')+'<br>Calls '+gammaMoney(row.call_gex,'M')+' · '+contracts(row.call_oi)+' OI'
      +'<br>Puts '+gammaMoney(-row.put_gex,'M')+' · '+contracts(row.put_oi)+' OI';
  }};
  const firstTick=Math.ceil(span[0]/5)*5;
  const tickCount=Math.max(0,Math.floor((span[1]-firstTick)/5)+1);
  const labelStep=Math.max(1,Math.ceil(tickCount/18));
  const labels=Array.from({length:tickCount},(_,i)=>{
    const strike=firstTick+i*5, at=xValue(strike);
    const tick='<line x1="'+at+'" x2="'+at+'" y1="'+B+'" y2="'+(B+4)+'" stroke="var(--muted)"/>';
    return tick+(i%labelStep===0?'<text x="'+at+'" y="292" text-anchor="middle">$'+strike+'</text>':'');
  }).join('');
  const legend=gammaLevels(g).map(([label,,color])=>'<span><i style="background:'+color+'"></i>'+label+'</span>').join('');
  return card('Strike profile','Gamma Exposure',ranges
    +plotFrame('Gamma exposure by strike. Use the arrow keys to inspect strikes.',grid(y,ticks,v=>oi?Math.round(v/1000)+'K':v.toFixed(0)+'M',false),marks,labels,true)
    +'<div class="chart-note"><span>'+rows.length+' saved strikes · '+(oi?'Open interest in contracts':'GEX in USD millions')+'</span><div class="gamma-legend">'+legend+'</div></div>',toggles,'gamma-plot in');
}
function gammaBreakdown(g, strikes) {
  const closest=target=>Number.isFinite(target)&&target>0 ? strikes.reduce((a,b)=>Math.abs(a.strike-target)<=Math.abs(b.strike-target)?a:b).strike : null;
  const tags = new Map();
  const tag = (row,label) => { if(!tags.has(row)) tags.set(row,label); };
  for(const [label,target] of [[ 'Call wall',g.call_wall],['Put wall',g.put_wall],['Zero flip',closest(g.zero_gamma_level)],['ATM spot',closest(g.spot_price)]]) {
    if(Number.isFinite(target)&&target>0) tag(target,label);
  }
  const byRank = [
    ['Absolute gamma peak',row=>row.call_gex+row.put_gex,1],
    ['Top call GEX',row=>row.call_gex,5],['Top put GEX',row=>row.put_gex,5],
    ['Net GEX imbalance',row=>Math.abs(row.net_gex),5],['Major open interest',row=>row.call_oi+row.put_oi,5]
  ];
  for(const [label,rank,count] of byRank) {
    [...strikes].filter(row=>rank(row)>0).sort((a,b)=>rank(b)-rank(a)).slice(0,count).forEach(row=>tag(row.strike,label));
  }
  const step=g.spot_price>100?5:g.spot_price>50?2.5:1;
  for(const row of strikes) {
    if(Math.abs(gammaDistance(row.strike,g.spot_price))>3.5) continue;
    if(row.call_oi+row.put_oi>=12000) tag(row.strike,'Major open interest');
    else if(row.strike%step===0) tag(row.strike,'Milestone strike');
  }
  if(strikes.some(row=>typeof row.is_relevant==='boolean')){
    tags.clear();
    strikes.filter(row=>row.is_relevant).forEach(row=>tags.set(row.strike,row.structural_role));
  }
  const relevant=strikes.filter(row=>tags.has(row.strike));
  const displayed=state.gammaTable==='Relevant only'&&relevant.length?relevant:strikes;
  const peakGex=key=>displayed.reduce((peak,row)=>Math.max(peak,Number.isFinite(row[key])?Math.abs(row[key]):0),0);
  const callPeak=peakGex('call_gex'), putPeak=peakGex('put_gex');
  const gexCell=(value,peak,side)=>{
    const intensity=Number.isFinite(value)&&peak>0?Math.abs(value)/peak:0;
    const color=side==='call'?'--positive-rgb':'--negative-rgb';
    const background='rgb(var('+color+') / '+(intensity*.36).toFixed(3)+')';
    return '<td style="background-color:'+background+'">'+gammaMoney(value,'M')+'</td>';
  };
  const rows=displayed.map(row=>'<tr data-gamma-strike="'+row.strike+'"><td>'+contracts(row.call_oi)+'</td>'+gexCell(row.call_gex,callPeak,'call')+'<th scope="row">'+money(row.strike)+'</th><td>'+signed(gammaDistance(row.strike,g.spot_price))+'</td><td class="'+tone(row.net_gex)+'">'+gammaMoney(row.net_gex,'M')+'</td><td>'+safe(tags.get(row.strike))+'</td>'+gexCell(Number.isFinite(row.put_gex)?-row.put_gex:NaN,putPeak,'put')+'<td>'+contracts(row.put_oi)+'</td></tr>').join('');
  const labels=['Call OI','Call GEX','Strike','vs. spot','Net GEX','Structural role','Put GEX','Put OI'];
  const header='<thead><tr><th colspan="2" scope="colgroup">Calls</th><th colspan="4" scope="colgroup">Strike &amp; dealer positioning</th><th colspan="2" scope="colgroup">Puts</th></tr><tr>'+labels.map(label=>'<th scope="col">'+label+'</th>').join('')+'</tr></thead>';
  const toggles=control('gammaTable',['Relevant only','All saved strikes'],['Relevant only ('+relevant.length+')','All saved strikes ('+strikes.length+')']);
  return card('Straddle view','Strike Gamma &amp; Open Interest','<div class="gamma-table-wrap" tabindex="0" role="region" aria-label="Scrollable strike breakdown"><table class="gamma-table"><colgroup span="2"></colgroup><colgroup span="4"></colgroup><colgroup span="2"></colgroup>'+header+'<tbody>'+rows+'</tbody></table></div>',toggles,'wide in');
}
function gammaPage(date) {
  const g=activeGamma(), strikes=savedStrikes(g);
  const header=hero(state.symbol+' / Dealer positioning','Gamma Exposure Map','Explore dealer gamma, key levels, and strike open interest.',g?.date||date,true);
  const filters=gammaFilters(g);
  if(!g||!strikes.length) return header+filters+'<div class="card empty">No saved gamma strikes for '+state.symbol+'. Choose another underlying.</div>';
  return header+filters+gammaSummary(g)+'<div class="gamma-workspace">'+gammaChart(g,strikes)+gammaBreakdown(g,strikes)+'</div>'+heatmap();
}
function heatmap() {
  const g=activeGamma(), history=gammaRows().filter(row=>row.date<=g?.date).slice(-8);
  if(history.length<2||!g?.strikes) return '';
  const key=state.gammaMode==='Open interest'?'call_oi':'net_gex';
  const keys=Object.keys(g.strikes).sort((a,b)=>Math.abs(+a-g.spot_price)-Math.abs(+b-g.spot_price)).slice(0,12).sort((a,b)=>+b-+a);
  const max=Math.max(1,...history.flatMap(row=>keys.map(k=>Math.abs(row.strikes?.[k]?.[key]||0))));
  const rows=keys.map(k=>'<tr><th scope="row">'+safe(k)+'</th>'+history.map(row=>{
    const v=row.strikes?.[k]?.[key];
    return Number.isFinite(v)?'<td style="background:rgb('+(v>=0?'var(--positive-rgb)':'var(--negative-rgb)')+' / '+(.09+Math.abs(v)/max*.43)+')">'+(key==='net_gex'?v.toFixed(1):v.toLocaleString())+'</td>':'<td>—</td>';
  }).join('')+'</tr>').join('');
  return card('Snapshot history',key==='net_gex'?'Net GEX migration · $M':'Call open-interest migration · contracts','<div class="heat-scroll"><table class="heat"><thead><tr><th scope="col">Strike</th>'+history.map(row=>'<th scope="col">'+short(row.date)+'</th>').join('')+'</tr></thead><tbody>'+rows+'</tbody></table></div>','','wide in');
}
function gammaStats(g) {
  return card('Positioning','Gamma levels',g ? '<div class="big '+tone(g.total_net_gex_billions)+'">'+signed(g.total_net_gex_billions,'B')+'</div><p>'+safe(g.regime_short)+' · net gamma exposure</p>'
    +stat('Call wall',money(g.call_wall),'up')+stat('Zero gamma',money(g.zero_gamma_level),'amber')+stat('Put wall',money(g.put_wall),'down')+stat('Snapshot',safe(g.date)) : '<p>No saved gamma snapshot for '+state.symbol+'.</p>');
}
function volStats(expanded) {
  const rows=priceRows(), r=returns(rows), g=latest(), h30=hv(r.slice(-30));
  const same=g?.date===rows.at(-1)[0], iv=g?.current_iv;
  const park=Math.sqrt(rows.slice(-30).reduce((s,row)=>s+Math.log(row[2]/row[3])**2,0)/30*252/(4*Math.log(2)))*100;
  return card('Volatility','Implied vs. realized','<div class="big amber">'+pct(iv)+'</div><p>Saved implied volatility'+(g?' · '+safe(g.date):' unavailable')+'</p>'+stat('30D realized',pct(h30))+stat('IV − 30D HV',same?signed(iv-h30,' pp'):'—','amber')+(expanded?stat('IV rank',Number.isFinite(g?.iv_rank)?g.iv_rank.toFixed(0)+' / 100':'—')+stat('IV percentile',pct(g?.iv_percentile))+stat('30D clean HV',pct(cleanHv(r.slice(-30))))+stat('30D Parkinson HV',pct(park))+'<details><summary>Volatility methodology</summary><p>Realized volatility uses the sample standard deviation of daily close-to-close returns, annualized over 252 sessions. Clean HV clips returns at three robust standard deviations around the median. Parkinson HV uses daily high–low ranges.</p></details>':'<a class="jump" href="#volatility">Open volatility lab →</a>'));
}
function priceStats() {
  const rows=priceRows(), a=rows.at(-1), ch=daily(rows);
  return card('Session',''+state.symbol+' daily range','<div class="big">'+money(a[4])+'</div><p class="'+tone(ch)+'">'+signed(ch)+' close to close</p>'+stat('Open',money(a[1]))+stat('High',money(a[2]))+stat('Low',money(a[3]))+stat('Volume',a[5].toLocaleString('en-US')));
}
function bindChart() {
  if(!hoverSeries) return;
  const series=hoverSeries, svg=document.querySelector('.chart svg[tabindex]'), tip=svg.parentElement.querySelector('.tooltip');
  const cross=byId('cross'), dot=byId('cross-dot');
  const strikeRows=series.strike?new Map([...document.querySelectorAll('[data-gamma-strike]')].map(row=>[Number(row.dataset.gammaStrike),row])):null;
  let activeRow;
  const highlightStrike=strike=>{
    const next=strikeRows?.get(strike);
    if(next===activeRow) return;
    activeRow?.classList.remove('gamma-chart-match');
    next?.classList.add('gamma-chart-match');
    activeRow=next;
    if(!next||!window.matchMedia('(min-width: 1100px)').matches) return;
    const viewport=next.closest('.gamma-table-wrap');
    const bounds=viewport.getBoundingClientRect(), row=next.getBoundingClientRect();
    const top=bounds.top+viewport.querySelector('thead').getBoundingClientRect().height;
    if(row.top<top) viewport.scrollTop+=row.top-top;
    else if(row.bottom>bounds.bottom) viewport.scrollTop+=row.bottom-bounds.bottom;
  };
  let index=series.size-1;
  const draw=i=>{
    index=Math.max(0,Math.min(series.size-1,i));
    const x=series.x(index), y=series.y(index);
    cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
    dot.setAttribute('cx',x);dot.setAttribute('cy',y);dot.setAttribute('visibility','visible');
    tip.innerHTML=series.text(index);tip.hidden=false;
    if(series.strike) highlightStrike(series.strike(index));
    const position=x/W*svg.clientWidth;
    tip.style.left=Math.max(0,Math.min(position+9,svg.clientWidth-tip.offsetWidth))+'px';
    tip.style.top=Math.max(0,Math.min(y/H*svg.clientHeight-36,svg.clientHeight-tip.offsetHeight))+'px';
  };
  const hide=()=>{highlightStrike();tip.hidden=true;cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden');};
  svg.addEventListener('pointermove',e=>{
    const bounds=svg.getBoundingClientRect(), x=(e.clientX-bounds.left)*W/bounds.width;
    let nearest=0;
    for(let i=1;i<series.size;i++) if(Math.abs(x-series.x(i))<Math.abs(x-series.x(nearest))) nearest=i;
    draw(nearest);
  });
  svg.addEventListener('pointerleave',hide);
  svg.addEventListener('blur',hide);
  svg.addEventListener('focus',()=>draw(index));
  svg.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();draw(index+(e.key==='ArrowLeft'?-1:1));}});
}
const biasScopes={core:'Core Universe',mag7:'Mag 7 only',indexes:'Major Indexes',sectors:'Sector ETFs',macro:'Macro Assets'};
const biasSelection={scope:'core',ticker:''};
const biasPoints=score=>score>0?'+'+score:score<0?'−'+Math.abs(score):'0';
const biasTone=score=>score>=25?'up':score<=-25?'down':'muted';
function biasChart(items,selected) {
  const bars=items.map(item=>{
    const {ticker,score,bias_label,bull_count,bear_count}=item;
    const label=ticker+' '+biasPoints(score)+', '+bias_label+', '+bull_count+' bullish and '+bear_count+' bearish confirmations';
    return '<button class="bias-bar-row" data-inspect-bias="'+ticker+'" aria-label="'+label+'" aria-pressed="'+(ticker===selected)+'"><span>'+ticker+'</span><span class="bias-track" aria-hidden="true"><i class="'+biasTone(score)+'" style="--start:'+(50+Math.min(0,score)/2)+'%;--length:'+Math.abs(score)/2+'%;--bar-origin:'+(score<0?'right':'left')+'"></i></span><strong class="'+biasTone(score)+'">'+biasPoints(score)+'</strong></button>';
  }).join('');
  return card('−100 bearish to +100 bullish','Conviction scores','<div class="bias-scale"><span>−100</span><span>0</span><span>+100</span></div><div class="bias-bars" tabindex="0" role="region" aria-label="Scrollable conviction scores">'+bars+'</div>');
}
function biasTable(items,selected) {
  const headings=['Ticker','Price','Score','Bias','Bull','Bear','Status','20D win rate','RSI','RVOL','vs. 50 EMA','ATH drawdown'];
  const body=items.map(item=>{
    const m=item.metrics, edge=m.edge_match;
    const cells=[money(m.close),'<strong class="'+biasTone(item.score)+'">'+biasPoints(item.score)+'</strong>',safe(item.bias_label),
      item.bull_count+' / 7',item.bear_count+' / 7',safe(item.flicker_status),pct(edge.win_rate_20d),
      m.rsi.toFixed(1),formatFactor(m.rvol),signed(m.dist_ema50),signed(m.drawdown_ath)];
    return '<tr'+(selected===item.ticker?' class="bias-selected"':'')+'><th scope="row"><button data-inspect-bias="'+item.ticker+'" aria-label="Inspect '+item.ticker+' confirmations">'+item.ticker+'</button></th>'+cells.map(v=>'<td>'+v+'</td>').join('')+'</tr>';
  }).join('');
  return card('Asset comparison','Multi-Factor Bias Matrix','<div class="gamma-table-wrap" tabindex="0" role="region" aria-label="Scrollable bias matrix"><table class="gamma-table bias-table"><thead><tr>'+headings.map(v=>'<th scope="col">'+v+'</th>').join('')+'</tr></thead><tbody>'+body+'</tbody></table></div>');
}
function biasInspector(item,items) {
  const options=items.map(asset=>'<option value="'+asset.ticker+'" '+(asset.ticker===item.ticker?'selected':'')+'>'+asset.ticker+'</option>').join('');
  const edge=item.metrics.edge_match;
  const layers=item.checklist.map(layer=>{
    const points=layer.points, direction=points>0?'Bullish':points<0?'Bearish':'Neutral';
    return '<article class="bias-layer '+(points>0?'up':points<0?'down':'muted')+'"><div><strong>'+safe(layer.layer)+'</strong><span>'+direction+'</span><b>'+biasPoints(points)+' pts</b></div><p>'+safe(layer.detail)+'</p></article>';
  }).join('');
  const match=edge.sample_size>0?safe(edge.rule_name)+' · '+edge.sample_size+' historical matches':'No historical matches for this setup';
  return card(item.ticker+' · '+safe(item.bias_label),'Confirmation Layer Inspector','<div class="bias-layers">'+layers+'</div><p class="bias-match">'+match+'</p>',
    '<label>Asset <select id="bias-asset">'+options+'</select></label>','bias-inspector');
}
function biasPage() {
  const source=data.bias;
  const header=hero('Market conviction','Bias Matrix','Compare bullish and bearish confirmations across seven factors.',data.prices.SPY.at(-1)[0],true);
  if(!source?.assets.length) return header+'<div class="card empty">No Bias Matrix data in this snapshot. Rebuild the local snapshot to load the scores.</div>';
  const group=source.groups[biasSelection.scope], items=remoteBias.get(biasSelection.scope)||(group?source.assets.filter(asset=>group.includes(asset.ticker)):source.assets);
  const current=items.find(asset=>asset.ticker===biasSelection.ticker)||items[0];
  const scopes=Object.entries(biasScopes).map(([key,label])=>'<button data-bias-scope="'+key+'" aria-pressed="'+(biasSelection.scope===key)+'">'+label+'</button>').join('');
  return header+'<div class="bias-toolbar controls" aria-label="Asset universe">'+scopes+'<span>'+items.length+' assets · saved daily data</span></div>'
    +(current?'<div class="bias-main">'+biasChart(items,current.ticker)+biasInspector(current,items)+'</div>'+biasTable(items,current.ticker):'<div class="card empty">No saved assets for this universe.</div>');
}

const flowMetrics = new Map((data.bias?.assets||[]).map(asset=>[asset.ticker,asset.metrics]));
const sectorNames = {XLK:'Technology',XLF:'Financials',XLV:'Health care',XLE:'Energy',XLI:'Industrials',XLY:'Consumer discretionary',XLP:'Consumer staples',XLU:'Utilities',XLB:'Materials',XLC:'Communication services',XBI:'Biotechnology',SMH:'Semiconductors'};
const flowBaskets = [
  ['Growth',['XLK','SMH','XLC','XLY']],['Cyclicals',['XLF','XLI','XLB']],['Defensives',['XLU','XLP','XLV']],['Energy',['XLE']]
];
const mag7Palette=['#f4f5fa','#518fff','#a96aff','#ff5ab4','#00d7e8','#a5ea4b','#ff9951'];
const mutedMag7 = new Set();
const flowChange = value => '<span class="'+tone(value)+'">'+signed(value)+'</span>';
function flowTable(title,subtitle,tickers,mega=false) {
  const headings=['Ticker','Price','1D','5D','20D',...(mega?['vs. 20 EMA']:[]),'vs. 50 EMA','RSI','RVOL',
    ...(!mega?['ATR stretch']:[]),'ATH drawdown',...(mega?['20D win rate','Median 20D return']:[]),'Technical state'];
  const namesByTicker=mega?Object.fromEntries(megaCaps.map(([t,n])=>[t,n])):{...names,...sectorNames};
  const rows=tickers.filter(t=>flowMetrics.has(t)&&data.overview.assets[t]).map(t=>{
    const m=flowMetrics.get(t), a=data.overview.assets[t], edge=m.edge_match;
    const values=[money(a.close),flowChange(a.day),flowChange(a.week),flowChange(a.month),...(mega?[flowChange(m.dist_ema20)]:[]),flowChange(m.dist_ema50),
      m.rsi.toFixed(1),formatFactor(m.rvol),...(!mega?[signed(m.atr_stretch,' ATR')]:[]),signed(m.drawdown_ath),
      ...(mega?[Number.isFinite(edge.win_rate_20d)?'<span title="'+edge.sample_size+' historical matches">'+pct(edge.win_rate_20d)+'</span>':'—',flowChange(edge.median_return_20d)]:[])];
    return '<tr><th scope="row"><button data-chart-symbol="'+t+'" aria-label="Open '+t+' price chart">'+t+'</button> <small>'+safe(namesByTicker[t])+'</small></th>'
      +values.map(v=>'<td>'+v+'</td>').join('')+'<td class="flow-signal">'+safe(a.status)+'</td></tr>';
  });
  const table='<div class="gamma-table-wrap flow-scroll" tabindex="0" role="region" aria-label="'+title+'"><table class="gamma-table flow-table"><thead><tr>'
    +headings.map(h=>'<th scope="col">'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.join('')+'</tbody></table></div>';
  return card(subtitle,title,rows.length?table:'<p>No saved data for this group.</p>');
}
function basketVelocity() {
  const baskets=flowBaskets.map(([label,members])=>{
    const values=members.map(t=>flowMetrics.get(t)?.pct_5d).filter(Number.isFinite);
    return {label,members,value:values.length===members.length?values.reduce((a,b)=>a+b,0)/values.length:NaN};
  });
  const limit=Math.max(1,...baskets.map(b=>Math.abs(b.value)).filter(Number.isFinite));
  const bars=baskets.map(({label,members,value})=>{
    const width=Number.isFinite(value)?Math.abs(value)/limit*48:0;
    return '<div class="flow-basket"><span><strong>'+label+'</strong><small>'+members.join(' · ')+'</small></span>'
      +'<span class="bias-track" aria-hidden="true"><i class="'+tone(value)+'" style="--start:'+(value<0?50-width:50)+'%;--length:'+width+'%;--bar-origin:'+(value<0?'right':'left')+'"></i></span>'
      +'<strong class="'+(Number.isFinite(value)?tone(value):'muted')+'">'+signed(value)+'</strong></div>';
  }).join('');
  return card('Mean five-session return','Basket Velocity Barometer','<div class="flow-basket-bars">'+bars+'</div>');
}
function normalizedFlow() {
  const tickers=data.bias.groups.mag7.filter(t=>data.prices[t]?.length>=60);
  const series=tickers.map((ticker,i)=>({ticker,color:mag7Palette[i],prices:new Map(remoteReturns.has(ticker)?remoteReturns.get(ticker).slice(-100).map(r=>[r.date,r.close]):data.prices[ticker].slice(-100).map(r=>[r[0],r[4]]))}));
  const sessions=[...(series[0]?.prices.keys()||[])].filter(date=>series.every(s=>s.prices.has(date))).slice(-60);
  if(sessions.length<2) return card('Price performance','Normalized Returns','<p>Not enough shared sessions to plot returns.</p>');
  const visible=series.filter(s=>!mutedMag7.has(s.ticker)).map(s=>({...s,values:sessions.map(date=>(s.prices.get(date)/s.prices.get(sessions[0])-1)*100)}));
  const selector=series.map(s=>'<button data-flow-series="'+s.ticker+'" aria-pressed="'+!mutedMag7.has(s.ticker)+'"><i style="background:'+s.color+'" aria-hidden="true"></i>'+s.ticker+'</button>').join('');
  if(!visible.length) return card('Price performance','Normalized Returns','<p>Select a ticker to show its return.</p>',selector,'flow-normalized');
  const {y,ticks}=scale([0,...visible.flatMap(s=>s.values)]), x=i=>L+(R-L)*i/(sessions.length-1);
  const lines=visible.map(s=>'<path class="flow-series" data-series-path="'+s.ticker+'" d="'+line(s.values,x,y)+'" fill="none" stroke="'+s.color+'" stroke-width="2" vector-effect="non-scaling-stroke"/>').join('');
  const baseline='<line x1="'+L+'" x2="'+R+'" y1="'+y(0)+'" y2="'+y(0)+'" stroke="var(--muted)" opacity=".48"/>';
  hoverSeries={size:sessions.length,x,y:i=>y(visible[0].values[i]),text:i=>sessions[i]+'<br>'+visible.map(s=>s.ticker+' '+signed(s.values[i])).join('<br>')};
  return card(sessions.length+' shared trading sessions','Normalized Returns',
    plotFrame('Normalized returns. Use left and right arrows to inspect a session.',grid(y,ticks,pct,false),baseline+lines,axis(sessions,x,short),true)
    +'<div class="chart-note">Each ticker starts at 0% on '+short(sessions[0])+'</div>',selector,'flow-normalized');
}
function flowPage(mega) {
  const date=data.prices.SPY.at(-1)[0];
  const title=mega?'Magnificent 7 Flow':'Index and Sector Flow';
  const subtitle=mega?'Compare mega-cap performance, technical signals, and historical odds.':'Track index performance and rotation across sector ETFs.';
  const header=hero(mega?'Mega-cap equities':'Market rotation',title,subtitle,date,true);
  if(!data.bias?.assets.length) return header+'<div class="card empty">No technical diagnostics in this snapshot.</div>';
  const groups=data.bias.groups;
  return header+'<div class="flow-panels in">'+(mega
    ?flowTable('Mega-Cap Diagnostics','Momentum and historical odds',groups.mag7,true)+normalizedFlow()
    :basketVelocity()+flowTable('Sector ETF Diagnostics','Technical and flow metrics',groups.sectors)+flowTable('Major Indexes','Benchmark diagnostics',groups.indexes))+'</div>';
}

const explorerSelection={horizon:'1Y',view:'Trend'};
const ratioSelection={pair:'SPY/RSP'};
const explorerStudies=new Map();
function priceStudies(symbol) {
  if(explorerStudies.has(symbol)) return explorerStudies.get(symbol);
  let peak=0, peakIndex=0, e20, e50, e200;
  const rows=data.prices[symbol].map(([date,,,,close],i)=>{
    if(close>=peak){peak=close;peakIndex=i;}
    e20=i?e20+2/21*(close-e20):close;
    e50=i?e50+2/51*(close-e50):close;
    e200=i?e200+2/201*(close-e200):close;
    return {date,close,e20,e50,e200,dd:(close/peak-1)*100,days:i-peakIndex};
  });
  explorerStudies.set(symbol,rows);
  return rows;
}
function chartPaths(rows, series, x, y) {
  return series.map(({key,color,dash,width})=>{
    let path='', open=false;
    rows.forEach((r,i)=>{
      if(!Number.isFinite(r[key])){open=false;return;}
      path+=(open?'L':'M')+x(i).toFixed(2)+','+y(r[key]).toFixed(2);
      open=true;
    });
    return '<path d="'+path+'" fill="none" stroke="'+color+'" stroke-width="'+(width||1.4)+'" stroke-dasharray="'+(dash||'none')+'"/>';
  }).join('');
}
function chartKey(series) {
  return '<div class="gamma-legend">'+series.map(({label,color})=>'<span><i style="background:'+color+'"></i>'+label+'</span>').join('')+'</div>';
}
const trendSeries=[
  {key:'close',label:'Close',color:'var(--text)',width:2},
  {key:'e20',label:'20 EMA',color:'var(--red)'},
  {key:'e50',label:'50 EMA',color:'#9d8df0'},
  {key:'e200',label:'200 EMA',color:'var(--iv)'}
];
function explorerEdge(m) {
  const edge=m?.edge_match;
  if(!edge) return card('Historical odds','Historical Analogue Match','<p>Historical diagnostics are unavailable for this ticker.</p>');
  const stats=[['5D win rate',pct(edge.win_rate_5d)],['20D win rate',pct(edge.win_rate_20d)],['60D win rate',pct(edge.win_rate_60d)],
    ['Median 20D return',signed(edge.median_return_20d)],['Median max adverse excursion',signed(edge.max_adverse_excursion)]];
  return card('Historical odds','Historical Analogue Match','<p>'+safe(edge.rule_name)+' · '+contracts(edge.sample_size)+' historical matches</p>'
    +'<div class="vol-metrics">'+stats.map(([k,v])=>'<span>'+k+' <strong>'+v+'</strong></span>').join('')+'</div>');
}
function explorerTrend(all) {
  const limit={'1Y':252,'2Y':504,'5Y':1260,'All':all.length}[explorerSelection.horizon];
  const rows=remoteTicker.get(state.symbol+'/'+explorerSelection.horizon)?.series||all.slice(-limit), values=rows.flatMap(row=>trendSeries.map(s=>row[s.key])).filter(Number.isFinite);
  const x=i=>L+(R-L)*i/Math.max(1,rows.length-1);
  const {y,ticks}=scale(values), ddMin=Math.min(-1,...rows.map(row=>row.dd));
  const ddTop=211, ddBottom=268, ddY=v=>ddTop+(ddBottom-ddTop)*v/ddMin;
  const priceY=v=>T+(y(v)-T)*.69;
  const ddPath=line(rows.map(row=>row.dd),x,ddY);
  const drawdown='<line x1="'+L+'" x2="'+R+'" y1="'+ddTop+'" y2="'+ddTop+'" stroke="var(--grid-line)"/>'
    +'<path d="'+ddPath+'L'+x(rows.length-1)+','+ddTop+'L'+x(0)+','+ddTop+'Z" fill="var(--red)" fill-opacity=".14"/>'
    +'<path d="'+ddPath+'" fill="none" stroke="var(--red)" stroke-width="1.4"/>'
    +'<text x="'+(R+8)+'" y="'+(ddTop+4)+'">0%</text><text x="'+(R+8)+'" y="'+(ddBottom+4)+'">'+pct(ddMin)+'</text>'
    +'<text x="'+L+'" y="'+(ddTop-6)+'">ATH drawdown</text>';
  hoverSeries={size:rows.length,x,y:i=>priceY(rows[i].close),text:i=>{
    const row=rows[i];
    return row.date+' · '+money(row.close)+'<br>EMA 20 '+money(row.e20)+' · EMA 50 '+money(row.e50)
      +'<br>EMA 200 '+money(row.e200)+' · Drawdown '+signed(row.dd);
  }};
  return card('Price trend &amp; drawdown',state.symbol+' historical price',
    plotFrame('Price, moving averages and drawdown. Use the arrow keys to inspect a session.',grid(priceY,ticks,money,false),chartPaths(rows,trendSeries,x,priceY)+drawdown,axis(rows,x,r=>short(r.date)),true)
    +'<div class="chart-note"><span>'+contracts(rows.length)+' sessions</span>'+chartKey(trendSeries)+'</div>',
    ['1Y','2Y','5Y','All'].map(v=>'<button data-explorer-horizon="'+v+'" aria-pressed="'+(explorerSelection.horizon===v)+'">'+v+'</button>').join(''),'explorer-chart');
}
function tickerExplorer(date) {
  const symbol=state.symbol, all=priceStudies(symbol), last=all.at(-1), m=remoteTicker.get(symbol+'/'+explorerSelection.horizon)?.metrics||flowMetrics.get(symbol);
  const head=hero(symbol+' / '+safe(names[symbol]||'Daily OHLC'),'Single Ticker Explorer','Inspect price trends, drawdown, and historical setup odds.',date,true);
  date=m?.date||date;
  const options=Object.keys(data.prices).sort().map(t=>'<option value="'+t+'" '+(t===symbol?'selected':'')+'>'+t+'</option>').join('');
  const controls='<div class="vol-toolbar controls"><label>Underlying <select id="explorer-symbol">'+options+'</select></label>'
    +['Trend','OHLC detail'].map(v=>'<button data-explorer-view="'+v+'" aria-pressed="'+(explorerSelection.view===v)+'">'+v+'</button>').join('')+'</div>';
  const day=m?.pct_1d??daily(priceRows()), emaDistance=m?.dist_ema50??(last.close/last.e50-1)*100;
  const metrics=[['Current price',money(m?.close??last.close),'<span class="'+tone(day)+'">'+signed(day)+' · 1D</span>'],
    ['50-day EMA',money(m?.ema50??last.e50),'<span class="'+tone(emaDistance)+'">'+signed(emaDistance)+' vs. EMA</span>'],
    ['ATH drawdown',signed(m?.drawdown_ath??last.dd),contracts(m?.days_in_drawdown??last.days)+' sessions since peak'],
    ['14D RSI &amp; volume',m?m.rsi.toFixed(1):'—','RVOL '+formatFactor(m?.rvol)]];
  const summary='<section class="explorer-kpis" aria-label="Ticker metrics">'+metrics.map(([label,value,sub])=>'<div class="gamma-kpi"><small>'+label+'</small><strong>'+value+'</strong><small>'+sub+'</small></div>').join('')+'</section>';
  return head+controls+summary+'<div class="flow-panels in">'+(explorerSelection.view==='Trend'?explorerTrend(all):
    '<div class="grid ohlc-grid">'+priceChart()+'<div class="stack">'+priceStats()+gammaStats(latest())+'</div></div>')+explorerEdge(m)+'</div>';
}
function ratioRows(pair) {
  const [base,denominator]=pair.split('/');
  const other=new Map(data.prices[denominator].map(row=>[row[0],row[4]]));
  let e20,e50;
  const window=[];
  return data.prices[base].filter(row=>other.has(row[0])).map(([date,,,,close],i)=>{
    const value=close/other.get(date);
    e20=i?e20+2/21*(value-e20):value;
    e50=i?e50+2/51*(value-e50):value;
    window.push(value);
    if(window.length>50) window.shift();
    let envelope=NaN;
    if(window.length===50){
      const avg=window.reduce((a,b)=>a+b,0)/50;
      envelope=2*Math.sqrt(window.reduce((sum,val)=>sum+(val-avg)**2,0)/49);
    }
    return {date,close:value,e20,e50,upper:e50+envelope,lower:e50-envelope};
  }).slice(-500);
}
function ratioPage() {
  const pair=ratioSelection.pair, rows=remoteRatios.get(pair)||ratioRows(pair);
  const series=[
    {key:'close',label:pair,color:'var(--text)',width:2},
    {key:'e20',label:'20 EMA',color:'#7cd0d9'},
    {key:'e50',label:'50 EMA',color:'var(--iv)'},
    {key:'upper',label:'+2σ',color:'var(--red)',dash:'4 4'},
    {key:'lower',label:'−2σ',color:'var(--green)',dash:'4 4'}
  ];
  const head=hero('Relative strength','Synthetic Ratio Rotations','Track breadth, mega-cap concentration, and small-cap rotation.',rows.at(-1)?.date||data.prices.SPY.at(-1)[0],true);
  const picker='<label>Ratio <select id="rotation-pair">'+breadthPairs.map(([key,name])=>'<option value="'+key+'" '+(key===pair?'selected':'')+'>'+key+' · '+name+'</option>').join('')+'</select></label>';
  if(!rows.length) return head+card('Relative strength',pair,'<p>No matching sessions for this ratio.</p>',picker);
  const x=i=>L+(R-L)*i/Math.max(1,rows.length-1);
  const {y,ticks}=scale(rows.flatMap(row=>series.map(s=>row[s.key])).filter(Number.isFinite));
  const fmt=v=>Number.isFinite(v)?v.toFixed(4):'—';
  hoverSeries={size:rows.length,x,y:i=>y(rows[i].close),text:i=>{
    const row=rows[i];
    return row.date+' · '+pair+' '+fmt(row.close)+'<br>EMA 20 '+fmt(row.e20)+' · EMA 50 '+fmt(row.e50)
      +'<br>Lower '+fmt(row.lower)+' · Upper '+fmt(row.upper);
  }};
  const summary=breadthPairs.find(([key])=>key===pair);
  return head+'<div class="grid rotation-grid in">'+card('500-session view',pair+' ratio',
    plotFrame('Price ratio, moving averages and two-sigma bands. Use the arrow keys to inspect sessions.',grid(y,ticks,fmt,false),chartPaths(rows,series,x,y),axis(rows,x,row=>short(row.date)),true)
    +'<div class="chart-note"><span>'+contracts(rows.length)+' sessions</span>'+chartKey(series)+'</div>',picker,'rotation-chart')+breadthCard(summary)+'</div>';
}

function fitOverview() {
  const content = byId('content');
  if (route() !== 'overview') return;
  content.style.removeProperty('--overview-room');
  const page = document.documentElement;
  const naturalBottom = content.closest('.stage').getBoundingClientRect().bottom + window.scrollY;
  const available = page.clientHeight - Math.ceil(naturalBottom);
  // Three section gaps and three single-row grids. Extra wrap rows keep natural spacing.
  const sevenAcross = getComputedStyle(content.querySelector('.overview-quotes')).gridTemplateColumns.split(' ').length === 7;
  if (sevenAcross && available > 0) content.style.setProperty('--overview-room', Math.min(available, 96) + 'px');
}

const reducedMotion=()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function animateGammaMarkers(previous) {
  if(reducedMotion()) return;
  document.querySelectorAll('[data-gamma-marker]').forEach(marker=>{
    const from=previous.get(marker.dataset.gammaMarker);
    const to=Number(marker.getAttribute('x1'));
    if(Number.isFinite(from)&&Math.abs(from-to)>.1) marker.animate(
      [{transform:'translateX('+(from-to)+'px)'},{transform:'translateX(0px)'}],
      {duration:320,easing:'cubic-bezier(.2,.7,.2,1)'});
  });
}
function bindFlowEmphasis() {
  const chart=document.querySelector('.flow-normalized');
  if(!chart) return;
  const emphasis=ticker=>chart.querySelectorAll('[data-series-path]').forEach(path=>{
    path.classList.toggle('flow-emphasized',!!ticker&&path.dataset.seriesPath===ticker);
    path.classList.toggle('flow-dimmed',!!ticker&&path.dataset.seriesPath!==ticker);
  });
  chart.querySelectorAll('[data-flow-series][aria-pressed="true"]').forEach(button=>{
    button.addEventListener('pointerenter',()=>emphasis(button.dataset.flowSeries));
    button.addEventListener('pointerleave',()=>emphasis(''));
    button.addEventListener('focus',()=>emphasis(button.dataset.flowSeries));
    button.addEventListener('blur',()=>emphasis(''));
  });
}
const remoteBias=new Map(), remoteRatios=new Map(), remoteTicker=new Map(), remoteReturns=new Map();
const backendRequests=new Map();
const loadedHistories=new Set();
function dataKey() {
  const page=route();
  if(page==='overview'||page==='sector'||page==='mag7') return page;
  if(page==='bias') return page+'/'+biasSelection.scope;
  if(page==='ratios') return page+'/'+ratioSelection.pair;
  if(page==='charts') return page+'/'+state.symbol+'/'+explorerSelection.horizon+'/'+explorerSelection.view;
  if(page==='volatility') return page+'/'+state.symbol+'/'+volSelection.expiry;
  if(page==='gamma') return page+'/'+state.symbol;
}
async function readBackend(path) {
  const response=await fetch('/api/'+path,{signal:AbortSignal.timeout(70000)});
  if(!response.ok) throw new Error('API returned '+response.status+'.');
  const value=await response.json();
  if(!value||value.error) throw new Error(value?.error||'Empty response.');
  return value;
}
function applyMarketResponse(payload) {
  const market=payload.market_data;
  if(!market||!payload.benchmarks) throw new Error('Invalid market overview response.');
  const assets={...payload.benchmarks,...market.macro_assets,...market.mag7,...market.sectors};
  for(const [ticker,m] of Object.entries(assets)){
    if(!data.overview.assets[ticker]||!Number.isFinite(m?.close)) continue;
    data.overview.assets[ticker]={close:m.close,day:m.pct_1d,week:m.pct_5d,month:m.pct_20d,drawdown:m.drawdown_ath,
      status:m.tech_status||data.overview.assets[ticker].status};
    flowMetrics.set(ticker,m);
  }
  for(const [key,m] of Object.entries(market.ratios||{})){
    if(!Number.isFinite(m?.current)) continue;
    data.overview.ratios[key.replace('_','/')]={close:m.current,z:m.z_score_50d,week:m.pct_5d,state:m.actionable_state,action:m.action_text};
  }
  Object.assign(remoteSparks,payload.sparklines||{});
  if(expiryPattern.test(payload.last_market_session||'')) marketSession=payload.last_market_session;
}
const remoteSparks={};
let marketSession='';
function unpackTimeSeries(payload, fields) {
  if(!Array.isArray(payload.dates)) throw new Error('No dated series returned.');
  const rows=payload.dates.map((date,index)=>Object.fromEntries([['date',date],...Object.entries(fields).map(([key,field])=>[key,payload[field]?.[index]])]))
    .filter(row=>expiryPattern.test(row.date)&&Number.isFinite(row.close));
  if(!rows.length) throw new Error('No valid price rows returned.');
  return rows;
}
function backendOptionData(payload,symbol) {
  if(payload.ticker!==symbol||!Array.isArray(payload.term_chains)) throw new Error('No expiration chains returned.');
  const chains=payload.term_chains.filter(r=>r&&expiryPattern.test(r.Expiration)&&Number.isFinite(r.DTE)&&r.DTE>=0&&Number.isFinite(r.ATM_IV)&&r.ATM_IV>0).sort(termOrder);
  if(!chains.length) throw new Error('No valid expiration chains returned.');
  const smiles=new Map();
  if(payload.smile?.expiration) smiles.set(payload.smile.expiration,payload.smile);
  optionSnapshots.set(symbol,{...payload,chains,smiles});
}
function gammaSessionDate(payload,fallback) {
  const match=/^([A-Za-z]{3}) (\d{1,2}), (\d{4})/.exec(payload.market_trade_time||'');
  if(!match) return fallback;
  const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const month=months.indexOf(match[1])+1;
  return month?match[3]+'-'+String(month).padStart(2,'0')+'-'+match[2].padStart(2,'0'):fallback;
}
async function hydrateView(page,symbol,force) {
  const requestedSession=state.gammaDate;
  if(page==='overview'||page==='sector'||page==='mag7'){
    applyMarketResponse(await readBackend('market-data'));
    if(page==='mag7'){
      await Promise.allSettled(data.bias.groups.mag7.map(async ticker=>{
        const p=await readBackend('ticker/'+ticker+'?horizon=1%20Year');
        remoteReturns.set(ticker,unpackTimeSeries(p,{close:'close'}));
      }));
    }
  }
  if(page==='bias'){
    const scope=biasSelection.scope;
    const items=await readBackend('confluence?scope='+encodeURIComponent(scope));
    if(!Array.isArray(items)) throw new Error('No confluence results returned.');
    remoteBias.set(scope,items.filter(item=>item?.metrics?.edge_match&&Array.isArray(item.checklist)).map(item=>({
      ...item,checklist:item.checklist.map(layer=>({...layer,points:Number(layer.points)||0}))
    })));
  }
  if(page==='ratios'){
    const pair=ratioSelection.pair;
    const payload=await readBackend('ratio/'+pair.replace('/','_'));
    remoteRatios.set(pair,unpackTimeSeries(payload,{close:'close',e20:'ema20',e50:'ema50',upper:'upper2',lower:'lower2'}));
  }
  if(page==='charts'){
    const horizon=explorerSelection.horizon, ohlc=explorerSelection.view==='OHLC detail';
    const apiHorizon={'1Y':'1 Year','2Y':'2 Years','5Y':'5 Years',All:'All Historical Data'}[horizon];
    if(data.splitHistory&&!loadedHistories.has(symbol)&&['5Y','All'].includes(horizon)&&!ohlc){
      const archived=await fetch('/history/'+symbol+'.json').then(response=>response.ok?response.json():null).catch(()=>null);
      if(Array.isArray(archived)&&archived.length>=data.prices[symbol].length){
        data.prices[symbol]=archived;
        loadedHistories.add(symbol);
        explorerStudies.delete(symbol);
      }
    }
    const payload=await readBackend('ticker/'+symbol+'?horizon='+encodeURIComponent(apiHorizon));
    remoteTicker.set(symbol+'/'+horizon,{metrics:payload.metrics,series:unpackTimeSeries(payload,{close:'close',e20:'ema20',e50:'ema50',e200:'ema200',dd:'drawdown_ath'})});
    if(ohlc){
      const o=await readBackend('preview/history/'+symbol);
      if(!Array.isArray(o.rows)||o.rows.length<2) throw new Error('No OHLC history returned.');
      const previous=data.prices[symbol], first=o.rows[0][0];
      data.prices[symbol]=previous.filter(row=>row[0]<first).concat(o.rows);
      explorerStudies.delete(symbol);
    }
  }
  if(page==='gamma'){
    const p=await readBackend('gamma/'+symbol+(force?'?force=true':''));
    if(!Array.isArray(p.strikes)||!p.strikes.length||!(p.spot_price>0)) throw new Error('No strike profile returned.');
    const day=gammaSessionDate(p,data.prices[symbol].at(-1)[0]);
    currentGamma.set(symbol,{...p,date:day,strikes:Object.fromEntries(p.strikes.map(row=>[row.strike.toFixed(2),row]))});
    if(route()===page&&symbol===state.symbol&&(force?state.gammaDate===requestedSession:!state.gammaDate)) state.gammaDate='live';
  }
  if(page==='volatility'){
    const expiry=volSelection.expiry;
    if(force||!optionSnapshots.has(symbol)) backendOptionData(await readBackend('options/'+symbol),symbol);
    const current=optionSnapshots.get(symbol);
    if(expiry&&!current.smiles.has(expiry)&&current.chains.some(row=>row.Expiration===expiry)){
      const reply=await readBackend('options/'+symbol+'/smile?expiration='+encodeURIComponent(expiry)+'&spot='+encodeURIComponent(current.spot_price||0));
      if(reply.smile?.expiration!==expiry) throw new Error('No smile returned for this expiration.');
      current.smiles.set(expiry,reply.smile);
    }
  }
}
function updateBackendNotice(message) {

  const node=byId('api-status');
  if(node) node.textContent=message;
  const badge=byId('source-badge');
  if(badge) badge.textContent=message.startsWith('Original API data')?'ORIGINAL API':message.startsWith('Loading')?'CONNECTING':'SAVED SNAPSHOT';
}
async function loadPageData(force=false) {
  const key=dataKey();
  if(!key||location.protocol==='file:') return;
  if(!force&&backendRequests.has(key)){
    if(backendRequests.get(key)==='ready') updateBackendNotice('Original API data · saved history where applicable');
    if(backendRequests.get(key)==='failed') updateBackendNotice('Saved data shown · original API data unavailable');
    return;
  }
  if(backendRequests.get(key)==='pending') return;
  const page=route(),symbol=state.symbol;
  backendRequests.set(key,'pending');
  updateBackendNotice('Loading original API data…');
  try {
    await hydrateView(page,symbol,force);
    backendRequests.set(key,'ready');
    if(dataKey()===key){
      updateBackendNotice('Original API data · saved history where applicable');
      const focused=document.activeElement;
      const id=focused?.id;
      render();
      if(id) byId(id)?.focus({preventScroll:true});
    }
  } catch(error) {
    backendRequests.set(key,'failed');
    if(dataKey()===key){
      updateBackendNotice('Saved data shown · '+error.message);
      if(page==='charts'&&loadedHistories.has(symbol)) render();
    }
  }
}

let lastWorkspaceView;
function positionNavIndicator() {
  const nav=document.querySelector('.rail nav'), active=nav.querySelector('[aria-current="page"]');
  if(!active) return;
  nav.style.setProperty('--active-x',active.offsetLeft+'px');
  nav.style.setProperty('--active-y',active.offsetTop+'px');
  nav.style.setProperty('--active-width',active.offsetWidth+'px');
  nav.style.setProperty('--active-height',active.offsetHeight+'px');
  if(!nav.classList.contains('indicator-ready')) {
    requestAnimationFrame(()=>nav.classList.add('indicator-ready'));
  }
}
function workspaceValues() {
  return [...byId('content').querySelectorAll('.gamma-kpi > strong,.vol-kpi-value strong,.ohlc-grid .big,.ohlc-grid .stat strong')];
}
function highlightChanges(elements,before) {
  if(reducedMotion()) return;
  elements.forEach((element,i)=>{
    if(before[i]&&before[i]!==element.textContent&&element.textContent!=='—') element.animate(
      [{backgroundColor:'color-mix(in srgb, var(--amber) 26%, transparent)'},{backgroundColor:'transparent'}],
      {duration:640,easing:'ease-out'});
  });
}

function render() {
  const previousMarkers=new Map([...document.querySelectorAll('[data-gamma-marker]')].map(marker=>
    [marker.dataset.gammaMarker,Number(marker.getAttribute('x1'))]));
  const page=route(), rows=priceRows(), date=rows.at(-1)[0];
  const nextView={page,symbol:state.symbol,session:state.gammaDate};
  const entering=lastWorkspaceView?.page!==page;
  const changed=!entering&&(lastWorkspaceView.symbol!==nextView.symbol||lastWorkspaceView.session!==nextView.session);
  const priorValues=changed?workspaceValues().map(el=>el.textContent):[];
  lastWorkspaceView=nextView;
  hoverSeries=null;
  document.querySelectorAll('nav a').forEach(a=>a.hash==='#'+page?a.setAttribute('aria-current','page'):a.removeAttribute('aria-current'));
  positionNavIndicator();
  byId('crumb').textContent='WORKSPACE / '+titles[page].toUpperCase();
  let html;
  if(page==='overview') html=hero('Market overview','Market desk.','Price action, volatility and dealer positioning in one workspace.',(marketSession||data.prices.SPY.at(-1)[0])).replace('class="hero"','class="hero market-sheen"')+marketOverview();
  if(page==='charts') html=tickerExplorer(date);
  if(page==='ratios') html=ratioPage();
  if(page==='volatility') html=volatilityPage(date);
  if(page==='gamma') html=gammaPage(date);
  if(page==='bias') html=biasPage();
  if(page==='sector'||page==='mag7') html=flowPage(page==='mag7');
  byId('content').innerHTML=html;
  byId('content').classList.toggle('workspace-entering',entering);
  if(changed) highlightChanges(workspaceValues(),priorValues);
  byId('content').closest('.stage').classList.toggle('has-overview', page==='overview');
  fitOverview();
  const masthead = document.querySelector('.hero');
  masthead.addEventListener('pointermove', event => {
    if (event.pointerType !== 'mouse') return;
    const bounds = masthead.getBoundingClientRect();
    masthead.style.setProperty('--pointer-x', (event.clientX - bounds.left) + 'px');
    masthead.style.setProperty('--pointer-y', (event.clientY - bounds.top) + 'px');
  });
  bindChart();
  if(page==='gamma') animateGammaMarkers(previousMarkers);
  if(page==='mag7') bindFlowEmphasis();
  if(page==='volatility'){bindVolatility();bindVolStudyTips();}
  byId('explorer-symbol')?.addEventListener('change',e=>{choose(e.target.value);byId('explorer-symbol')?.focus({preventScroll:true});});
  byId('rotation-pair')?.addEventListener('change',e=>{ratioSelection.pair=e.target.value;render();byId('rotation-pair')?.focus({preventScroll:true});});
  byId('bias-asset')?.addEventListener('change',e=>{biasSelection.ticker=e.target.value;render();byId('bias-asset')?.focus({preventScroll:true});});
  byId('gamma-symbol')?.addEventListener('change',e=>{choose(e.target.value);byId('gamma-symbol')?.focus();});
  byId('gamma-recalculate')?.addEventListener('click',()=>loadPageData(true));
  byId('session')?.addEventListener('change',e=>{state.gammaDate=e.target.value;render();byId('session')?.focus();});
  void loadPageData();
}
const finder=byId('finder'), query=byId('query');
function choose(symbol) {
  if(!Object.hasOwn(data.prices,symbol)) return;
  if(state.symbol!==symbol) resetVolSelections();
  state.symbol=symbol;
  state.gammaDate='';
  if(finder.open) finder.close();
  render();
}
function results() {
  const q=query.value.trim().toUpperCase();
  const matches=Object.keys(data.prices).filter(t=>t.includes(q)||(names[t]||'').toUpperCase().includes(q)).sort((a,b)=>(b===q)-(a===q)||a.localeCompare(b));
  byId('results').innerHTML=matches.map(t=>'<button data-symbol="'+t+'">'+t+(names[t]?' · '+names[t]:'')+'</button>').join('')||'<p>No matching symbols.</p>';
}
function search() {if(finder.open)return;query.value='';results();finder.showModal();query.focus();}
document.addEventListener('click',e=>{
  const button=e.target.closest('button') ?? e.target.closest('.overview-back:not([inert])')?.querySelector('.flip-close');
  if(!button) return;
  if(button.dataset.explorerView || button.dataset.explorerHorizon) {
    const key=button.dataset.explorerView?'view':'horizon';
    const value=button.dataset.explorerView||button.dataset.explorerHorizon;
    explorerSelection[key]=value;
    render();
    document.querySelector('[data-explorer-'+key+'="'+value+'"]')?.focus({preventScroll:true});
    return;
  }
  if(button.dataset.flowSeries) {
    const ticker=button.dataset.flowSeries;
    const path=document.querySelector('[data-series-path="'+ticker+'"]');
    const toggle=()=>{
      if(!button.isConnected) return;
      if(mutedMag7.has(ticker)) mutedMag7.delete(ticker);
      else mutedMag7.add(ticker);
      render();
      document.querySelector('[data-flow-series="'+ticker+'"]').focus({preventScroll:true});
    };
    if(path&&!reducedMotion()&&document.querySelectorAll('[data-series-path]').length>1){
      if(path.getAnimations().some(animation=>animation.id==='flow-exit')) return;
      const animation=path.animate([{opacity:getComputedStyle(path).opacity},{opacity:0}],{duration:140,fill:'forwards'});
      animation.id='flow-exit';
      animation.finished.then(toggle,()=>{});
    } else toggle();
    return;
  }
  if(button.dataset.biasScope) {
    if(biasSelection.scope===button.dataset.biasScope) return;
    biasSelection.scope=button.dataset.biasScope;
    render();
    document.querySelector('.bias-bars')?.classList.add('bias-bars-entering');
    document.querySelector('[data-bias-scope="'+biasSelection.scope+'"]').focus({preventScroll:true});
    return;
  }
  if(button.dataset.inspectBias) {
    biasSelection.ticker=button.dataset.inspectBias;
    const inspector=button.closest('.bias-table');
    const scrollTop=document.querySelector('.bias-bars').scrollTop;
    render();
    document.querySelector('.bias-bars').scrollTop=scrollTop;
    if(inspector) byId('bias-asset')?.focus();
    else document.querySelector('[data-inspect-bias="'+biasSelection.ticker+'"]').focus({preventScroll:true});
    return;
  }
  if(button.dataset.volFilter) {
    const filter=button.dataset.volFilter;
    volSelection.filter=filter;
    render();
    document.querySelector('[data-vol-filter="'+filter+'"]').focus({preventScroll:true});
    return;
  }
  if(button.dataset.volInspect) {
    volSelection.expiry=button.dataset.volInspect;
    render();
    byId('vol-expiry').focus();
    return;
  }
  if(button.dataset.volPair) {
    [volSelection.front,volSelection.back]=button.dataset.volPair.split('/');
    render();
    byId('vol-front').focus({preventScroll:true});
    return;
  }
  if(button.dataset.flip) {
    const tile=button.closest('.overview-flip');
    const front=tile.querySelector('.overview-front');
    const back=tile.querySelector('.overview-back');
    const open=button.dataset.flip==='open';
    tile.classList.toggle('is-flipped',open);
    front.setAttribute('aria-expanded',String(open));
    front.inert=open;
    back.inert=!open;
    (open?back.querySelector('.flip-close'):front).focus({preventScroll:true});
    return;
  }
  if(button.dataset.chartSymbol) {
    if(state.symbol!==button.dataset.chartSymbol) resetVolSelections();
    state.symbol=button.dataset.chartSymbol;
    state.gammaDate='';
    location.hash='charts';
    return;
  }
  if(button.dataset.symbol) {
    choose(button.dataset.symbol);
  }
  const {key,value}=button.dataset;
  if(!key) return;
  state[key]=key==='ema'?!state.ema:['range','vol'].includes(key)?Number(value):value;
  render();
  document.querySelector('[data-key="'+key+'"][data-value="'+value+'"]')?.focus({preventScroll:true});
});
byId('open-search').addEventListener('click',search);
byId('close-search').addEventListener('click',()=>finder.close());
query.addEventListener('input',results);
query.addEventListener('keydown',e=>{if(e.key==='Enter')byId('results').querySelector('button')?.click();});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape') e.target.closest('.overview-flip.is-flipped')?.querySelector('.flip-close').click();
});
document.addEventListener('keydown',e=>{if(e.key==='/'&&!e.metaKey&&!e.ctrlKey&&!['INPUT','TEXTAREA'].includes(e.target.tagName)){e.preventDefault();search();}});
window.addEventListener('resize', fitOverview);
window.addEventListener('resize', positionNavIndicator);
document.fonts?.ready.then(positionNavIndicator);
window.addEventListener('hashchange',()=>{render();window.scrollTo(0,0);});
let ambientFrame=0;
const ambientMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
function ambientPrice(sample) {
  const t=sample*.085;
  return 290+96*Math.sin(t*.42)+49*Math.sin(t*1.13+.7)
    +27*Math.sin(t*2.19+Math.sin(t*.31))+13*Math.sin(t*4.35)+6*Math.sin(t*8.47);
}
function renderAmbientChart() {
  cancelAnimationFrame(ambientFrame);
  ambientFrame=0;
  const host=byId('market-replay');
  host.replaceChildren();
  if(document.documentElement.dataset.sweep!=='replay') return;
  host.innerHTML='<svg viewBox="0 0 1200 560" preserveAspectRatio="none" focusable="false">'
    +'<defs><linearGradient id="ambient-fill" x2="0%" y2="100%"><stop stop-color="var(--iv)" stop-opacity=".18"/><stop offset="1" stop-color="var(--iv)" stop-opacity="0"/></linearGradient></defs>'
    +'<path class="ambient-area" fill="url(#ambient-fill)"/><path class="replay-trace" fill="none"/>'
    +'<circle class="replay-pulse" r="2.8"/></svg>';
  const trace=host.querySelector('.replay-trace'), area=host.querySelector('.ambient-area'), tip=host.querySelector('.replay-pulse');
  const step=8, tipX=1180, points=Math.ceil(tipX/step)+2;
  let start, last=0;
  const draw=sample=>{
    const offset=(sample-Math.floor(sample))*step;
    const nearest=Math.floor(sample);
    let path='';
    for(let i=0;i<points;i++){
      const px=tipX-(points-i-2)*step-offset;
      if(px>=tipX) break;
      const index=nearest-points+i+2, level=ambientPrice(index);
      path+=(i?'L':'M')+px.toFixed(1)+','+level.toFixed(1);
    }
    const tipY=ambientPrice(sample);
    path+='L'+tipX+','+tipY.toFixed(1);
    trace.setAttribute('d',path);
    area.setAttribute('d',path+'L'+tipX+',560L-20,560Z');
    tip.setAttribute('cx',tipX);
    tip.setAttribute('cy',tipY);
  };
  draw(152);
  if(ambientMotion.matches) return;
  const advance=now=>{
    start??=now;
    if(now-last>=32){draw(152+(now-start)*.0028);last=now;}
    ambientFrame=requestAnimationFrame(advance);
  };
  ambientFrame=requestAnimationFrame(advance);
}
ambientMotion.addEventListener('change',renderAmbientChart);

renderAmbientChart();
const syncStatus=byId('sync-status'), syncButtons=[byId('sync-yfinance'),byId('sync-csv')];
try {
  syncStatus.textContent=sessionStorage.getItem('double-edge-sync-result')||'';
  sessionStorage.removeItem('double-edge-sync-result');
  const priorSymbol=sessionStorage.getItem('double-edge-sync-symbol');
  sessionStorage.removeItem('double-edge-sync-symbol');
  if(priorSymbol&&Object.hasOwn(data.prices,priorSymbol)) state.symbol=priorSymbol;
} catch {}
async function syncPreview(source) {
  if(location.protocol==='file:'){
    syncStatus.textContent='Open the local server preview to sync data.';
    return;
  }
  syncButtons.forEach(button=>button.disabled=true);
  syncStatus.textContent=source==='csv'?'Reading local CSV files…':'Requesting yfinance data…';
  try {
    const syncPath=location.hostname==='127.0.0.1'&&location.port==='4173'?'/preview/sync-':'/api/preview/sync-';
    const response=await fetch(syncPath+source,{method:'POST',headers:{'X-Preview-Sync':'1'}});
    const result=await response.json();
    syncStatus.textContent=result.message;
    if(response.ok){
      try {
        sessionStorage.setItem('double-edge-sync-result',result.message);
        sessionStorage.setItem('double-edge-sync-symbol',state.symbol);
      } catch {}
      window.location.reload();
      return;
    }
  } catch {
    syncStatus.textContent='The preview server could not complete the sync.';
  }
  syncButtons.forEach(button=>button.disabled=false);
}
syncButtons[0].addEventListener('click',()=>syncPreview('yfinance'));
syncButtons[1].addEventListener('click',()=>syncPreview('csv'));
render();










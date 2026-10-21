(() => {
  const svg = document.getElementById('chart');
  const wrap = document.getElementById('chart-wrap');
  const tooltip = document.getElementById('tooltip');
  const status = document.getElementById('status');
  const backButton = document.getElementById('back');
  const resetButton = document.getElementById('reset');
  const ns = 'http://www.w3.org/2000/svg';
  const palette = ['#081d58','#225ea8','#1d91c0','#41b6c4','#7fcdbb','#c7e9b4','#ffffcc'];
  let overview = null;
  let current = null;
  let history = [];
  let drawing = null;
  let geometry = null;
  let resizeTimer = null;

  function node(name, attributes = {}, text = null) {
    const element = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== null) element.textContent = text;
    return element;
  }

  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function formatTime(seconds, includeSeconds = false) {
    const whole = Math.round(seconds);
    const h = Math.floor(whole / 3600) % 24;
    const m = Math.floor(whole / 60) % 60;
    const s = whole % 60;
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}${includeSeconds ? `:${String(s).padStart(2,'0')}` : ''}`;
  }
  function formatDuration(seconds) {
    if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
    if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
    return `${(seconds / 3600).toFixed(2)} h`;
  }
  function formatFrequency(value) {
    return value >= 1000 ? `${(value/1000).toFixed(value % 1000 ? 1 : 0)} kHz` : `${Math.round(value)} Hz`;
  }
  function mixColor(first, second, amount) {
    const parse = value => [1,3,5].map(index => parseInt(value.slice(index,index+2),16));
    const a=parse(first), b=parse(second);
    const c=a.map((value,index)=>Math.round(value+(b[index]-value)*amount));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function color(value, domain) {
    const t=clamp((value-domain[0])/(domain[1]-domain[0] || 1),0,1);
    const scaled=t*(palette.length-1), index=Math.min(palette.length-2,Math.floor(scaled));
    return mixColor(palette[index],palette[index+1],scaled-index);
  }
  function quantileLevel(value, domain, levels=64) {
    return clamp(Math.floor((value-domain[0])/(domain[1]-domain[0] || 1)*levels),0,levels-1);
  }
  function binaryBin(edges, value) {
    let low=0, high=edges.length-1;
    while (low+1<high) { const mid=(low+high)>>1; if (edges[mid]<=value) low=mid; else high=mid; }
    return clamp(low,0,edges.length-2);
  }
  function makeScale(domainLow, domainHigh, rangeLow, rangeHigh, logarithmic=false) {
    const first=logarithmic?Math.log(domainLow):domainLow;
    const span=(logarithmic?Math.log(domainHigh):domainHigh)-first;
    const forward=value=>rangeLow+((logarithmic?Math.log(value):value)-first)/span*(rangeHigh-rangeLow);
    const inverse=pixel=>{
      const transformed=first+(pixel-rangeLow)/(rangeHigh-rangeLow)*span;
      return logarithmic?Math.exp(transformed):transformed;
    };
    return {forward,inverse};
  }

  function tickValues(low, high, count) {
    const step=(high-low)/(count-1);
    return Array.from({length:count},(_,index)=>low+index*step);
  }
  function logTicks(low, high) {
    const candidates=[20,30,50,100,200,300,500,1000,2000,3000,4000];
    return candidates.filter(value=>value>=low&&value<=high);
  }

  function updateReadout(data) {
    const t=data.time_edges_seconds, f=data.frequency_edges_hz, r=data.resolution;
    const seconds=t[t.length-1]-t[0];
    document.getElementById('meta').textContent=`${data.deployment} · ${data.date} UTC · 连续真实海洋录音`;
    document.getElementById('range').textContent=`${formatTime(t[0],seconds<600)}–${formatTime(t[t.length-1],seconds<600)} · ${formatFrequency(f[0])}–${formatFrequency(f[f.length-1])}`;
    document.getElementById('resolution').textContent=data.kind==='overview'
      ? '60 s 时间聚合 · 48 个对数频带（概览）'
      : `${formatDuration(r.time_step_seconds)} 步长 · ${formatDuration(r.stft_window_seconds)} 窗 · ${r.frequency_resolution_hz} Hz FFT 分辨率`;
    document.getElementById('cells').textContent=`${r.time_bins.toLocaleString()} × ${r.frequency_bins.toLocaleString()} SVG`;
    document.getElementById('legend-low').textContent=`${data.color_domain[0].toFixed(1)}`;
    document.getElementById('legend-high').textContent=`${data.color_domain[1].toFixed(1)}`;
    document.getElementById('unit').textContent=data.unit;
    backButton.disabled=history.length===0;
    resetButton.disabled=data===overview;
  }

  function render(data) {
    current=data;
    svg.replaceChildren();
    const width=Math.max(700,Math.round(wrap.clientWidth));
    const height=Math.max(480,Math.round(svg.clientHeight));
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const margin={left:76,right:24,top:20,bottom:58};
    const left=margin.left, right=width-margin.right, top=margin.top, bottom=height-margin.bottom;
    const timeEdges=data.time_edges_seconds, frequencyEdges=data.frequency_edges_hz;
    const x=makeScale(timeEdges[0],timeEdges[timeEdges.length-1],left,right,false);
    const y=makeScale(frequencyEdges[0],frequencyEdges[frequencyEdges.length-1],bottom,top,data.scale==='log');
    geometry={width,height,left,right,top,bottom,x,y};

    const spectral=node('g',{class:'spectral-layer','aria-hidden':'true'});
    const levels=64, paths=Array.from({length:levels},()=>[]);
    for (let i=0;i<data.values.length;i++) {
      const x0=x.forward(timeEdges[i]), x1=x.forward(timeEdges[i+1]);
      for (let j=0;j<data.values[i].length;j++) {
        const y0=y.forward(frequencyEdges[j+1]), y1=y.forward(frequencyEdges[j]);
        const level=quantileLevel(data.values[i][j],data.color_domain,levels);
        paths[level].push(`M${x0.toFixed(2)},${y0.toFixed(2)}H${x1.toFixed(2)}V${y1.toFixed(2)}H${x0.toFixed(2)}Z`);
      }
    }
    paths.forEach((commands,index)=>{
      if (!commands.length) return;
      spectral.appendChild(node('path',{d:commands.join(''),fill:color(data.color_domain[0]+(index+.5)/levels*(data.color_domain[1]-data.color_domain[0]),data.color_domain)}));
    });
    svg.appendChild(spectral);

    const axis=node('g',{class:'axis'});
    axis.appendChild(node('rect',{class:'frame',x:left,y:top,width:right-left,height:bottom-top,fill:'none'}));
    const span=timeEdges[timeEdges.length-1]-timeEdges[0];
    tickValues(timeEdges[0],timeEdges[timeEdges.length-1],width<900?5:9).forEach(value=>{
      const px=x.forward(value);
      axis.appendChild(node('line',{class:'minor',x1:px,x2:px,y1:top,y2:bottom}));
      axis.appendChild(node('text',{x:px,y:bottom+22,'text-anchor':'middle'},formatTime(value,span<600)));
    });
    const frequencyTicks=data.scale==='log'?logTicks(frequencyEdges[0],frequencyEdges[frequencyEdges.length-1]):tickValues(frequencyEdges[0],frequencyEdges[frequencyEdges.length-1],7);
    frequencyTicks.forEach(value=>{
      const py=y.forward(value);
      axis.appendChild(node('line',{class:'minor',x1:left,x2:right,y1:py,y2:py}));
      axis.appendChild(node('text',{x:left-9,y:py+4,'text-anchor':'end'},value>=1000?`${(value/1000).toFixed(1)}k`:Math.round(value)));
    });
    axis.appendChild(node('text',{class:'axis-title',x:(left+right)/2,y:height-13,'text-anchor':'middle'},`${data.date} UTC 时间`));
    axis.appendChild(node('text',{class:'axis-title',transform:`translate(18 ${(top+bottom)/2}) rotate(-90)`,'text-anchor':'middle'},'频率 (Hz)'));
    svg.appendChild(axis);

    const selection=node('rect',{class:'selection',x:0,y:0,width:0,height:0,visibility:'hidden'});
    svg.appendChild(selection);
    const interaction=node('rect',{class:'interaction',x:left,y:top,width:right-left,height:bottom-top,'aria-label':'拖动框选要重新计算的时间频率区域'});
    svg.appendChild(interaction);
    bindInteraction(interaction,selection,data);
    updateReadout(data);
  }

  function pointerPosition(event) {
    const rect=svg.getBoundingClientRect();
    return [(event.clientX-rect.left)*geometry.width/rect.width,(event.clientY-rect.top)*geometry.height/rect.height];
  }
  function bindInteraction(interaction, selection, data) {
    interaction.addEventListener('pointerdown',event=>{
      interaction.setPointerCapture(event.pointerId);
      const [x,y]=pointerPosition(event);
      drawing={x:clamp(x,geometry.left,geometry.right),y:clamp(y,geometry.top,geometry.bottom)};
      selection.setAttribute('visibility','visible');
      tooltip.style.display='none';
    });
    interaction.addEventListener('pointermove',event=>{
      const [rawX,rawY]=pointerPosition(event), px=clamp(rawX,geometry.left,geometry.right), py=clamp(rawY,geometry.top,geometry.bottom);
      if (drawing) {
        const x=Math.min(drawing.x,px), y=Math.min(drawing.y,py);
        selection.setAttribute('x',x); selection.setAttribute('y',y);
        selection.setAttribute('width',Math.abs(px-drawing.x)); selection.setAttribute('height',Math.abs(py-drawing.y));
        return;
      }
      const time=geometry.x.inverse(px), frequency=geometry.y.inverse(py);
      const i=binaryBin(data.time_edges_seconds,time), j=binaryBin(data.frequency_edges_hz,frequency);
      tooltip.textContent=`${formatTime(time,true)} UTC · ${formatFrequency(frequency)} · ${data.values[i][j].toFixed(1)} ${data.unit}`;
      tooltip.style.display='block';
      tooltip.style.left=`${Math.min(wrap.clientWidth-280,event.clientX-wrap.getBoundingClientRect().left+12)}px`;
      tooltip.style.top=`${Math.max(6,event.clientY-wrap.getBoundingClientRect().top-35)}px`;
    });
    interaction.addEventListener('pointerleave',()=>{if(!drawing)tooltip.style.display='none';});
    interaction.addEventListener('pointerup',async event=>{
      if (!drawing) return;
      const [rawX,rawY]=pointerPosition(event), px=clamp(rawX,geometry.left,geometry.right), py=clamp(rawY,geometry.top,geometry.bottom);
      const start=drawing; drawing=null;
      selection.setAttribute('visibility','hidden');
      if (Math.abs(px-start.x)<8 || Math.abs(py-start.y)<8) return;
      const timeLow=Math.min(geometry.x.inverse(start.x),geometry.x.inverse(px));
      const timeHigh=Math.max(geometry.x.inverse(start.x),geometry.x.inverse(px));
      const frequencyLow=Math.max(0,Math.min(geometry.y.inverse(start.y),geometry.y.inverse(py)));
      const frequencyHigh=Math.min(4000,Math.max(geometry.y.inverse(start.y),geometry.y.inverse(py)));
      await zoomTo(timeLow,timeHigh,frequencyLow,frequencyHigh);
    });
  }

  async function zoomTo(start,end,low,high) {
    status.textContent='从原始 FLAC 重新计算…';
    const params=new URLSearchParams({start:start.toFixed(4),end:end.toFixed(4),low:low.toFixed(2),high:high.toFixed(2)});
    try {
      const response=await fetch(`/api/detail?${params}`);
      const data=await response.json();
      if (!response.ok) throw new Error(data.error||'重新计算失败');
      history.push(current); render(data);
      status.textContent='已从原始音频重算';
    } catch (error) {
      status.textContent=`错误：${error.message}`;
    }
  }

  backButton.addEventListener('click',()=>{
    if (!history.length) return;
    render(history.pop()); status.textContent=current.kind==='overview'?'全天概览':'已恢复上一级';
  });
  resetButton.addEventListener('click',()=>{history=[];render(overview);status.textContent='全天概览';});
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>current&&render(current),120);});

  fetch('/api/overview').then(response=>response.json()).then(data=>{
    overview=data; history=[]; render(data); status.textContent='全天概览';
  }).catch(error=>{status.textContent=`载入失败：${error.message}`;});
})();

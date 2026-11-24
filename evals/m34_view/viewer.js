const $=id=>document.getElementById(id);
const status={AVAILABLE:'可选择',RESERVED:'已选择',IN_PROGRESS:'进行中',REACHED:'已抵达',RESOLVED:'已完成',BLOCKED:'受阻',STALE:'边界失配',PLANNED:'待前往',SCANNED:'已扫描',PARTIALLY_EXPLORED:'部分探索',UNEXPLORED:'未探索',OBSERVING:'观察中'};
let data,i=0,key;
function line(parent,text){const p=document.createElement('p');p.textContent=text;parent.append(p)}
function render(){
 const f=data.frames[i],regs=f.regions,ids=Object.keys(regs);
 if(!regs[key])key=ids[0];$('sel').replaceChildren(...ids.map(k=>new Option(k+' · '+(status[regs[k].status]||regs[k].status),k)));
 if(key)$('sel').value=key;
 const r=regs[key],fid=key?.replace(/^R/,'F'),fr=f.frontiers[fid];
 $('frame').textContent='帧 '+i+' / '+(data.frames.length-1);$('slider').value=i;$('prev').disabled=!i;$('next').disabled=i===data.frames.length-1;
 $('rgb').src='../rgb'+i+'.png';$('scene').textContent=data.scene;
 const svg=$('map');svg.replaceChildren();
 const shape=(tag,attrs)=>{const a=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))a.setAttribute(k,v);svg.append(a);return a};
 const [ox,oz,w,h]=f.window,s=Math.min(480/w,360/h),x=a=>(480-w*s)/2+(a-ox)*s,z=a=>(360-h*s)/2+(a-oz)*s;
 shape('image',{href:'../grid'+i+'.png',x:x(ox),y:z(oz),width:w*s,height:h*s,preserveAspectRatio:'none'});
 for(const [row,col,seen] of r?.mask||[])shape('rect',{x:x(data.origin[0]+col*data.resolution),y:z(data.origin[1]+row*data.resolution),width:s*data.resolution,height:s*data.resolution,fill:seen?'#0c9c6e':'#f29a22',opacity:.8});
 for(const [id,rec] of Object.entries(f.frontiers)){const [a,b]=rec.centroid_xy;const dot=shape('circle',{cx:x(a),cy:z(b),r:id===fid?6:3,fill:id===fid?'#b72f60':rec.status==='STALE'?'#999':'#403c68',stroke:'white'});dot.onclick=()=>{key=id.replace('F','R');render()}}
 for(const v of Object.values(r?.views||{})){const [row,col]=v.cell;const a=x(data.origin[0]+(col+.5)*data.resolution),b=z(data.origin[1]+(row+.5)*data.resolution);shape('polygon',{points:[a+','+(b-6),(a-5)+','+(b+5),(a+5)+','+(b+5)].join(' '),fill:'#ffe45b',stroke:'#523c06'})}
 shape('circle',{cx:x(f.xy[0]),cy:z(f.xy[1]),r:5,fill:'#137cc3',stroke:'white'});
 $('stats').textContent=fr?fid+' · '+status[fr.status]+'　选择 '+fr.selection_count+'　尝试 '+fr.attempt_count+'　抵达 '+fr.visit_count+'　失败 '+fr.failed_count:'暂无边界';
 $('metrics').replaceChildren();
 if(r){line($('metrics'),key+' · '+status[r.status]);line($('metrics'),'实际观测 '+r.observed+' / 可观测 '+r.eligible+' 格　'+(r.coverage==null?'暂无分母':(r.coverage*100).toFixed(1)+'%')+'　门槛 85%');line($('metrics'),'剩余 Frontier '+r.remaining+'　连续完整版本确认 '+r.checks+' / 2');line($('metrics'),'代表观察点 '+Object.values(r.views).map(v=>status[v.status]||v.status).join('、'))}
 const rows=($('past').checked?data.frames.slice(0,i+1).flatMap(g=>g.events):f.events).filter(e=>JSON.stringify(e).toLowerCase().includes($('q').value.toLowerCase()));
 $('log').replaceChildren();for(const e of rows.slice().reverse())line($('log'),'#'+e.sequence+' · 帧 '+e.timestamp+' · '+e.event_type+'　'+[e.frontier_id,e.object_id,e.place_id,e.details?.region_id,e.details?.category,e.details?.after].filter(Boolean).join(' / '));
 if(!rows.length)line($('log'),'无匹配事件');
}
Promise.all(['frames.json','controlled.json'].map(k=>fetch(k).then(r=>{if(!r.ok)throw Error(r.status);return r.json()}))).then(([d,rows])=>{
 data=d;$('slider').max=d.frames.length-1;$('slider').oninput=e=>{i=+e.target.value;render()};$('prev').onclick=()=>{i--;render()};$('next').onclick=()=>{i++;render()};$('q').oninput=$('past').onchange=render;$('sel').onchange=e=>{key=e.target.value;render()};
 rows.forEach((r,j)=>{const tr=document.createElement('tr');for(const v of [['初次扫描','抵达观察点','完成扫描','首次确认','重复同版本','第二次确认','边界重新出现'][j],r.observed,r.remaining,r.checks,status[r.status]]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('demo').append(tr)});render()
}).catch(e=>$('frame').textContent='加载失败 '+e.message)

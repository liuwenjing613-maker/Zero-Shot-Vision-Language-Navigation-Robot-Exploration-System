'use strict';
const $=x=>document.getElementById(x), names={start:'起点',forward:'前进',back:'后退',left:'左转',right:'右转',observe:'观察',frontier:'选定 Frontier'}, phases={queued:'排队',working:'推理中',ready:'已入库',error:'需重试'};
let sid=localStorage.getItem('m1-live-session'),state=null,selected=null,busy=false,polling=false,epoch=0,rev=0,refInfo={},lastHistory='',lastOptions='';
const images=new Map();
function img(url){if(!images.has(url)){const promise=new Promise((ok,no)=>{const im=new Image();im.onload=()=>ok(im);im.onerror=no;im.src=url});images.set(url,promise);promise.catch(()=>images.delete(url));if(images.size>80)images.delete(images.keys().next().value)}return images.get(url)}
const base=()=>'/api/live/'+sid, asset=(n,name)=>base()+'/'+n+'/'+name;
function say(s,bad=false){$('connection').textContent=s;$('connection').classList.toggle('error',bad)}
async function json(url,options){const r=await fetch(url,options),d=await r.json();if(!r.ok)throw Error(d.error||String(r.status));return d}
function controls(){document.querySelectorAll('[data-go],#choose').forEach(b=>b.disabled=busy||!state||selected!==null);$('choose').disabled||=!$('frontier').value;$('start').disabled=busy;$('retry').disabled=busy}
async function act(action){if(busy)return;const creating=action==='start';if(!creating&&!state)return;const id=creating?crypto.randomUUID().replaceAll('-',''):sid;busy=true;epoch++;controls();say(creating?'正在加载 MP3D 场景…':'正在执行…');try{
 const d=await json('/api/live/control',{method:'POST',headers:{'Content-Type':'application/json','X-M1-Lab':'run'},body:JSON.stringify({session:id,action,expected:state?.frame?.step??-1,preset:$('preset').value,frontier:$('frontier').value})});
 sid=d.config.session;localStorage.setItem('m1-live-session',sid);state=d;selected=null;say('已连接 · '+sid.slice(0,8));await render()
 }catch(e){say(e.message,true)}finally{busy=false;controls()}}
async function poll(){if(!sid||busy||polling)return;const token=epoch;polling=true;try{const d=await json(base());if(token!==epoch||busy)return;
 let changed=!state||state.frame.step!==d.frame.step||d.history.map(x=>x.status).join()!==state.history.map(x=>x.status).join();state=d;
 if(d.config.session!==sid){sid=d.config.session;localStorage.setItem('m1-live-session',sid);changed=true} if(changed){say('已连接 · '+sid.slice(0,8));await render()}else progress()
 }catch(e){if(!state)say(e.message,true)}finally{polling=false;controls()}}
function progress(){if(!state)return;const n=state.frame.step,c=state.committed,w=state.history.find(x=>x.status==='working'),err=state.history.find(x=>x.status==='error');
 $('sync').textContent='机器人第 '+n+' 帧　·　语义记忆'+(c<0?'等待首帧':'已到第 '+c+' 帧')+'　·　待处理 '+(n-c)+' 帧'+(w?'　·　第 '+w.step+' 帧已运行 '+Math.max(0,Math.floor(Date.now()/1000-w.started))+' 秒':'')+(err?'　·　'+err.error:'');$('retry').hidden=!err}
function row(parent,text,cls='row'){let d=document.createElement('div');d.className=cls;d.textContent=text;parent.append(d);return d}
function sem(v,n,cam){const p=$('perception');p.replaceChildren();const cv=$('boxes');cv.width=480;cv.height=360;const ctx=cv.getContext('2d');
 if(!v){row(p,'等待该帧完成感知。');return}
 const result=v.result,entities=[...result.target_entities,...result.context_entities],all=[...entities,...result.visual_cues];
 if(n!==cam){const mini=document.createElement('img');mini.src=asset(n,'view.jpg');mini.className='mini';mini.alt='感知结果对应的相机帧';p.append(mini)}
 row(p,'第 '+n+' 帧 · '+(v.source==='identical_rgb'?'同会话相同 RGB 复用':'请求 '+(result.latency_ms/1000).toFixed(1)+' 秒'+(v.timing?'，模型计算 '+((v.timing.prompt_eval_duration+v.timing.eval_duration)/1e9).toFixed(1)+' 秒':''))+' · 目标状态 '+result.target_status);
 row(p,'目标：'+(result.target_entities.map(e=>e.category).join('、')||'无')+'　环境实体：'+(result.context_entities.map(e=>e.category).join('、')||'无'),'small');
 row(p,'导航线索：'+(result.visual_cues.map(e=>e.cue_type+' / '+e.relative_direction).join('，')||'无'),'small');
 if(n===cam)for(const e of all){const [a,b,c,d]=e.bbox.map((x,i)=>x*(i%2?360:480)/1000);ctx.strokeStyle=e.cue_type?'#ffac40':'#5ae7d1';ctx.lineWidth=2;ctx.strokeRect(a,b,c-a,d-b);ctx.fillStyle='#15333ce8';ctx.fillRect(a,Math.max(0,b-14),Math.min(125,8*(e.category||e.cue_type).length),14);ctx.fillStyle='white';ctx.font='12px sans-serif';ctx.fillText(e.category||e.cue_type,a+1,Math.max(11,b-3))}
}
async function draw(frame,mem,token){const canvas=$('map'),g=canvas.getContext('2d'),mode=$('basemap').value;let picture;let ref=refInfo[state.config.preset];try{picture=await img(mode==='grid'?asset(frame.step,'map.png'):'/live/'+state.config.preset+'.jpg')}catch{picture=null}if(token!==rev)return;
 let [ox,oz,w,h]=frame.window;const scale=Math.min((canvas.width-24)/w,(canvas.height-24)/h),dx=(canvas.width-w*scale)/2,dy=(canvas.height-h*scale)/2;
 const pt=p=>[dx+(p[0]-ox)*scale,dy+(p[1]-oz)*scale];
 g.fillStyle='#a5aeb1';g.fillRect(0,0,canvas.width,canvas.height);g.imageSmoothingEnabled=mode==='ref';
 if(picture){if(mode==='grid')g.drawImage(picture,dx,dy,w*scale,h*scale);else if(ref){const [x,y]=pt(ref);g.drawImage(picture,x,y,ref[2]*scale,ref[3]*scale)}}
 function line(ps,color,width){if(!ps.length)return;g.beginPath();ps.map(pt).forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));g.lineWidth=width;g.strokeStyle=color;g.stroke()}
 function dot(p,color,r,label=''){const [x,y]=pt(p);g.fillStyle=color;g.beginPath();g.arc(x,y,r,0,2*Math.PI);g.fill();if(label){g.font='bold 13px sans-serif';g.fillText(label,x+6,y-5)}}
 line(state.trajectory.slice(0,frame.step+1),'#336ba5',2);
 if(mem){mem.portals.forEach(q=>line([q.a,q.b],['VALIDATED','CROSSED'].includes(q.status)?'#e98418':'#976f47',4));mem.places.forEach(p=>dot([p.x,p.y],'#097851',5,p.place_id))}
 frame.options.forEach(p=>dot(p.xy,'#92539b',3));if(frame.frontier)dot(frame.frontier.xy,'#823597',6,'F');
 const [x,y]=pt(frame.xy);g.save();g.translate(x,y);g.rotate(frame.heading*Math.PI/180);g.beginPath();g.moveTo(11,0);g.lineTo(-7,-6);g.lineTo(-5,0);g.lineTo(-7,6);g.closePath();g.fillStyle='#0b4e95';g.fill();g.restore();
}
async function render(){if(!state?.frame)return;const token=++rev,n=selected??state.frame.step,latest=selected===null,f=latest?state.frame:await json(asset(n,'step.json'));if(token!==rev)return;
 const done=state.history[n]?.status==='ready',m=latest?state.memory:done?await json(asset(n,'memory.json')):null,v=latest?state.perception:done?await json(asset(n,'vlm.json')):null;if(token!==rev)return;
 $('rgb').hidden=false;$('empty').hidden=true;$('rgb').src=asset(n,'view.jpg');$('viewid').textContent=(latest?'实时':'回看')+' · 第 '+n+' 帧';
 $('pose').textContent='X '+f.xy[0].toFixed(2)+' m　Z '+f.xy[1].toFixed(2)+' m　朝向 '+f.heading.toFixed(0)+'°　'+(f.floor_block?'已到本层边界':f.collision?'碰撞，本步未前进':'本步移动 '+f.travel_m.toFixed(2)+' m')+'　空闲 '+f.grid_cells.free+' 格 / 占据 '+f.grid_cells.occupied+' 格';
 const key=sid+'-'+n;if(lastOptions!==key){lastOptions=key;$('frontier').replaceChildren(...f.options.map(p=>new Option(p.id+' · '+p.distance+' m',p.id)));if(!f.options.length)$('frontier').add(new Option('暂无可达候选',''))}
 $('target').textContent=f.frontier?'选定 '+f.frontier.id+' · 距离 '+Math.hypot(f.xy[0]-f.frontier.xy[0],f.xy[1]-f.frontier.xy[1]).toFixed(2)+' 米；抵达后按原地观察。':'手动走到选定点 0.35 米内，再按原地观察。';
 progress();sem(v,m?.step,n);$('commit').textContent=m?'已提交 · 第 '+m.step+' 帧':'等待语义提交';$('graph').textContent=m?'当前区域 '+m.area+' · 最近拓扑节点 '+m.place+' · '+m.places.length+' 个节点 / '+m.edges.length+' 条有向边':'扫描已单独保存，可继续驾驶。';
 const nodes=$('places');nodes.replaceChildren();if(m){m.places.forEach(p=>row(nodes,p.place_id+'　'+p.node_types.join(' / ')+'　访问 '+p.visit_count+' 次　'+p.observation_ids.length+' 帧　'+p.nearby_object_ids.length+' 条实体观测'));m.edges.forEach(e=>row(nodes,e.source_id+' → '+e.target_id+'　'+e.path_cost.toFixed(2)+' m','small'))}
 const historyKey=sid+n+JSON.stringify(state.history.map(h=>h.status));if(historyKey!==lastHistory){lastHistory=historyKey;const h=$('history');h.replaceChildren();for(const a of [...state.history].reverse()){let b=document.createElement('button');b.textContent='#'+a.step+' '+names[a.action]+' · '+phases[a.status]+(a.events.length?' · '+a.events.map(x=>x.reason).join('/'):'');if(a.step===n)b.className='picked';b.onclick=()=>{selected=a.step;render()};h.append(b)}}
 $('export').href=base()+'/export.zip';$('export').hidden=false;controls();await draw(f,m,token)}
$('start').onclick=()=>act('start');$('retry').onclick=()=>act('retry');$('choose').onclick=()=>act('frontier');$('live').onclick=()=>{selected=null;render()};$('basemap').onchange=render;
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>act(b.dataset.go));document.addEventListener('keydown',e=>{if(e.repeat||busy||!state||selected!==null||e.ctrlKey||e.metaKey||e.altKey||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;const a={KeyW:'forward',KeyS:'back',KeyA:'left',KeyD:'right',Space:'observe'}[e.code];if(a){e.preventDefault();act(a)}});
fetch('/live/reference.json').then(r=>r.json()).then(d=>{refInfo=d;if(state)render()}).catch(()=>{});controls();poll();setInterval(poll,1700);
async function modelHealth(){try{const m=await json("/api/live/model");$("model").textContent=m.model+" · "+(m.resident?"已预热，GPU "+(m.vram_bytes/2**30).toFixed(1)+" GiB":"等待模型驻留")}catch{$("model").textContent="qwen3.8:27b · 服务连接中"}}modelHealth();setInterval(modelHealth,15000);

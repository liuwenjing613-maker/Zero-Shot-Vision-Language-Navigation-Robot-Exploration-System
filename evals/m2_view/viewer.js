const $ = id => document.getElementById(id);
const el = (tag, text, cls) => {const node=document.createElement(tag); if(text!=null)node.textContent=text; if(cls)node.className=cls; return node};
const statusName={NEW:'新建物体',MATCHED:'关联旧物体',UNLOCALIZED:'仅保留观测',AMBIGUOUS:'关联有歧义',STABLE:'稳定',TRACKED:'已跟踪',PROVISIONAL:'暂定',SUPPORTED:'有一致支持',CONFLICTED:'证据冲突'};
const actionName={forward:'前进',backward:'后退',left:'左转',right:'右转',start:'起点',observe:'观察'};
const position=p=>p?p.map(v=>v.toFixed(2)).join(', ')+' m':'无可靠位置';
const scores=v=>Object.entries(v||{}).sort((a,b)=>b[1]-a[1]).map(([k,s])=>k+' '+s.toFixed(2)).join(' · ')||'无';
let data, i=0, oid=null, focus=null, timer=null;
function selectObs(id,o){focus=id;oid=o.object_id;render()}
function stop(){clearInterval(timer);timer=null;$('play').textContent='播放'}
function go(n){i=Math.max(0,Math.min(n,data.frames.length-1));focus=null;render()}
function mapFrame(f){
 const svg=$('map');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
 function shape(tag,attrs){const a=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))a.setAttribute(k,v);svg.append(a);return a}
 const [x,z,w,h]=f.window, s=Math.min(480/w,360/h), px=v=>(480-w*s)/2+(v-x)*s, py=v=>(360-h*s)/2+(v-z)*s;
 const ref=$('base').value==='ref', b=ref?data.reference:f.window;
 shape('image',{href:ref?'reference.jpg':'grid'+i+'.png',x:px(b[0]),y:py(b[1]),width:b[2]*s,height:b[3]*s,preserveAspectRatio:'none'});
 shape('polyline',{points:data.frames.slice(0,i+1).map(v=>[px(v.xy[0]),py(v.xy[1])].join(',')).join(' '),stroke:'#24639d','stroke-width':2,fill:'none'});
 for(const [id,obj] of Object.entries(f.memory.objects)){
 const [a,b]=obj.map_position_xz, selected=id===oid;
 const dot=shape('circle',{cx:px(a),cy:py(b),r:selected?6:4,fill:selected?'#ef880e':'#147d5c',stroke:'white','stroke-width':1.4});
 dot.onclick=()=>{oid=id;focus=null;render()};dot.style.cursor='pointer';
 if(selected){const t=shape('text',{x:px(a)+8,y:py(b)-6,'font-size':12,stroke:'white','stroke-width':3,'paint-order':'stroke'});t.textContent=id}
 }
 shape('polygon',{points:'10,0 -7,-6 -4,0 -7,6',fill:'#13558c',stroke:'white',transform:'translate('+px(f.xy[0])+' '+py(f.xy[1])+') rotate('+f.heading+')'});
}
function render(){
 const f=data.frames[i], mem=f.memory, rows=Object.entries(mem.observations);
 if(!mem.objects[oid] && !focus)oid=rows.find(([,o])=>o.object_id)?.[1].object_id||Object.keys(mem.objects)[0]||null;
 const selected=focus||rows.find(([,o])=>o.object_id===oid)?.[0];
 $('slider').value=i;$('frame').textContent='帧 '+i+' / '+(data.frames.length-1);
 $('action').textContent=actionName[f.action]||f.action;$('prev').disabled=i===0;$('next').disabled=i===data.frames.length-1;
 $('rgb').src='rgb'+i+'.png';$('boxes').replaceChildren();$('events').replaceChildren();
 if(!rows.length)$('events').append(el('p','本帧没有物体观测，历史记录保留。'));
 rows.forEach(([id,o])=>{
 const [x,y,r,b]=o.bbox, box=el('button',null,'box'+(id===selected?' selected':''));
 Object.assign(box.style,{left:x/10+'%',top:y/10+'%',width:(r-x)/10+'%',height:(b-y)/10+'%'});
 box.append(el('span',o.local_id+' '+o.category_raw));box.onclick=()=>selectObs(id,o);$('boxes').append(box);
 const row=el('button',null,'event'+(id===selected?' chosen':''));
 row.append(el('b',o.local_id+' · '+o.category_raw+' → '+(o.object_id||'未分配 ID')+'　'+statusName[o.association_status]));
 row.append(el('span','深度有效 '+Math.round(100*o.depth_valid_ratio)+'%　观测位置 '+position(o.map_position_xz)));
 if(o.object_id)row.append(el('span',o.independent?'独立视角：更新类别和位置证据':'重复视角：追加观测历史，融合权重不变'));
 if(o.candidates.length)row.append(el('span','候选关联分　'+o.candidates.map(([k,v])=>k+' '+v.toFixed(2)).join(' / ')));
 if(id===selected)row.append(el('span','Qwen 原始描述　'+o.visual_evidence));
 row.onclick=()=>selectObs(id,o);$('events').append(row);
 });
 const list=$('objects');list.replaceChildren(new Option('选择物体 · 共 '+Object.keys(mem.objects).length+' 个',''));
 for(const [id,obj] of Object.entries(mem.objects))list.add(new Option(id+' · '+Object.keys(obj.category_scores).sort((a,b)=>obj.category_scores[b]-obj.category_scores[a])[0],id));
 list.value=oid||'';const panel=$('detail');panel.replaceChildren();const obj=mem.objects[oid];
 if(obj){
 const old=data.frames[i-1]?.memory.objects[oid];
 panel.append(el('b',oid+'　身份 '+statusName[obj.track_status]+'　语义 '+(obj.semantic_status==='AMBIGUOUS'?'类别有歧义':statusName[obj.semantic_status]||obj.semantic_status)));
 panel.append(el('p','累计观测 '+(old?.observation_ids.length||0)+' → '+obj.observation_ids.length+'　独立视角 '+(old?.independent_ids.length||0)+' → '+obj.independent_ids.length,'facts'));
 panel.append(el('p','类别证据（累计分）　'+scores(old?.category_scores)+' → '+scores(obj.category_scores)));
 panel.append(el('p','融合位置 X/Z　'+position(old?.map_position_xz)+' → '+position(obj.map_position_xz)+'　σ '+obj.position_sigma_m.toFixed(2)+' m'));
 for(const [k,v] of Object.entries(obj.attribute_scores))panel.append(el('p',k+'　'+scores(v)));
 panel.append(el('b','观测历史 · 点击回到原帧'));const hist=el('div',null,'history');
 for(const frame of data.frames.slice(0,i+1))for(const [key,o] of Object.entries(frame.memory.observations)){
 if(o.object_id!==oid)continue;const btn=el('button','帧 '+frame.step+' · '+o.category_raw);btn.onclick=()=>{stop();i=frame.step;focus=key;render()};hist.append(btn)}
 panel.append(hist);
 }else panel.append(el('p','当前选择没有全局物体 ID。可在下拉框查看已有记忆。'));
 mapFrame(f);
}
$('prev').onclick=()=>{stop();go(i-1)};$('next').onclick=()=>{stop();go(i+1)};
$('slider').oninput=e=>{stop();go(+e.target.value)};
$('objects').onchange=e=>{oid=e.target.value;focus=null;render()};$('base').onchange=()=>render();
$('play').onclick=()=>{if(!data)return;if(timer){stop();return}if(i===data.frames.length-1)go(0);$('play').textContent='暂停';timer=setInterval(()=>{go(i+1);if(i===data.frames.length-1)stop()},1500)};
fetch('data.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{data=d;$('slider').max=d.frames.length-1;$('meta').textContent=d.scene+' · '+d.model+' · '+d.frames.length+' 帧实录';render()}).catch(e=>{$('meta').textContent='回放加载失败 '+e.message});

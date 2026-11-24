"use strict";
const db=JSON.parse(document.getElementById("records").textContent);
const $=id=>document.getElementById(id);
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num=(x,k=2)=>Number.isFinite(x)?x.toFixed(k):"—";
const names={EXPLORE_FRONTIER:"探索前沿",OBSERVE_TARGET:"观察候选",APPROACH_TARGET:"接近目标",VERIFY_AT_ARRIVAL:"到达复核",SCAN:"环视扫描",STOP:"停止搜索",RECOVER:"执行恢复",RELOCALIZE:"重新定位",SUCCEEDED:"执行成功",SUCCESS:"回合成功",FRAME_LIMIT:"达到帧数上限",SEARCH_EXHAUSTED:"搜索预算耗尽",ARRIVAL_UNVERIFIED:"到达未确认",VISIBLE:"目标可见",NOT_VISIBLE:"目标不可见",UNCERTAIN:"不确定",AMBIGUOUS:"目标有歧义",VERIFIED:"模型确认",REJECTED:"模型否定",AVAILABLE:"可选",BLOCKED:"阻塞",REACHED:"已到达",MATCHED:"跨帧关联",NEW:"新对象",PROVISIONAL:"暂定",CONFIRMED:"已确认",PARTIALLY_EXPLORED:"部分探索",UNEXPLORED:"未探索"};
const cats={chair:"椅子",table:"桌子",wall:"墙",cabinet:"柜子",door:"门",doorway:"门洞",opening:"开口",corridor:"走廊",picture_frame:"画框",door_panel:"门板",floor:"地面",window:"窗",sofa:"沙发",lamp:"灯",coffee_machine:"咖啡机",staircase:"楼梯",picture:"挂画",ceiling:"天花板",misc:"其他物体",plant:"植物",counter:"台面",bookshelf:"书架"};
const tr=x=>names[x]||x||"—", cat=x=>cats[x]||x||"—";
const c={goal:"#ffb05c",context:"#78d7df",cue:"#aea2ff",frontier:"#38d5be",track:"#e8fcff",objects:"#ffc166",places:"#b7adff"};
const tok=x=>esc(x); const one=(s,cl="")=>'<span class="'+cl+'">'+esc(s)+'</span>';
const jump=new URLSearchParams(location.hash.slice(1));
let ei=Math.max(0,db.findIndex(e=>e.id===jump.get("run"))),fi=Math.max(0,Math.min(db[ei].frames.length-1,(Number(jump.get("frame"))||1)-1));
let mode="rgb",tab="flow",timer=0,sel="",frameTimer=0;
const on={goal:true,context:true,cue:true,frontier:true,track:true,objects:true,places:false};
const e=()=>db[ei], f=()=>e().frames[fi], action=r=>r.stop_status?"SUCCESS":r.action||r.initial_action;
const shortAct={EXPLORE_FRONTIER:"探索",OBSERVE_TARGET:"观察",APPROACH_TARGET:"接近",SUCCESS:"复核",STOP:"停止",SCAN:"环视",VERIFY_AT_ARRIVAL:"复核"};
const acColor=x=>({EXPLORE_FRONTIER:"#1ba394",OBSERVE_TARGET:"#9b80d9",APPROACH_TARGET:"#e58c45",SUCCESS:"#4d6ae1",STOP:"#8b939d"}[x]||"#86a6b3");
const glyph={camera:'<rect x="3" y="6" width="14" height="12" rx="3"/><path d="m17 10 4-3v10l-4-3z"/><circle cx="10" cy="12" r="2.4"/>',map:'<path d="m2 6 7-3 6 3 7-3v15l-7 3-6-3-7 3zM9 3v15m6-12v15"/>',track:'<circle cx="5" cy="16" r="2.5"/><circle cx="18" cy="7" r="2.5"/><path d="M7.5 16h3a3 3 0 0 0 3-3v-3a3 3 0 0 1 3-3"/>'};
for(const el of document.querySelectorAll("[data-icon]"))el.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round">'+glyph[el.dataset.icon]+'</svg>';
$("episodes").innerHTML=db.map((d,i)=>'<option value="'+i+'">'+esc(d.label+' · '+d.goal+'（'+d.frames.length+' 帧）')+'</option>').join("");
function layers(id,keys,labels){$(id).innerHTML=keys.map((k,i)=>'<button class="layer" data-layer="'+k+'" aria-pressed="'+on[k]+'"><i class="sw" style="--c:'+c[k]+'"></i>'+labels[i]+'</button>').join("")}
function infoHeader(){
 const ep=e(),rows=ep.frames.map(x=>x.row),last=rows[rows.length-1];
 $("episodes").value=ei;$("goal").textContent=ep.goal;$("scenario").textContent=ep.setting+" · MP3D / "+ep.scene;
 $("goalInstruction").textContent=ep.instruction.instruction||"";
 $("term").innerHTML='<span class="badge '+(ep.terminal==="SUCCESS"?"":"warn")+'">'+esc(tr(ep.terminal))+'</span>';
 const stats=[[rows.length,"观测帧"],[rows.filter(r=>r.action==="EXPLORE_FRONTIER").length,"前沿导航"],[ep.frames[ep.frames.length-1].objects.length,"累计记忆对象"],[last.distance_m!=null?num(last.distance_m)+" m":"—","停止时目标估距"]];
 $("stats").innerHTML=stats.map(([v,k])=>'<div class="metric"><b>'+esc(v)+'</b><small>'+k+'</small></div>').join("");
}
function camera(){
 const fr=f(),width=640,height=480,labels=[];
 const step=poseStep(),src=step?movie.assets[step.img][mode==="rgb"?0:1]:fr[mode]; let output='<image href="'+src+'" width="640" height="480"/>';
 for(const b of (step?[]:[...fr.boxes]).sort((a,b)=>(a.kind==="goal")-(b.kind==="goal"))){
  if(!on[b.kind])continue;
  const [x,y,r,h]=b.bbox.map((v,i)=>v*(i%2?height:width)/1000);
  const id=b.globalId||b.local_id,color=c[b.kind],name=id+" "+cat(b.category||b.cue_type);
  const w=Math.min(225,8+Array.from(name).reduce((sum,ch)=>sum+(ch.charCodeAt(0)>127?14:8),0));
  let tx=Math.min(width-w,Math.max(0,x)),ty=(y>20?y-19:y+1);
  for(let j=0;j<12&&labels.some(t=>Math.abs(t[1]-ty)<17&&tx<t[0]+t[2]&&tx+w>t[0]);j++)ty=ty+19<height-18?ty+19:Math.max(0,y-38);
  labels.push([tx,ty,w]);
  output+='<g class="svgpick" tabindex="0" role="button" aria-label="'+esc(name)+'" data-pick="'+esc(id)+'"><title>'+esc(name+" · "+(b.evidence||""))+'</title><rect x="'+x+'" y="'+y+'" width="'+(r-x)+'" height="'+(h-y)+'" fill="'+color+'" fill-opacity="'+(sel===id?.14:.025)+'" stroke="'+color+'" stroke-width="'+(sel===id?4:2)+'" '+(b.kind==="cue"?'stroke-dasharray="7 4"':'')+'/>';
  output+='<rect x="'+tx+'" y="'+ty+'" width="'+w+'" height="18" rx="2" fill="'+color+'"/><text x="'+(tx+4)+'" y="'+(ty+13.5)+'" font-size="12.5" font-weight="650" fill="#142431">'+esc(name)+'</text></g>';
 }
 $("view").innerHTML=output;$("stamp").textContent=step?"执行步重渲染":fr.row.observation.toUpperCase()+" · 原始观测";$("depthLegend").hidden=mode!=="depth";
 document.querySelectorAll("[data-mode]").forEach(n=>n.setAttribute("aria-pressed",n.dataset.mode===mode));
}

const metrics=JSON.parse(document.getElementById("metrics").textContent);
const bounds=JSON.parse(document.getElementById('bounds').textContent);
const extra=JSON.parse($("overlayData").textContent), movie=JSON.parse($("motionData").textContent);
let micro=0,focusId="",zoomed=false,playing=false;
Object.assign(on,{patch:true,region:true,viewpoints:true});
const movement=()=>movie.episodes[e().id][fi];
const poseStep=()=>micro>0?movement()[micro-1]:null;
const areaSet=()=>extra[e().id].frames[fi];
const focused=()=>areaSet()[focusId]?focusId:(f().row.frontier||"");
function mapView(){
 const ep=e(),fr=f(),[ox,oz,w,h]=ep.extent,all=areaSet(),inspect=focused(),chosen=fr.row.frontier,active=poseStep();
 let k=Math.min(604/w,408/h),dx=(640-w*k)/2,dy=(480-h*k)/2;
 const visible=bounds[ep.id][fi];
 if(visible){
   const [x0,z0,x1,z1]=visible;
   k=Math.min(7,560/(x1-x0+15),380/(z1-z0+15));
   dx=320-(x0+x1)*k/2;dy=240-(z0+z1)*k/2;
 }
 if(zoomed&&all[inspect]){
   const q=fr.frontiers.find(x=>x.id===inspect),parts=all[inspect].cells,vx=(q.xy[0]-ox)/ep.res,vz=(q.xy[1]-oz)/ep.res;
   const left=Math.min(vx,...parts.map(x=>x[1]))-8,right=Math.max(vx,...parts.map(x=>x[1]+x[2]))+8;
   const top=Math.min(vz,...parts.map(x=>x[0]))-8,bottom=Math.max(vz,...parts.map(x=>x[0]+1))+8;
   k=Math.min(11,585/(right-left),400/(bottom-top));dx=320-(left+right)*k/2;dy=235-(top+bottom)*k/2;
 }
 const proj=([x,z])=>[dx+(x-ox)/ep.res*k,dy+(z-oz)/ep.res*k];
 const pathCells=runs=>runs.map(([r,c,w])=>"M"+c+" "+r+"h"+w+"v1h-"+w+"z").join("");
 const mapTransform="translate("+dx+" "+dy+") scale("+k+")", pick=(id)=>' class="svgpick" data-pick="'+esc(id)+'" role="button" tabindex="0"';
 let s='<defs><pattern id="hatch" width="3" height="3" patternUnits="userSpaceOnUse"><rect width="3" height="3" fill="#a28aec" opacity=".38"/><path d="M-1 1l2-2M0 3l3-3M2 4l2-2" stroke="#cbb7ff" stroke-width=".6"/></pattern></defs>';
 s+='<g transform="'+mapTransform+'"><image href="'+fr.map+'" width="'+w+'" height="'+h+'" style="image-rendering:pixelated"/>';
 if(on.region&&inspect){const a=all[inspect];s+='<path data-region="'+a.rid+'" d="'+pathCells(extra[ep.id].regions[a.rid])+'" fill="url(#hatch)" pointer-events="none"/>'}
 if(on.patch){
  const ids=Object.keys(all).sort((a,b)=>Number(a===chosen||a===inspect)-Number(b===chosen||b===inspect));
  for(const id of ids){const selected=id===chosen,color=selected?"#ffb540":id===inspect?"#bfb5ff":"#28e6c0";
   s+='<path'+pick(id)+' data-cluster="'+id+'" aria-label="前沿 '+id+'，'+all[id].count+' 格" d="'+pathCells(all[id].cells)+'" fill="'+color+'" stroke="'+color+'" stroke-width="'+(selected?2.1:.7)+'" vector-effect="non-scaling-stroke"><title>'+id+' · 连通前沿 '+all[id].count+' 格</title></path>'}
 }s+='</g>';
 const point=p=>proj([p[0],p[2]]).map(x=>x.toFixed(2)).join(",");
 const line=(pts,col,width,dash)=>'<polyline points="'+pts.map(point).join(" ")+'" fill="none" stroke="'+col+'" stroke-width="'+width+'" '+(dash?'stroke-dasharray="'+dash+'"':'')+' stroke-linejoin="round" pointer-events="none"/>';
 if(on.track){
  const old=[ep.frames[0].row.pose];for(let j=0;j<fi;j++)for(const step of movie.episodes[ep.id][j])if(step.kind==="move")old.push(step.p);
  old.push(fr.row.pose);const actual=old.concat(movement().slice(0,micro).filter(x=>x.kind==="move").map(x=>x.p));
  const rest=[active?.p||fr.row.pose,...movement().slice(micro).filter(x=>x.kind==="move").map(x=>x.p)];
  s+=line(rest,"#192939",3.7,"5 4")+line(rest,"#92bdf6",2.4,"5 4")+line(actual,"#273f53",4.3)+line(actual,"#f3fbff",2.3);
 }
 const textAt=(x,y,msg,color)=>'<text x="'+x+'" y="'+y+'" font-size="13" font-weight="650" fill="'+color+'" paint-order="stroke" stroke="#152532" stroke-width="3.5" stroke-linejoin="round" pointer-events="none">'+esc(msg)+'</text>';
 const object=fr.row.target||fr.row.initial_target;
 for(const p of fr.frontiers){
  const id=p.id,hot=chosen===id,looking=inspect===id;
  if(!(hot||on.viewpoints||looking))continue;
  const [x,y]=proj(p.xy);s+='<g'+pick(id)+' aria-label="'+id+' 观察点" transform="translate('+x+' '+y+')"><title>'+id+(hot?' · 本轮前往的观察点':' · 候选观察点')+'</title><circle r="8" fill="transparent"/>';
  if(hot)s+='<circle r="10.5" fill="#302d23" stroke="#ffc861" stroke-width="1.7"/><path data-symbol="destination" d="M0-7.7 2.2-2.4 7.6-2.4 3.3 1.2 4.7 6.8 0 3.6-4.7 6.8-3.3 1.2-7.6-2.4-2.2-2.4Z" fill="#ffbd54"/>';
  else s+='<path d="M0-4.2 4.2 0 0 4.2-4.2 0Z" fill="'+(looking?'#c4b6ff':'#f9ffff')+'" stroke="#225052" stroke-width="1.3"/>';
  s+='</g>';if(hot||looking)s+=textAt(Math.max(4,Math.min(500,x+12)),Math.max(24,y-11),(hot?"前往观察点 ":"查看观察点 ")+id,hot?"#ffe0a2":"#e6e0ff")
 }
 if(on.objects)for(const o of fr.objects.filter(z=>z.xy)){const [x,y]=proj(o.xy),match=o.id===object||sel===o.id;s+='<g'+pick(o.id)+' transform="translate('+x+' '+y+')"><title>'+esc(cat(o.cat)+' '+o.id+' · 对象估计位置')+'</title><circle r="'+(match?7:3.6)+'" fill="'+(match?'#ff906f':'#e3aa99')+'" stroke="#503836" stroke-width="1.4"/></g>';if(match)s+=textAt(x+9,y+17,"对象 "+o.id,"#ffd0bf")}
 if(on.places)for(const p of fr.places){const [x,y]=proj([p.x,p.y]);s+='<rect'+pick(p.place_id)+' x="'+(x-4)+'" y="'+(y-4)+'" width="8" height="8" fill="#baadff"><title>'+p.place_id+' 地点</title></rect>'}
 const pos=active?.p||fr.row.pose,heading=active?.h||fr.heading,robot=proj([pos[0],pos[2]]);
 const angle=Math.atan2(heading[1],heading[0])*180/Math.PI;
 s+='<g data-robot="1" transform="translate('+robot.join(" ")+') rotate('+angle+')"><path d="M0 0 27-16A31 31 0 0 1 27 16Z" fill="#486cf6" opacity=".45"/><path d="m9 0-15-6 3.5 6L-6 6Z" fill="#536fff" stroke="white" stroke-width="1.7"/></g>';
 $("map").innerHTML=s;$("mapVer").textContent="M1 · 地图 v"+fr.row.map_version;
 const rect=$("map").getBoundingClientRect(),meters=zoomed?.5:2;$("scale").style.width=(meters/ep.res*k*Math.min(rect.width/640,rect.height/480))+"px";$("scale").textContent=meters+" m";
}

function timeline(){
 const ep=e(),fr=f(),r=fr.row;
 $("timeline").innerHTML=ep.frames.map((z,i)=>{const a=action(z.row),label=shortAct[a]||tr(a);
 return '<button class="tick '+(i===fi?"active":"")+'" data-frame="'+i+'" aria-current="'+(i===fi?"step":"false")+'" style="--col:'+acColor(a)+'" title="'+esc(z.row.observation+" · "+tr(a))+'"><span class="tick_num">'+String(i+1).padStart(2,"0")+'</span><span class="tick_txt">'+esc(label)+'</span></button>'}).join("");
 $("counter").textContent="当前 "+(fi+1)+" / "+ep.frames.length+" 帧";$("prev").disabled=fi===0;$("next").disabled=fi===ep.frames.length-1;
 $("play").classList.toggle("on",!!timer);$("play").textContent=timer?"暂停":"播放回合";
 const key=r.frontier||r.target||r.initial_target||"";
 $("now").textContent=tr(action(r))+(key?" · "+key:"")+(r.execution?" · "+tr(r.execution.status):r.stop_status?" · 停机成功":"");
}
const tbl=(heads,rows)=>'<div class="tablewrap"><table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join("")+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(t=>'<td>'+t+'</td>').join("")+'</tr>').join("")+'</tbody></table></div>';
const picked=id=>'<button data-pick="'+esc(id)+'">'+esc(id)+'</button>';
const card=(k,title,body)=>'<div class="card"><span>'+esc(k)+'</span><h3>'+esc(title)+'</h3><p>'+esc(body)+'</p></div>';
const note=x=>'<p class="empty">'+esc(x)+'</p>', info=(title,inner)=>'<div class="info"><h3>'+esc(title)+'</h3>'+inner+'</div>';
const xyz=a=>a?a.map(n=>num(n)).join(", "):"—";
function flow(){
 const fr=f(),r=fr.row,ex=r.execution,key=r.frontier||r.target||r.initial_target;
 const tgt=fr.boxes.filter(b=>b.kind==="goal"),ct=fr.boxes.filter(b=>b.kind==="context");
 const purpose=r.stop_status?"新帧复核同一对象，目标估距 "+num(r.distance_m)+" m，停机成功。":ex?tr(ex.status)+" · 规划路径 "+num(ex.path_length)+" m。"+(r.reason||""):r.verdicts.length?"本帧视觉判断："+r.verdicts.map(v=>v.object+" "+tr(v.status)).join("，")+"。":"等待新视角的图像证据。";
 return '<div class="cards">'+card("感知 · Qwen3.8 27B",tr(fr.status),tgt.length+" 个目标候选 · "+ct.length+" 个上下文实体 · 推理 "+num(fr.latency/1000,1)+" s")+card("记忆 · M1 / M2","已知 "+num(fr.known*.08*.08,1)+" m²","当前地点 "+fr.place+" · 面积按已知栅格计算。累计记录 "+fr.objects.length+" 个对象。")+card("探索 · M3",fr.frontiers.length+" 个当前前沿","当前区域 "+fr.area+" · 本帧"+(r.frontier?"选择观察点 "+r.frontier:"未派发前沿导航")+"。")+card("决策与执行 · V4 / TLS",tr(action(r))+(key?" · "+key:""),purpose)+'</div>';
}

function entities(){
 const fr=f(),id=sel||fr.row.frontier||fr.row.target||fr.row.initial_target||fr.boxes[0]?.globalId||fr.boxes[0]?.local_id||"";
 const b=fr.boxes.find(z=>(z.globalId||z.local_id)===id),o=fr.objects.find(z=>z.id===id),p=fr.frontiers.find(z=>z.id===id),pl=fr.places.find(z=>z.place_id===id);
 let body="";
 if(p)body='<p>当前前沿 · '+esc(tr(p.status))+'</p><p>观察点 X/Z：'+xyz(p.xy)+' m</p><p>簇中心 X/Z：'+xyz(p.center)+' m · 历史抵达 '+p.visits+' 次</p>';
 else if(pl)body='<p>地点类型 '+esc(pl.node_types.join(" / "))+' · 区域 '+esc(pl.area_id)+'</p><p>X/Z：'+xyz([pl.x,pl.y])+' m · 访问 '+pl.visit_count+' 次</p>';
 else if(b||o)body=(b?'<p>'+esc(b.evidence||"")+'</p><p>局部 ID '+esc(b.local_id)+' · '+(b.confidence!=null?"感知置信度 "+num(b.confidence):esc(b.relative_direction||"导航线索"))+'</p>':'')+(o?'<p>记忆坐标 X/Z：'+xyz(o.xy)+' m · 位置标准差 '+num(o.sigma)+' m</p><p>类别 '+esc(cat(o.cat))+' · 跟踪状态 '+esc(tr(o.status))+' · 最近出现于第 '+(o.last+1)+' 帧</p>':'');
 else body=note("选择画面中的标注框或地图中的标记查看对应记录。");
 const rows=fr.boxes.map(z=>[picked(z.globalId||z.local_id),esc(cat(z.category||z.cue_type)),esc(({goal:"目标候选",context:"上下文",cue:"导航线索"})[z.kind]),esc(z.local_id)]);
 return '<div class="details-grid">'+info(id||"当前选择",body)+info("本帧图像标注",rows.length?tbl(["对象 / 线索","类别","来源","局部 ID"],rows):note("该帧没有图像标注。"))+'</div><details class="methods"><summary>全部记忆对象、前沿和区域</summary>'+tbl(["记忆对象","类别","世界 X/Z（米）","最近观测帧"],fr.objects.map(z=>[picked(z.id),esc(cat(z.cat)),xyz(z.xy),z.last+1]))+tbl(["前沿","观察点 X/Z（米）","状态"],fr.frontiers.map(z=>[picked(z.id),xyz(z.xy),esc(tr(z.status))]))+tbl(["区域","状态","已观察 / 可评估格数"],Object.entries(fr.regions).map(([k,v])=>[esc(k),esc(tr(v.status)),v.observed+" / "+v.eligible]))+'</details>';
}
function checks(){
 const fr=f();let sections=[];
 if(!fr.row.verdicts.length){sections.push(note("本帧没有调用目标验证器。"))}
 for(const v of fr.row.verdicts){
  const a=fr.audit.find(q=>q.global_id===v.object&&q.local_id===v.candidate_id);
  const model=info("模型判定 · "+v.object,'<p><b>'+esc(tr(v.status))+'</b> · '+esc(v.candidate_id)+' · '+num(v.latency_ms/1000,2)+' s</p><p>'+esc((v.supporting_evidence||[]).join(" / "))+'</p>'+(v.uncertainty_reason?'<p>'+esc(v.uncertainty_reason)+'</p>':''));
  const target=e().instruction.target_category;
  const audit=a?info("独立核查 · MP3D 实例标注",'<p class="'+(a.center_label===target?'auditok':'auditbad')+'"><b>框中心：'+esc(cat(a.center_label))+' #'+a.center_id+'</b></p><p>中心半宽、半高区域内各实例的像素占比</p>'+tbl(["实例","类别","占比"],a.regions["0.5"].map(x=>[x.id,esc(cat(x.label)),num(x.fraction*100,1)+"%"]))):info("独立核查",note("此回合未保存该框的离线实例审计。"));
  sections.push('<div class="details-grid">'+model+audit+'</div>');
  if(v.opposing_evidence?.length)sections.push(note("模型观察到的反证："+v.opposing_evidence.join(" / ")));
  if(a?.nearby_table){const t=a.nearby_table;sections.push(note("独立空间核查：同视野椅子 #"+a.center_id+" 与桌子 #"+t.table_id+" 的标注包围盒中心水平间距 "+num(t.centers_m,2)+" m；水平包围盒间隙 "+num(t.aabb_gap_m,2)+" m。标注类别为 table，木质与餐桌用途由视觉模型判断。"))}
 }
 if(fr.row.stop_status)sections.push(note("到达后使用新图像复核 · 估计水平距离 "+num(fr.row.distance_m)+" m · "+tr(fr.row.stop_status)+"。"));
 const links=e().frames.flatMap((x,i)=>x.row.verdicts.length?['<button class="linkbtn" data-frame="'+i+'">第 '+(i+1)+' 帧</button>']:[]);
 if(links.length)sections.push('<p class="tiny">有验证记录的帧　'+links.join("　")+'</p>');
 return sections.join("");
}
function eventsView(){
 const rows=f().events.map(v=>[esc(v.sequence),esc(v.source),esc(v.event_type),esc([v.frontier_id,v.object_id,v.place_id,v.action_id].filter(Boolean).join(" · ")||"—")]);
 return rows.length?tbl(["序号","模块","原始事件类型","关联 ID"],rows):note("本帧没有与该观测编号关联的事件。");
}
function rawRecord(){
 const fr=f(),{rgb,depth,map,...record}=fr;
 return {run:e().id,scene:e().scene,goal:e().instruction,frame:record};
}
function details(){
 document.querySelectorAll("[data-tab]").forEach(x=>{x.classList.toggle("active",x.dataset.tab===tab);x.setAttribute("aria-pressed",x.dataset.tab===tab)});
 $("details").innerHTML=tab==="flow"?flow():tab==="entities"?entities():tab==="checks"?checks():tab==="events"?eventsView():'<button class="linkbtn" id="saveFrame">下载本帧 JSON</button><pre>'+esc(JSON.stringify(rawRecord(),null,2))+'</pre>';
}
function tags(){
 layers("camLayers",["goal","context","cue"],["目标框","上下文","导航线索"]);
 const cfg=[["patch","▰","前沿格簇","#17ac94"],["viewpoints","◇","候选观察点","#376d73"],["region","▧","所看区域 R","#8c66bc"],["track","━","移动轨迹","#4f7396"],["objects","●","记忆对象","#ba694f"],["places","■","拓扑地点","#8c73c0"]];
 $("mapLayers").innerHTML=cfg.map(([key,sym,t,color])=>'<button class="layer" data-layer="'+key+'" aria-pressed="'+on[key]+'"><b style="color:'+color+'">'+sym+'</b>'+t+'</button>').join("");
}
function areaInfo(){
 const fr=f(),all=areaSet(),ids=Object.keys(all).sort((a,b)=>all[a].rank-all[b].rank),id=focused(),a=all[id],r=fr.row;
 $("areaSelect").innerHTML='<option value="">选择一个前沿</option>'+ids.map(q=>'<option value="'+q+'" '+(q===id?'selected':'')+'>'+q+' · 第 '+all[q].rank+' 名'+(q===r.frontier?' · 本轮选中':'')+'</option>').join("");
 $("zoomArea").disabled=!id;$("zoomArea").textContent=zoomed&&id?"返回当前地图":"放大前沿";
 $("areaCaption").innerHTML=r.frontier?'<b class="star">★</b> 本轮前往 <strong>'+esc(r.frontier)+'</strong> 的观察点 · 金色边界 '+all[r.frontier].count+' 格':'本轮动作：<strong>'+esc(tr(action(r)))+'</strong> · 无前沿观察点被选中';
 if(a){const region=fr.regions[a.rid];
 $("areaDetails").innerHTML='<b>'+id+' · '+a.count+' 个连通前沿格</b>　候选排名 '+a.rank+' / '+ids.length+'，效用 '+num(a.score,3)+'。<br>观察点预计可见 '+a.gain+' 个未知格，栅格路径 '+num(a.path)+' m，净空 '+num(a.clearance)+' m。<br>记忆区域 '+a.rid+' 已观测 '+region.observed+' / '+region.eligible+' 个可评估格。';
 }else $("areaDetails").textContent="点击地图上的前沿边界或从上方选择编号，可以检查该簇的形状、观察点、排名及覆盖统计。";
}
function motionBar(){
 const seg=movement(),r=f().row,step=poseStep(),seq=[r.pose,...seg.map(z=>z.p)];
 let total=0,travel=0;for(let j=1;j<seq.length;j++){const d=Math.hypot(...seq[j].map((v,i)=>v-seq[j-1][i]));total+=d;if(j<=micro)travel+=d}
 $("scrub").max=seg.length||1;$("scrub").value=micro;$("scrub").disabled=!seg.length;
 $("stepBack").disabled=!micro;$("stepNext").disabled=micro>=seg.length;
 const phase=step?(step.kind==="turn"?({align:"转向行进方向",review:"转向目标复核",scan:"原地环视"}[step.phase]||"原地转向"):"沿路移动"):"原始观测";
 $("motionRead").textContent=phase+"　"+micro+" / "+seg.length+" 步";
 $("distanceRead").textContent="累计平移 "+num(travel)+" / "+num(total)+" m";
 $("motionNote").textContent=step?"同场景重渲染 · 地图与语义标注在原始观测时更新":seg.length?"播放或拖动滑块，检查这一段移动或转向":"本段没有中间位姿更新";
 const metric=metrics.find(x=>x.episode===e().id);
 if(e().align_with_travel && metric?.moving_steps){
  const current=step?.kind==="move"?"本步相机与行进方向夹角 "+num(step.heading_error,1)+"°":"平移前先转向";
  $("motionNote").textContent=current+" · 本回合 "+metric.moving_steps+" 个平移步，最大偏角 "+num(metric.max_heading_error,1)+"°"+(micro?" · 执行步重渲染":"");
 }
 $("stamp").textContent=step?"重渲染 · "+phase+" "+micro+"/"+seg.length:f().row.observation.toUpperCase()+" · 原始观测";
 $("play").classList.toggle("on",playing);$("play").textContent=playing?"暂停":"播放全过程";
}
function render(){infoHeader();tags();camera();mapView();areaInfo();timeline();motionBar();details();
 try{history.replaceState(null,"","#"+new URLSearchParams({run:e().id,frame:fi+1}))}catch(err){}}
function motionRender(){camera();mapView();motionBar()}
function stop(){clearTimeout(timer);timer=0;playing=false;motionBar()}
function go(n,manual=true){if(manual)stop();fi=Math.max(0,Math.min(e().frames.length-1,n));micro=0;sel="";focusId="";if(!focused())zoomed=false;render()}
function schedule(delay){timer=setTimeout(()=>{
 if(!playing)return;
 const factor=Number($("speed").value)/2200;
 if(micro<movement().length){micro++;motionRender();schedule(145*factor)}
 else if(fi<e().frames.length-1){go(fi+1,false);schedule(680*factor)}
 else stop();
 },delay)}
function play(){
 if(playing){stop();return}
 if(fi===e().frames.length-1&&micro===movement().length)go(0);
 playing=true;motionBar();schedule(90);
}
function seekStep(n){stop();micro=Math.min(movement().length,Math.max(0,n));motionRender()}
$("scrub").oninput=ev=>seekStep(Number(ev.target.value));
$("stepBack").onclick=()=>seekStep(micro-1);$("stepNext").onclick=()=>seekStep(micro+1);
$("areaSelect").onchange=ev=>{focusId=ev.target.value;sel=focusId;areaInfo();mapView();if(tab==="entities")details()};
$("zoomArea").onclick=()=>{zoomed=!zoomed;areaInfo();mapView()};
$("episodes").onchange=ev=>{stop();ei=Number(ev.target.value);fi=0;micro=0;sel="";focusId="";zoomed=false;render()};
$("speed").onchange=()=>{if(playing){clearTimeout(timer);schedule(60)}};
$("play").onclick=play;$("prev").onclick=()=>go(fi-1);$("next").onclick=()=>go(fi+1);
document.addEventListener("click",ev=>{
 const el=ev.target.closest("[data-mode],[data-frame],[data-tab],[data-layer],[data-pick],#saveFrame");if(!el)return;
 if(el.dataset.mode){mode=el.dataset.mode;camera()}else if(el.dataset.frame!==undefined){go(Number(el.dataset.frame))}
 else if(el.dataset.tab){tab=el.dataset.tab;details()}else if(el.dataset.layer){on[el.dataset.layer]=!on[el.dataset.layer];tags();camera();mapView()}
 else if(el.dataset.pick){stop();sel=el.dataset.pick;if(areaSet()[sel])focusId=sel;tab="entities";camera();mapView();areaInfo();motionBar();details()}
 else if(el.id==="saveFrame"){const u=URL.createObjectURL(new Blob([JSON.stringify(rawRecord(),null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download=e().id+"-"+f().row.observation+".json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
});
document.addEventListener("keydown",ev=>{
 if(ev.altKey||ev.ctrlKey||ev.metaKey)return;
 if(ev.target.closest("select,input,textarea"))return;
 if(ev.key==="ArrowLeft"||ev.key==="ArrowRight"){ev.preventDefault();go(fi+(ev.key==="ArrowLeft"?-1:1))}
 else if((ev.key==="Enter"||ev.key===" ")&&ev.target.closest("svg [data-pick]")){ev.preventDefault();ev.target.closest("svg [data-pick]").dispatchEvent(new MouseEvent("click",{bubbles:true}))}
 else if(ev.key===" "&&!ev.target.closest("button,summary,a")){ev.preventDefault();play()}
});
window.addEventListener("resize",()=>{clearTimeout(frameTimer);frameTimer=setTimeout(mapView,80)});
render();

if(new URLSearchParams(location.search).get("demo")==="1") play();
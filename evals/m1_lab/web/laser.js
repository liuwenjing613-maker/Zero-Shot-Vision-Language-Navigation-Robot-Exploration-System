"use strict";
const $=id=>document.getElementById(id);
const text=(id,value)=>{$(id).textContent=value};
const pct=x=>(100*x).toFixed(1)+"%";
let mode=location.hash==="#room"?"room360":"laser360",run="verified",frames=[],index=0,current=null;
let timer=null,live=false,request=0,layout,chosen=null;
let reference=null,refError="";
async function loadReference(){
 try{
   const meta=await get("/mp3d/topdown.json");
   const photo=new Image();photo.src="/mp3d/topdown.jpg";await photo.decode();
   if(photo.naturalWidth!==meta.width||photo.naturalHeight!==meta.height)throw Error("图像尺寸不符");
   reference={meta,photo};
 }catch(e){refError=e.message}
 if(current)mapView();
}
const root=()=>"/runs/"+run+"/"+mode+"/";
async function get(url,options){const r=await fetch(url,options);if(!r.ok)throw Error("HTTP "+r.status);return r.json()}
function stop(){clearInterval(timer);timer=null;text("play","播放")}
function mapView(){
 const f=current,c=$("grid"),g=c.getContext("2d"),[h,w]=f.shape;
 const scale=Math.min((c.width-20)/w,(c.height-20)/h),ox=(c.width-w*scale)/2,oy=(c.height-h*scale)/2;
 layout={scale,ox,oy};const map=p=>[ox+(p[0]-f.origin[0])*scale/f.resolution,oy+(p[1]-f.origin[1])*scale/f.resolution];
 g.fillStyle="#eaf0f1";g.fillRect(0,0,c.width,c.height);
 const src=document.createElement("canvas");src.width=w;src.height=h;const sc=src.getContext("2d"),im=sc.createImageData(w,h);
 const data=f.decoded;for(let i=0;i<data.length;i++){const col=data[i]===255?[185,201,206]:data[i]===100?[21,57,72]:[252,252,248];im.data.set([...col,255],4*i)}
 sc.putImageData(im,0,0);g.imageSmoothingEnabled=false;g.drawImage(src,ox,oy,w*scale,h*scale);
 if($("truth").checked){g.strokeStyle="#953eaa";g.lineWidth=1;for(const i of f.scan.reference)g.strokeRect(ox+(i%w)*scale,oy+Math.floor(i/w)*scale,scale,scale)}
 const [cx,cy]=map(f.xy);g.strokeStyle="#de902b";g.lineWidth=2;g.beginPath();
 f.scan.trajectory.forEach((p,i)=>i?g.lineTo(...map(p)):g.moveTo(...map(p)));g.stroke();
 const rays=$("rays").checked,dots=$("dots").checked;
 f.scan.ranges.forEach((d,i)=>{if(d===null)return;const theta=f.scan.heading+i*Math.PI/360,len=d==="clear"?f.scan.limit:d;
 const px=cx+len*Math.cos(theta)*scale/f.resolution,py=cy+len*Math.sin(theta)*scale/f.resolution;
 if(rays){g.strokeStyle="rgba(17,130,125,0.14)";g.lineWidth=.65;g.beginPath();g.moveTo(cx,cy);g.lineTo(px,py);g.stroke()}
 if(dots&&d!=="clear"){g.fillStyle="#0eaa95";g.fillRect(px-1.3,py-1.3,2.6,2.6)}});
 g.fillStyle="#ec5538";g.beginPath();g.arc(cx,cy,4.5,0,2*Math.PI);g.fill();
 g.strokeStyle="#ec5538";g.lineWidth=2;g.beginPath();g.moveTo(cx,cy);g.lineTo(cx+12*Math.cos(f.scan.heading),cy+12*Math.sin(f.scan.heading));g.stroke();
 if(chosen){const [r,k]=chosen;g.strokeStyle="#c21f66";g.strokeRect(ox+k*scale,oy+r*scale,scale,scale);
 const i=r*w+k,odds=new DataView(Uint8Array.from(atob(f.odds),x=>x.charCodeAt(0)).buffer);
 text("cell","行 "+r+" / 列 "+k+" · "+({0:"空闲",100:"占用",255:"未知"})[data[i]]+" · log-odds "+(odds.getInt16(i*2,true)/100).toFixed(2))}
 drawOrientation(g,scale,f.resolution);drawReference()
}
function drawOrientation(g,scale,resolution){
 const meter=scale/resolution;
 g.save();g.fillStyle="rgba(255,255,255,.92)";g.fillRect(11,10,Math.max(95,meter+25),40);
 g.fillStyle="#173846";g.font="13px sans-serif";g.fillText("+X →   +Z ↓",17,26);
 g.fillRect(17,38,meter,2);g.fillText("1 m",21+meter,43);g.restore()
}
function drawReference(){
 if(mode!=="laser360"||!current)return;
 const c=$("scene"),g=c.getContext("2d"),f=current,[h,w]=f.shape,{scale,ox,oy}=layout;
 g.fillStyle="#eaf0f1";g.fillRect(0,0,c.width,c.height);
 if(!reference){text("sceneNote",refError?"俯视图读取失败："+refError:"读取餐厅纹理…");return}
 const m=reference.meta;
 if(m.shape[0]!==h||m.shape[1]!==w||Math.abs(m.resolution-f.resolution)>1e-8||m.origin.some((v,i)=>Math.abs(v-f.origin[i])>1e-6)){
   text("sceneNote","本次地图范围与俯视图不同，已停用叠加。");return
 }
 g.imageSmoothingEnabled=true;g.drawImage(reference.photo,ox,oy,w*scale,h*scale);
 if($("occupiedOverlay").checked){
   g.fillStyle="rgba(0,156,165,.75)";f.decoded.forEach((v,i)=>{if(v===100)g.fillRect(ox+(i%w)*scale,oy+Math.floor(i/w)*scale,scale,scale)})
 }
 const pt=p=>[ox+(p[0]-m.origin[0])*scale/m.resolution,oy+(p[1]-m.origin[1])*scale/m.resolution];
 const line=()=>{g.beginPath();f.scan.trajectory.forEach((p,i)=>i?g.lineTo(...pt(p)):g.moveTo(...pt(p)))};
 line();g.lineWidth=4;g.strokeStyle="#624628";g.stroke();line();g.lineWidth=2.4;g.strokeStyle="#ffc65a";g.stroke();
 const [x,y]=pt(f.xy),a=f.scan.heading;g.beginPath();g.arc(x,y,5,0,2*Math.PI);g.fillStyle="#ec5538";g.fill();g.strokeStyle="#fff";g.lineWidth=1.5;g.stroke();
 g.beginPath();g.moveTo(x,y);g.lineTo(x+12*Math.cos(a),y+12*Math.sin(a));g.strokeStyle="#f53f23";g.lineWidth=2.5;g.stroke();
 if(chosen){const [r,k]=chosen;g.strokeStyle="white";g.lineWidth=4;g.strokeRect(ox+k*scale,oy+r*scale,scale,scale);g.strokeStyle="#c21f66";g.lineWidth=2;g.strokeRect(ox+k*scale,oy+r*scale,scale,scale)}
 drawOrientation(g,scale,m.resolution);
 text("sceneNote","橙线为累计轨迹，红点为当前位姿。点击任一张图可对照同一格。俯视图从地面上方 2.05 m 向下取景，浅灰处缺少该高度范围内的可见表面。")
}
function polar(){
 const c=$("polar"),g=c.getContext("2d"),s=current.scan,cx=c.width/2,cy=c.height/2+5,rad=195;
 g.fillStyle="#142e3b";g.fillRect(0,0,c.width,c.height);g.font="14px sans-serif";g.lineWidth=1;
 for(let m=1;m<=6;m++){g.strokeStyle="#35505c";g.beginPath();g.arc(cx,cy,rad*m/6,0,Math.PI*2);g.stroke();if(m%2===0){g.fillStyle="#adc5cb";g.fillText(m+" m",cx+3,cy-rad*m/6+15)}}
 for(let j=0;j<4;j++){const a=j*Math.PI/2;g.strokeStyle="#35505c";g.beginPath();g.moveTo(cx,cy);g.lineTo(cx+rad*Math.sin(a),cy-rad*Math.cos(a));g.stroke()}
 s.ranges.forEach((d,i)=>{if(d===null)return;const hit=d!=="clear",r=(hit?d:s.limit)*rad/s.limit,a=i*Math.PI/360;
 const x=cx+Math.sin(a)*r,y=cy-Math.cos(a)*r;g.strokeStyle=hit?"rgba(82,190,182,0.11)":"rgba(224,158,65,0.15)";
 g.beginPath();g.moveTo(cx,cy);g.lineTo(x,y);g.stroke();g.fillStyle=hit?"#69e3cf":"#eeb356";g.fillRect(x-1,y-1,2,2)});
 g.fillStyle="#fff";g.fillText("前",cx-7,22);g.fillText("右",c.width-29,cy);g.fillText("后",cx-7,c.height-12);g.fillText("左",16,cy);
 g.beginPath();g.arc(cx,cy,3,0,2*Math.PI);g.fill()
}
function details(){
 const f=current,s=f.scan,m=s.metrics;const box=$("metrics");box.replaceChildren();
 for(const [name,val] of [["占用 IoU",pct(m.iou)],["有效束空闲 IoU",pct(m.free_iou)],["全角理想空闲 IoU",pct(m.ideal_free_iou)],["容差精确率",pct(m.precision)],["容差召回率",pct(m.recall)],["本圈测距 P95",(100*m.range_p95).toFixed(2)+" cm"]]){
 const item=document.createElement("div"),b=document.createElement("b"),small=document.createElement("small");b.textContent=val;small.textContent=name;item.append(b,small);box.append(item)}
 text("how","参考每圈 1,440 个方向。IoU 按栅格精确比对；精确率与召回率允许 0.15 m 偏差。测距误差使用本圈 "+m.paired+" 对有效回波。理想空闲分数也计入传感器缺失回波的角区。");
 text("returns",s.hits+" 束命中 · "+s.clear+" 束量程内空闲 · "+s.invalid+" 束缺失");
 text("pose","X "+f.xy[0].toFixed(2)+" m · Z "+f.xy[1].toFixed(2)+" m · 朝向 "+(s.heading*180/Math.PI).toFixed(1)+"°");
 text("topo",f.places.map(p=>p.place_id+"（访问 "+p.visit_count+"）").join("、")+"　"+f.edges.map(e=>e.source_id+" → "+e.target_id+" "+e.status).join("，"));
 text("coverage","空闲 "+f.counts.free+" 格 · 占用 "+f.counts.occupied+" 格 · 未知 "+f.counts.unknown+" 格。本圈有 "+f.changed+" 格改变分类。");
 $("checks").replaceChildren(...f.checks.map(x=>{const p=document.createElement("p");p.className=x.passed?"check":"bad";p.textContent=(x.passed?"✓　":"×　")+x.label;return p}));
 text("title",f.title);text("detail",f.detail);text("num","第 "+(index+1)+" / "+frames.length+" 圈");$("seek").max=frames.length-1;$("seek").value=index;
 $("prev").disabled=index===0;$("next").disabled=index===frames.length-1;$("raw").href=root()+index+".json"
}
async function show(i){const ticket=++request;const url=root();i=Math.max(0,Math.min(i,frames.length-1));const f=await get(url+i+".json");if(ticket!==request||url!==root())return;
 index=i;current=f;current.decoded=Uint8Array.from(atob(f.cells),c=>c.charCodeAt(0));mapView();polar();details()}
async function refresh(latest=false,initialIndex=0){const url=root(),list=await get(url+"steps.json");if(url!==root())return;frames=list;await show(latest?list.length-1:initialIndex)}
function setMode(m){$("scanviews").classList.toggle("room",m==="room360");stop();++request;mode=m;chosen=null;text("cell","点击栅格查看占用证据。");$("mp").classList.toggle("active",m==="laser360");$("room").classList.toggle("active",m==="room360")}
function watch(fn){return ()=>fn().catch(e=>text("state","读取失败 · "+e.message))}
for(const [id,m] of [["room","room360"],["mp","laser360"]])$(id).onclick=watch(async()=>{live=false;setMode(m);await refresh()});
for(const [id,offset] of [["prev",-1],["next",1]])$(id).onclick=watch(async()=>{live=false;stop();await show(index+offset)});
$("seek").oninput=watch(async()=>{live=false;stop();await show(Number($("seek").value))});
$("play").onclick=()=>{live=false;if(timer){stop();return}if(index===frames.length-1)show(0);text("play","暂停");timer=setInterval(()=>{if(index===frames.length-1)stop();else show(index+1)},850)};
for(const id of ["dots","truth","rays","occupiedOverlay"])$(id).onchange=()=>current&&mapView();
for(const id of ["grid","scene"])$(id).onclick=e=>{if(!current)return;const c=$(id),b=c.getBoundingClientRect(),r=Math.floor(((e.clientY-b.top)*c.height/b.height-layout.oy)/layout.scale),k=Math.floor(((e.clientX-b.left)*c.width/b.width-layout.ox)/layout.scale);
 if(r>=0&&k>=0&&r<current.shape[0]&&k<current.shape[1]){chosen=[r,k];mapView()}};
async function follow(){
 const st=await get("/api/scan-status");$("run").disabled=st.status==="running";
 text("state",st.status==="running"?"服务器计算中":st.status==="failed"?"运行失败 · "+st.error:"服务器已连接");
 if(live){for(const m of ["laser360","room360"]){try{const list=await get("/runs/"+st.id+"/"+m+"/steps.json");if(mode!==m)setMode(m);run=st.id;frames=list;await show(list.length-1);break}catch(e){if(!e.message.includes("404"))throw e}}}
 if(st.status==="running")setTimeout(watch(follow),700);else live=false
}
$("run").onclick=watch(async()=>{stop();$("run").disabled=true;try{await get("/api/scan",{method:"POST",headers:{"X-M1-Lab":"run"}});live=true;await follow()}catch(e){$("run").disabled=false;throw e}});
watch(async()=>{const st=await get("/api/scan-status");run=st.status==="ready"?st.id:"verified";setMode(mode);const scan=Number(new URLSearchParams(location.search).get("scan"));await refresh(false,Number.isSafeInteger(scan)&&scan>0?scan-1:0);await follow()})();

loadReference();

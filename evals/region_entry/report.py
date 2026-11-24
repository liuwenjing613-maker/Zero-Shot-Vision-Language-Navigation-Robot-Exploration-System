"""Package an offline step-by-step replay; maps use the identical X/Z window."""
import base64
import json
import sys
from pathlib import Path


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MP3D 穿门实测</title>
<style>*{box-sizing:border-box}body{margin:20px auto;padding:0 16px;max-width:1240px;font:16px/1.55 system-ui;color:#21313a}h1{font-size:26px;margin:0}button,select{padding:6px;font:inherit}input{width:min(400px,48vw);vertical-align:middle}main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}canvas,img{width:100%;background:#e4e7e7}canvas{aspect-ratio:1}figcaption{font-weight:650}#msg{padding:.6em;background:#e8f3f5}button{cursor:pointer}@media(max-width:700px){main{grid-template-columns:1fr 1fr}figure:first-child{grid-column:span 2}figure:first-child img{max-height:260px;object-fit:contain}}</style>
<h1>MP3D 穿门实测</h1><p>场景 2azQ1b91cZZ · 前进每步约 0.25 米 · 灰色未知，白色可通行，深色为激光命中。两张地图方向和比例一致，每张宽 9 米。</p>
<select id="route"></select> <button id="play">播放</button> <input id="bar" type="range" min="0" value="0"> <output id="step"></output>
<p id="msg"></p><main><figure><figcaption>机器人第一视角</figcaption><img id="rgb" alt="当前帧 RGB"></figure><figure><figcaption>360° 扫描占据网格</figcaption><canvas id="grid" width="720" height="720"></canvas></figure><figure><figcaption>同位置 MP3D 纹理俯视图</figcaption><canvas id="ref" width="720" height="720"></canvas></figure></main>
<p>橙线为检测门口，蓝线为已走轨迹，蓝点为当前位置，绿点为已触发的区域入口节点。第一帧橙框为 Qwen 候选。紫色虚线标出 MP3D 人工区域边界的局部方向。俯视图和 R 编号只用于验收。</p><p id="summary"></p><p id="before"></p><div id="events"></div><p><a id="raw" download>下载此路线的逐帧记录与原始感知结果</a></p>
<script>const data=DATA; const $=id=>document.getElementById(id), names={1:'厨房 R3 ↔ 过厅 R9',6:'过厅 R16 ↔ 卧室 R13',2:'厨房 R3 ↔ 餐厅 R4',5:'过厅 R9 ↔ 厨房 R3',15:'餐厅 R2 ↔ 过厅 R12',9:'卧室 R13 ↔ 卫浴 R14',17:'复核 · 过厅 R12 ↔ 餐厅 R2'},phase={start:'起点',approach:'靠近门口',retreat:'门前退回',outbound:'穿门去程',return:'原路返回'};let timer,rev=0,rawURL,rawID;
function pic(id,src,d,k){const im=new Image();im.onload=()=>{if(k!==rev)return;const c=$(id).getContext('2d');c.imageSmoothingEnabled=id==='ref';c.drawImage(im,0,0,720,720);const pt=p=>p.map((v,i)=>(v-d.origin[i])*80);
const line=(ps,color,w)=>{c.beginPath();ps.map(pt).forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle=color;c.lineWidth=w;c.stroke()};
const dot=(p,col,r)=>{let [x,y]=pt(p);c.beginPath();c.arc(x,y,r,0,7);c.fillStyle=col;c.fill();c.strokeStyle='white';c.lineWidth=2;c.stroke()};
let trace=d.trace.slice(0,+$('bar').value+1);if(id==='ref'){const f=d.fixture,n=f.normal,t=[-n[1],n[0]],ends=[-1,1].map(h=>f.center.map((v,i)=>v+h*t[i]*.75));c.setLineDash([8,6]);line(ends,'#92369e',3);c.setLineDash([])}d.portals.filter(p=>trace.at(-1).known_portals.includes(p.id)).forEach(p=>line([p.a,p.b],'#eb7913',6));line(trace.map(x=>x.xy),'#3588bc',4);trace.filter(x=>x.event).forEach(x=>dot(x.xy,'#15a64d',10));dot(trace.at(-1).xy,'#185886',5)};im.src=src}
function show(){let d=data[$('route').value],n=+$('bar').value,r=d.trace[n],k=++rev;if(rawID!==d.index){if(rawURL)URL.revokeObjectURL(rawURL);let {rgbs,grids,ref,start,...log}=d;rawURL=URL.createObjectURL(new Blob([JSON.stringify(log,null,2)],{type:"application/json"}));$('raw').href=rawURL;$('raw').download='mp3d-route-'+d.index+'.json';rawID=d.index}$('bar').max=d.trace.length-1;$('step').textContent=n+' / '+(d.trace.length-1);
$('rgb').src=n?d.rgbs[n]:d.start;pic('grid',d.grids[n],d,k);pic('ref',d.ref,d,k);$('msg').textContent=phase[r.phase]+'　|　人工区域 '+(r.regions.map(x=>'R'+x).join('/')||'边界')+'　|　记忆区域 '+r.area+'　|　'+(r.event?'触发 '+r.event.portal_id+' → '+r.event.place:'本帧无建点');
$('summary').textContent='本次轨迹几何验证通过 '+new Set(d.proposals.filter(x=>x.status==='VALIDATED').map(x=>x.portal_id)).size+' 个门口。门前折返、去程、返程和门口复用四项验收'+(Object.values(d.criteria).every(Boolean)?'均通过。':'未通过：'+Object.keys(d.criteria).filter(k=>!d.criteria[k]).map(k=>({negative:'门前折返',outbound:'去程',inbound:'返程',same_portal:'门口复用'})[k]).join('、')+'。');
$('before').textContent=d.baseline?'初版对照：只处理门口或开口候选时，这条路线去程和返程均漏检。':(d.index===17?'此方向在规则固定后单独复核。':'');if(d.repeat_events)$('before').textContent+=' 第二轮往返复用了原节点：'+d.repeat_events.map(e=>e.place).join(' → ')+'。';$('events').replaceChildren(...d.trace.filter(x=>x.event).map(x=>{let b=document.createElement('button');b.textContent='跳到第 '+x.step+' 步 · '+phase[x.phase];b.onclick=()=>{$('bar').value=x.step;show()};return b}))}
Object.keys(data).forEach(i=>$('route').add(new Option(names[i]||i,i)));$('route').onchange=()=>{$('bar').value=0;show()};$('bar').oninput=show;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';return}if(+$('bar').value>=+$('bar').max)$('bar').value=0;$('play').textContent='暂停';timer=setInterval(()=>{if(+$('bar').value>=+$('bar').max){$('play').click();return}$('bar').value=+$('bar').value+1;show()},550)};show()</script></html>"""


def run(out, indices):
    def uri(name):
        p = out / name
        return (
            "data:image/"
            + ("png" if p.suffix == ".png" else "jpeg")
            + ";base64,"
            + base64.b64encode(p.read_bytes()).decode()
        )

    archive = out / "front-only-baseline.json"
    baseline = json.loads(archive.read_text()) if archive.is_file() else {}
    bundle = {}
    for i in indices:
        d = json.loads((out / f"{i}.json").read_text())
        d["baseline"] = baseline.get(str(i))
        extra = out / f"repeat-{i}.json"
        if extra.is_file():
            record = json.loads(extra.read_text())
            d["repeat_events"] = [r["event"] for r in record["trace"] if r["event"]]
        d.update(
            start=uri(f"{i}-start.jpg"),
            ref=uri(f"reference-{i}/topdown.jpg"),
            grids=[uri(f"{i}-grid-{r['step']}.png") for r in d["trace"]],
            rgbs=[uri(f"{i}-rgb-{r['step']}.jpg") for r in d["trace"]],
        )
        bundle[i] = d
    (out / "index.html").write_text(
        PAGE.replace(
            "const data=DATA",
            "const data=" + json.dumps(bundle).replace("<", "\\u003c"),
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2:])

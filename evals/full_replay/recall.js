(() => {
  const report = JSON.parse(document.getElementById("recall-data").textContent);
  if (!report) return;
  const panel = document.getElementById("recall");
  const episode = db.find(ep => ep.id === report.source);
  const last = episode?.frames.find(frame => frame.row.observation === report.source_frame);
  panel.hidden = false;
  panel.open = new URLSearchParams(location.search).get("view") === "memory";
  const sighting=report.return_checks.flatMap(check=>check.candidates.map(item=>({...item,check}))).find(item=>item.instance===report.instance_before);
  const photos = [
    [report.history_image, "1 · 离开前确认", "历史对象 " + report.target + "，场景椅子 #" + report.instance_before,report.history_box],
    [report.query_image, "2 · 回到起点附近", "当前帧未检出目标。语义图中原椅子占 " + report.query_instance_pixels + " 像素"],
    [sighting?.check.image || report.return_image, "3 · 完整记忆组回访", sighting ? "同一场景椅子 #"+sighting.instance+"，系统关联为 "+(sighting.associated_object||"未关联") : "到达后的实际视野",sighting?.bbox]
  ];
  const photoStrip = photos.filter(([url]) => url).map(([url,title,caption,box]) =>
    '<figure><svg viewBox="0 0 640 480" role="img" aria-label="'+esc(title)+'"><image href="'+esc(url)+'" width="640" height="480"/>'+(box?'<rect x="'+box[0]*.64+'" y="'+box[1]*.48+'" width="'+(box[2]-box[0])*.64+'" height="'+(box[3]-box[1])*.48+'" fill="none" stroke="#ffd462" stroke-width="4"/>':'')+'</svg><figcaption><b>'+esc(title)+'</b>'+esc(caption)+'</figcaption></figure>').join("");
  const verdict = report.return_checks.flatMap(check => check.verdicts).slice(-1)[0];
  function decision(label, arm, text) {
    return '<article><h3>'+label+'</h3><strong>'+esc(tr(arm.action))+'</strong> <code>'+esc(arm.target || arm.frontier || "")+'</code><p>'+esc(text)+'</p></article>';
  }
  const arms = decision("保留跨帧对象记忆", report.full,
    report.full.object_count+" 个对象进入决策。"+tr(report.full.status)+"，规划路径 "+num(report.full.length)+" m。") +
    decision("仅当前帧对象", report.current,
      report.current.object_count+" 个对象进入决策。"+tr(report.current.status)+"，规划路径 "+num(report.current.length)+" m。");
  let map="";
  if (last) {
    const [ox,oz,w,h]=episode.extent;
    const cell=(x,z)=>[(x-ox)/episode.res,(z-oz)/episode.res];
    const routes=[report.detour_path, report.full.path||[], report.current.path||[]].map(path=>path.map(p=>cell(p[0],p[2])));
    const goal=cell(...report.goal_xz), start=cell(report.query_pose[0],report.query_pose[2]);
    const pts=[goal,start,...routes.flat()];
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const x=Math.min(...xs)-12,y=Math.min(...ys)-12,sw=Math.max(...xs)-x+12,sh=Math.max(...ys)-y+12;
    const lines=routes.map((route,i)=>'<polyline fill="none" stroke="'+["#fcfcfc","#28dbb5","#ffa853"][i]+'" stroke-width="'+[1,2,1.6][i]+'" '+(i===0?'stroke-dasharray="2 2"':'')+' points="'+route.map(p=>p.join(",")).join(" ")+'"/>').join("");
    map='<figure class="rec-route"><svg role="img" aria-label="两组从同一位置出发的实际移动轨迹" viewBox="'+[x,y,sw,sh].join(" ")+'"><image href="'+esc(last.map)+'" width="'+w+'" height="'+h+'"/>'+lines+'<circle cx="'+start[0]+'" cy="'+start[1]+'" r="2.3" fill="#fff" stroke="#172830"/><circle cx="'+goal[0]+'" cy="'+goal[1]+'" r="2.3" fill="#f48ada" stroke="#172830"/></svg><figcaption>历史占据网格 · 白虚线为离开椅子的路线，白点为共同出发点，粉点为记忆位置。绿线为完整组，橙线为当前帧组。</figcaption></figure>';
  }
  const final = verdict ? "回访复核 "+tr(verdict.status)+"，距对象 "+num(verdict.distance_m)+" m，框中心场景实例 #"+verdict.instance+"。" : sighting ? "回访画面里的候选框再次命中场景椅子 #"+sighting.instance+"，但 M2 把它关联为 "+sighting.associated_object+"。与任务锁定的 "+report.target+" 身份不一致，系统没有触发同一目标复核或成功停机。" : "本次未取得回访对象的复核结果。";
  panel.querySelector(".rec-body").innerHTML='<p>任务要求先确认餐桌旁的木椅，再沿导航网格回到起点附近，最后返回同一把椅子。离开椅子的 '+num(report.detour_m)+' m 路段由测试指定。两组从同一个相机位姿出发，共用同一张 RGB、深度、占据网格和 '+report.frontiers+' 个前沿候选。</p><div class="rec-photos">'+photoStrip+'</div><div class="rec-split">'+map+'<div><div class="rec-arms">'+arms+'</div><p>'+esc(final)+' '+(report.same_instance?'离开前后命中同一场景实例。':'')+'</p><p>两个分支都执行了首个导航动作。仅当前帧组的后续搜索成功率未测量。</p></div></div><details><summary>展开联合决策的输入与对照</summary><p>对象记忆 M2 提供历史身份、坐标与验证记录。历史验证来自 '+esc(report.verification_frames.join("、"))+'。保留 M2、仅从可达集合移除该对象，规划器选择了 <b>'+esc(tr(report.path_gate.action))+'</b>。此项只重新计算决策。完整组的决策证据 ID 为 '+esc(report.full.evidence_ids.join('、'))+'。</p><p>前沿候选来自占据网格与 M3 前沿记录。M4 提供事件记录，本次实验没有检验移除 M4 对这一动作的影响。场景实例号只用于事后核对。</p></details>';
  if (panel.open) requestAnimationFrame(()=>panel.scrollIntoView({block:"start"}));
})();
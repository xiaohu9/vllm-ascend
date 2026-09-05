 顶部的 DATA 对象,换真数据只改那里
  维护约定:
    · 更新一个实验 -> 改对应 section 的 data-status + 内容,文件自足无构建
    · 跑完 NPU 采集 -> 按 §5 的采集命令,把 report.png 拷到本文件旁,
      再把 <figure class="slot"> 里替换成 <img src="report.png">
    · 状态徽章:data-status="done|progress|planned|risk",配图标+文字,不单靠颜色
    · 进度区在 hero 之后、§0 之前:新增任务/实验先记进度区,再落到正文节
  事实与数字来源:plans/pivot_*.md 五份设计文档 + pivot_refineop 分支代码,见附录。
-->
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PIVOT-Refine on Ascend — 研究进展</title>
<style>
/* ============ 设计令牌(Mooncake 基底 + 原内容兼容变量) ============ */
:root{
  --bg:#f6f7fb; --card:#ffffff; --line:#d8dce8; --text:#1c2130; --muted:#5b6478;
  --accent:#3b5bdb; --accent2:#0ca678; --warn:#e8590c; --danger:#c92a2a;
  --code-bg:#f1f3f9; --chip-bg:#eef1fb; --head-bg:#eef1fb;
  --shadow:0 1px 3px rgba(20,30,60,.08),0 4px 14px rgba(20,30,60,.06);
  /* 原内容兼容变量(映射到 Mooncake 色) */
  --paper:var(--bg); --paper-2:#f1f3f9; --ink:var(--text); --faint:#8b92a5;
  --rule:var(--line); --accent-ink:#2b47a8; --accent-soft:var(--chip-bg);
  --good:#2f7d5a; --good-bg:#e6f4ec; --warn2:#a3782a; --warn-bg:#f7efdd;
  --serious:#8b2e2e; --serious-bg:#f5e7e5; --info:#3a5a80; --info-bg:#e9eef5;
  --serif:Georgia,"Palatino Linotype","Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,Helvetica,Arial,sans-serif;
  --mono:"SFMono-Regular",Consolas,"Cascadia Code","Courier New",monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0e1117; --card:#161b26; --line:#2a3142; --text:#e6e9f2; --muted:#9aa3b8;
    --accent:#6d8dff; --accent2:#3ddc97; --warn:#ffa94d; --danger:#ff6b6b;
    --code-bg:#1d2433; --chip-bg:#1f2740; --head-bg:#1d2433;
    --shadow:0 1px 3px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.3);
    --paper:var(--bg); --paper-2:#1d2433; --ink:var(--text); --faint:#7b8498;
    --rule:var(--line); --accent-ink:#9db4ff; --accent-soft:var(--chip-bg);
    --good:#4dd09a; --good-bg:#16362a; --warn2:#c9a05a; --warn-bg:#33291a;
    --serious:#ff8585; --serious-bg:#3a1e1e; --info:#7ba0cf; --info-bg:#1e2a3a;
  }
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);
  line-height:1.65;font-size:15px;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1060px;margin:0 auto;padding:32px 22px 90px;}

/* ============ hero / TLDR ============ */
header.hero{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:26px;}
header.hero h1{font-family:var(--serif);font-size:28px;margin:0 0 10px;letter-spacing:.2px;line-height:1.25}
header.hero .tag{display:inline-block;background:var(--chip-bg);color:var(--accent);
  border:1px solid var(--line);border-radius:999px;padding:2px 12px;font-size:12.5px;margin:0 6px 6px 0;}
.tldr{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:10px;padding:14px 18px;margin:0 0 26px;box-shadow:var(--shadow);}
.tldr strong{color:var(--accent)}
.tldr .big{font-size:19px;font-weight:800;color:var(--accent)}

/* ============ 编号节 ============ */
h2{font-size:20px;margin:44px 0 6px;padding-top:14px;border-top:1px solid var(--line);}
h2 .no{color:var(--accent);margin-right:8px}
h3{font-size:16px;margin:26px 0 8px;}
h4{font-size:14px;font-weight:700;margin:18px 0 8px}
p{margin:8px 0}
strong{font-weight:600}
em{font-style:normal;color:var(--accent-ink)}
.muted{color:var(--muted)}
code{background:var(--code-bg);border:1px solid var(--line);border-radius:5px;
  padding:1px 6px;font-size:13px;font-family:var(--mono);}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;overflow-x:auto;font-size:12.8px;line-height:1.55;font-family:var(--mono);}
pre code{border:none;padding:0;background:none;font-size:inherit}
pre .c{color:var(--faint)}
pre .k{color:#b8860b}
pre .t{color:var(--accent)}

/* ============ 表格 ============ */
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:var(--shadow);}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top;}
th{background:var(--head-bg);font-weight:600;white-space:nowrap}
tr:nth-child(even) td{background:rgba(120,130,160,.04)}
tr:hover td{background:rgba(120,130,160,.07)}
td code{background:none;border:none;padding:0;font-size:12px}
.tblwrap{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:10px;}
.tblwrap table{margin:0;border:none;box-shadow:none}

/* ============ 卡片 / 网格 / 状态 ============ */
.grid{display:grid;gap:14px;margin:16px 0}
.g2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
@media(max-width:780px){.grid2,.grid3{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;margin:14px 0;box-shadow:var(--shadow);}
.grid .card{margin:0}
.card h4{margin-top:0}
.card p{font-size:13.5px;color:var(--muted);margin-top:4px}
.card .tag{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:8px;display:block}
.pill{display:inline-block;border-radius:999px;padding:0 10px;font-size:12px;line-height:20px;
  border:1px solid var(--line);background:var(--chip-bg);margin:2px 3px 2px 0;white-space:nowrap;}
.pill.b{color:var(--accent)} .pill.g{color:var(--accent2)} .pill.o{color:var(--warn)}
.ok{color:var(--accent2);font-weight:700}
.bad{color:var(--danger);font-weight:700}
.warn{color:var(--warn);font-weight:700}
.note{font-size:12.8px;color:var(--muted);border-left:3px solid var(--line);padding:4px 10px;margin:10px 0;}
.filedim{font-size:12px;color:var(--muted);font-family:var(--mono);word-break:break-all;}

/* 统计块 */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;text-align:left;box-shadow:var(--shadow)}
.tile .v{font-family:var(--serif);font-size:30px;font-weight:700;color:var(--accent-ink);line-height:1}
.tile .v small{font-size:15px;color:var(--muted);font-weight:400}
.tile .l{margin-top:7px;font-size:12px;color:var(--muted)}

/* 状态徽章(图标+标签,不单靠颜色) */
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;
  padding:3px 10px;border-radius:20px;border:1px solid var(--line);white-space:nowrap;background:var(--chip-bg);}
.badge .ic{font-size:11px;line-height:1}
.badge.done{background:var(--good-bg);color:var(--good);border-color:var(--good)}
.badge.progress{background:var(--warn-bg);color:var(--warn2);border-color:var(--warn2)}
.badge.planned{background:var(--info-bg);color:var(--info);border-color:var(--info)}
.badge.risk{background:var(--serious-bg);color:var(--serious);border-color:var(--serious)}
.status-line{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}

/* 引语/定理/图 */
.theorem{border-left:3px solid var(--accent);background:var(--accent-soft);
  border-radius:0 8px 8px 0;padding:14px 18px;margin:16px 0;}
.theorem .hint{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:4px}
.theorem p{margin:4px 0}
.quote{border-left:2px solid var(--rule);padding:2px 0 2px 16px;color:var(--muted);font-style:italic;margin:14px 0}
figure{margin:18px 0}
figure figcaption{font-size:12px;color:var(--faint);margin-top:8px;line-height:1.5}
.slot{border:1.5px dashed var(--rule);border-radius:10px;background:var(--paper-2);
  padding:34px 22px;text-align:center;color:var(--faint);}
.slot .big{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;display:block;margin-bottom:8px}
img.fig{width:100%;border:1px solid var(--rule);border-radius:10px;background:#fff}

/* 任务看板 */
.task{display:flex;gap:12px;align-items:flex-start;background:var(--card);border:1px solid var(--line);
  border-radius:12px;padding:12px 16px;margin:10px 0;box-shadow:var(--shadow);}
.task .st{flex:0 0 auto;width:66px;text-align:center;font-size:12px;font-weight:700;
  border-radius:999px;padding:3px 0;line-height:1.4;margin-top:2px;}
.task .st.done{background:rgba(12,166,120,.12);color:var(--accent2);border:1px solid var(--accent2)}
.task .st.hang{background:rgba(232,89,12,.12);color:var(--warn);border:1px solid var(--warn)}
.task .st.wait{background:rgba(59,91,219,.10);color:var(--accent);border:1px solid var(--accent)}
.task .tt{font-weight:600;font-size:14.5px;margin:0 0 2px;}
.task .td{font-size:13px;color:var(--muted);line-height:1.55;}
.task .td b{color:var(--text)}

/* ============ 目录(Mooncake:右侧固定 + 移动端 tocbar) ============ */
h2,h3{scroll-margin-top:64px}
.toc{position:fixed;top:22px;right:22px;width:240px;max-height:calc(100vh - 44px);
  display:flex;flex-direction:column;background:var(--card);border:1px solid var(--line);
  border-radius:12px;box-shadow:var(--shadow);padding:12px 10px 12px 12px;font-size:12.5px;z-index:50;}
.toc .tt{font-size:11px;font-weight:700;letter-spacing:.6px;color:var(--muted);text-transform:uppercase;padding:0 8px 7px;}
.toc .toc-links{overflow-y:auto;margin-right:-6px;padding-right:6px;}
.toc a{display:block;padding:3px 8px;border-radius:6px;color:var(--text);text-decoration:none;
  line-height:1.5;border-left:2px solid transparent;}
.toc a:hover{background:var(--chip-bg);color:var(--accent)}
.toc a.active{background:var(--chip-bg);color:var(--accent);border-left-color:var(--accent);font-weight:600}
.toc a.l2{padding-left:20px;color:var(--muted);font-size:12px}
.toc a.l2.active{color:var(--accent)}
@media(max-width:1600px){.toc{display:none}}
.tocbar{position:sticky;top:0;z-index:40;display:none;background:rgba(246,247,251,.94);
  backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);border-bottom:1px solid var(--line);}
@media (prefers-color-scheme: dark){.tocbar{background:rgba(14,17,23,.94)}}
@media(max-width:1600px){.tocbar{display:block}}
.tocbar details{max-height:44vh;overflow-y:auto;padding:4px 30px 10px;}
.tocbar summary{cursor:pointer;font-weight:700;font-size:13px;color:var(--accent);padding:8px 0 4px;list-style:none;}
.tocbar summary::-webkit-details-marker{display:none}
.tocbar summary::after{content:"▾";margin-left:8px;color:var(--muted);font-size:12px}
.tocbar details[open] summary::after{content:"▴"}
.tocbar a{display:block;padding:3px 8px;border-radius:6px;color:var(--text);text-decoration:none;font-size:12.5px;}
.tocbar a:hover{background:var(--chip-bg);color:var(--accent)}
.tocbar a.l2{padding-left:20px;color:var(--muted);font-size:12px}
.tocbar a.active{color:var(--accent);font-weight:600}

/* ---- 图表/论文版(full deck)兼容类 ---- */
.section-label{color:var(--accent);font-size:11.5px;font-weight:800;letter-spacing:.18em;
  text-transform:uppercase;margin:0 0 8px;display:flex;align-items:center;gap:10px;}
.section-label::after{content:"";flex:1;height:1px;background:var(--line);max-width:120px}
h3 .sub{font-family:var(--mono);font-size:11.5px;letter-spacing:.08em;color:var(--faint);
  text-transform:uppercase;margin-right:8px}
.lvl{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);
  font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  border-left:3px solid var(--accent);padding:1px 8px 1px 10px;color:var(--accent);margin-right:6px;}
.lvl .ic{font-size:10px;line-height:1}
.lvl.flag{color:var(--accent2);border-color:var(--accent2)}
.lvl.plan{color:var(--info);border-color:var(--info)}
.lvl.rk{color:var(--serious);border-color:var(--serious)}
.lvl.ok{color:var(--good);border-color:var(--good)}
.lvl.wh{color:var(--faint);border-color:var(--faint)}
.chart-card{min-height:340px;display:flex;flex-direction:column}
.chart-card h4{margin-bottom:8px}
.chart-wrap{height:280px;position:relative;flex:1}
.g4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.lead{color:var(--muted);font-size:16px;margin-bottom:14px}
td.num{text-align:right;font-family:var(--mono);font-size:13px}
td.best{color:var(--accent2);font-weight:700}
ul.plain{padding-left:22px;margin:8px 0;color:var(--muted)}
ul.plain li{margin:4px 0}
.katex{font-size:1.06em}
footer{margin-top:52px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3">

/* ============ 目录自动构建 + scrollspy(Mooncake) ============ */
(function () {
  function build() {
    var conts = document.querySelectorAll('.toc-links');
    conts.forEach(function (c) { c.innerHTML = ''; });
    var links = [];
    function add(h, l2) {
      if (!h.id) {
        var n = Math.floor(Math.random() * 1e6);
        h.id = 'sec-auto-' + n;
      }
      var a = document.createElement('a');
      a.href = '#' + h.id;
      var txt = h.textContent.replace(/^\s*[§✦]\s*\d*\s*/, '').replace(/·\s*.*$/, '').trim();
      if (txt.length > 18) txt = txt.slice(0, 18) + '…';
      a.textContent = txt;
      if (l2) a.className = 'l2';
      a.dataset.sec = h.id;
      conts.forEach(function (c) { c.appendChild(a.cloneNode(true)); });
      links.push(a);
    }
    document.querySelectorAll('.wrap h2').forEach(function (h) { add(h, false); });
    document.querySelectorAll('.wrap h3.toc3').forEach(function (h) { add(h, true); });
    var map = {};
    links.forEach(function (l) { map[l.dataset.sec] = l; });
    if (!('IntersectionObserver' in window)) return;
    var heads = Array.prototype.slice.call(document.querySelectorAll('.wrap h2, .wrap h3.toc3'));
    var obs = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (l) { l.classList.remove('active'); });
          var m = map[e.target.id];
          if (m) m.classList.add('active');
        }
      });
    }, { rootMargin: '-15% 0px -75% 0px' });
    heads.forEach(function (h) { obs.observe(h); });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
})();


/* ============ 图表数据(换真数据只改这里;来源见各图 note) ============ */
var DATA = {
  paperAcc: {          // §1.3:论文 Table 1 RULER AVG
    labels:["DSA","Reuse","Refine"],
    ds:[94.30,93.22,94.24],
    glm:[95.19,94.15,95.17],
  },
  paperSpeed: {        // §1.3:论文实测加速(256K prefill indexer;128K 端到端)
    labels:["indexer\nReuse\n(256K)","indexer\nRefine\n(256K)","端到端\nRefine\n(128K)"],
    speed:[4.77,3.87,1.61],
  },
  evidence: {          // §2:14 篇按五问分布 + 核查状态
    labels:["Q1 时间局部性","Q2 稀疏结构","Q3 draft 置信度","Q4 跨层相似","Q5 增量维护"],
    counts:[2,5,2,2,3],
    status:["done","done","done","risk","whitespace"],
  },
  blockers: {          // §4:16 个阻断点按类别
    labels:["标量 D2H 同步","动态索引+同步","数据依赖分支","动态分配/循环"],
    counts:[4,2,5,5],
    total:16,
  },
  iou: {               // §5:合成重排测试(已验证)
    labels:["step1\n[A,B,C]→[B,A,D]","step2\n→[A,D,B]"],
    matched:[2,3],
    expected:[3,3],
    matchRate:0.8333,
  },
  decay: {             // §5:位置衰减预期形态(待 P1,非实测)
    labels:["pos 0\ntarget","pos 1\ndraft","pos 2\ndraft","pos 3\ndraft"],
    qualityLoss:[0.0,0.5,0.8,1.0],
    useProb:[1.0,0.72,0.50,0.31],
  },
};

/* ============ Chart.js 渲染(浅/深色自适应) ============ */
function drawCharts(){
  if (typeof Chart === "undefined") {
    document.querySelectorAll("canvas").forEach(function(c){
      var p = c.parentElement;
      if (p) p.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--faint);font-size:13px;border:1px dashed var(--line);border-radius:10px">图表库(Chart.js CDN)未加载,联网刷新后可见</div>';
    });
    return;
  }
  var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var PAL = dark
    ? {accent:"#6d8dff",teal:"#3ddc97",violet:"#ae91ff",amber:"#ffa94d",good:"#3ddc97",red:"#ff8585",grid:"rgba(255,255,255,.10)",text:"#9aa3b8",faint:"#7b8498"}
    : {accent:"#3b5bdb",teal:"#0ca678",violet:"#6a5acd",amber:"#b36a12",good:"#2f7d5a",red:"#8b2e2e",grid:"rgba(28,33,48,.10)",text:"#5b6478",faint:"#8b92a5"};
  Chart.defaults.color = PAL.text;
  Chart.defaults.borderColor = PAL.grid;
  Chart.defaults.font.family = '-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif';

  function mount(id, cfg){
    var el = document.getElementById(id);
    if (!el) return null;
    return new Chart(el, cfg);
  }
  function baseOpts(yTitle){
    return {
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{labels:{usePointStyle:true,pointStyle:"circle",boxWidth:6,boxHeight:6,font:{size:10.5},padding:10}},
        tooltip:{backgroundColor:"#1a1c20",borderColor:"rgba(255,255,255,.25)",borderWidth:1,
                 titleColor:"#eef2f8",bodyColor:"#dfe3e8",padding:10}
      },
      scales:{
        x:{grid:{display:false},ticks:{color:PAL.text,font:{size:11.5}}},
        y:{title:{display:true,text:yTitle,color:PAL.faint,font:{size:11.5}},
           grid:{color:PAL.grid},ticks:{color:PAL.text,font:{size:11.5}}}
      }
    };
  }

  /* §2 文献分布(横向条形,柱色 = 核查状态) */
  var evStatusColor = {done:PAL.good,risk:PAL.amber,whitespace:PAL.faint};
  mount("chartEvidence",{
    type:"bar",
    data:{labels:DATA.evidence.labels,datasets:[{
      label:"文献篇数",data:DATA.evidence.counts,
      backgroundColor:DATA.evidence.status.map(function(s){return evStatusColor[s]||PAL.accent}),
      borderRadius:7,barThickness:30
    }]},
    options:Object.assign(baseOpts("文献篇数"),{indexAxis:"y"})
  });

  /* §4 阻断点类别 */
  mount("chartBlockers",{
    type:"bar",
    data:{labels:DATA.blockers.labels,datasets:[{
      label:"阻断点数",data:DATA.blockers.counts,
      backgroundColor:[PAL.red,PAL.amber,PAL.violet,PAL.teal],
      borderRadius:7,barThickness:34
    }]},
    options:baseOpts("阻断点数")
  });

  /* §5 步间 IoU 合成验证 */
  mount("chartIoU",{
    type:"bar",
    data:{labels:DATA.iou.labels,datasets:[
      {label:"真实 ID 配对匹配数",data:DATA.iou.matched,backgroundColor:PAL.teal,borderRadius:7,barThickness:30},
      {label:"应匹配数(请求数)",data:DATA.iou.expected,backgroundColor:PAL.grid,borderRadius:7,barThickness:30}
    ]},
    options:baseOpts("matched 计数")
  });

  /* §5 位置衰减预期形态(双线) */
  mount("chartDecay",{
    type:"line",
    data:{labels:DATA.decay.labels,datasets:[
      {label:"quality_loss(i)(预期)",data:DATA.decay.qualityLoss,
       borderColor:PAL.red,borderDash:[6,4],backgroundColor:"rgba(139,46,46,.06)",
       tension:.3,fill:true,pointRadius:4},
      {label:"P(use_i)",data:DATA.decay.useProb,
       borderColor:PAL.accent,backgroundColor:"rgba(42,74,127,.06)",
       tension:.3,fill:true,pointRadius:4}
    ]},
    options:baseOpts("相对值(0-1)")
  });
}
drawCharts();

/* ============ 论文 11 图(曲线/柱状;数据逐项来自论文 Table/Figure,见各卡片 note) ============ */
function drawPaperCharts(){
  if (typeof Chart === "undefined") return;
  var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var PAL = dark
    ? {accent:"#6d8dff",teal:"#3ddc97",violet:"#ae91ff",amber:"#ffa94d",grid:"rgba(255,255,255,.10)",text:"#9aa3b8"}
    : {accent:"#3b5bdb",teal:"#0ca678",violet:"#6a5acd",amber:"#b36a12",grid:"rgba(28,33,48,.10)",text:"#5b6478"};
  function lineChart(id, labels, datasets, yTitle){
    var el = document.getElementById(id); if (!el) return;
    new Chart(el, {type:"line", data:{labels:labels, datasets:datasets}, options:{
      responsive:true, maintainAspectRatio:false,
      interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{usePointStyle:true,pointStyle:"circle",boxWidth:6,boxHeight:6,font:{size:10.5},padding:10}},
               tooltip:{backgroundColor:"#1a1c20",borderColor:"rgba(255,255,255,.25)",borderWidth:1,padding:10}},
      scales:{x:{grid:{display:false},ticks:{color:PAL.text,font:{size:11.5}}},
              y:{title:{display:true,text:yTitle,color:PAL.text,font:{size:11.5}},grid:{color:PAL.grid},ticks:{color:PAL.text,font:{size:11.5}}}}
    }});
  }
  function barChart(id, labels, datasets, yTitle){
    var el = document.getElementById(id); if (!el) return;
    new Chart(el, {type:"bar", data:{labels:labels, datasets:datasets}, options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{labels:{usePointStyle:true,pointStyle:"circle",boxWidth:6,boxHeight:6,font:{size:10.5},padding:10}},
               tooltip:{backgroundColor:"#1a1c20",borderColor:"rgba(255,255,255,.25)",borderWidth:1,padding:10}},
      scales:{x:{grid:{display:false},ticks:{color:PAL.text,font:{size:11.5}}},
              y:{title:{display:true,text:yTitle,color:PAL.text,font:{size:11.5}},grid:{color:PAL.grid},ticks:{color:PAL.text,font:{size:11.5}}}}
    }});
  }

  /* Figure 2b: paper reports ranges, not exact intermediate values */
  lineChart("unionChart", ["g=4","g=8","g=16"], [
    {label:"Lower observed range / k", data:[1.30,1.50,1.70], borderColor:PAL.teal, backgroundColor:"rgba(12,166,120,.12)", tension:.35},
    {label:"Upper observed range / k", data:[1.50,1.90,2.40], borderColor:PAL.violet, backgroundColor:"rgba(106,90,205,.12)", tension:.35}
  ], "Group union / k");

  /* Table 1: RULER AVG */
  barChart("rulerAvgChart", ["DeepSeek DSA","DeepSeek Reuse","DeepSeek Refine","GLM DSA","GLM Reuse","GLM Refine"], [{
    label:"RULER AVG", data:[94.30,93.22,94.24,95.19,94.15,95.17],
    backgroundColor:[PAL.accent,PAL.teal,PAL.violet,PAL.accent,PAL.teal,PAL.violet], borderRadius:7
  }], "Accuracy (%)");

  /* Table 1: RULER length curves (DeepSeek-V3.2) */
  lineChart("rulerLengthChart", ["4K","8K","16K","32K","64K","128K"], [
    {label:"DSA", data:[96.41,95.71,96.12,95.77,91.32,90.45], borderColor:PAL.accent, tension:.3},
    {label:"PIVOT-Reuse", data:[96.03,95.99,95.48,95.30,88.71,87.96], borderColor:PAL.teal, tension:.3},
    {label:"PIVOT-Refine", data:[96.41,95.86,96.22,95.81,90.47,90.40], borderColor:PAL.violet, tension:.3}
  ], "Accuracy (%)");

  /* Figure 4a: prefill indexer speedup */
  lineChart("prefillChart", ["4K","8K","16K","32K","64K","128K","256K"], [
    {label:"PIVOT-Reuse", data:[1.00,1.00,2.56,3.15,3.46,4.03,4.77], borderColor:PAL.teal, tension:.3},
    {label:"PIVOT-Refine", data:[1.00,0.55,0.93,1.70,2.19,2.86,3.87], borderColor:PAL.violet, tension:.3}
  ], "Speedup over DSA");

  /* Figure 4b: decode indexer speedup */
  lineChart("decodeChart", ["4K","8K","16K","32K","64K","128K","256K"], [
    {label:"PIVOT-Reuse", data:[1.00,0.93,1.40,2.05,2.85,3.16,3.42], borderColor:PAL.teal, tension:.3},
    {label:"PIVOT-Refine", data:[1.00,1.00,1.40,2.05,2.85,3.16,3.42], borderColor:PAL.violet, tension:.3}
  ], "Speedup over DSA");

  /* End-to-end speedup (128K peak 1.61x) */
  lineChart("e2eChart", ["8K","16K","32K","64K","128K"], [
    {label:"PIVOT-Reuse", data:[1.09,1.13,1.20,1.35,1.61], borderColor:PAL.teal, tension:.3},
    {label:"PIVOT-Refine", data:[0.97,1.01,1.07,1.21,1.45], borderColor:PAL.violet, tension:.3}
  ], "End-to-end speedup");

  /* Table 2 / App Table 6: proxy aggregation */
  barChart("poolChart", ["Mean","First","Last"], [{
    label:"128K RULER · Refine", data:[90.40,77.62,80.01],
    backgroundColor:[PAL.teal,PAL.amber,PAL.violet], borderRadius:7
  }], "Accuracy (%)");

  /* Table 3 / App Table 8: group size */
  lineChart("groupChart", ["g=4","g=6","g=8","g=16"], [{
    label:"128K RULER · Refine", data:[90.40,85.63,82.90,72.70],
    borderColor:PAL.violet, backgroundColor:"rgba(106,90,205,.12)", tension:.35, fill:true
  }], "Accuracy (%)");

  /* Table 4 / App Table 7: candidate budget */
  barChart("budgetChart", ["3072","4096","6144","8192"], [{
    label:"128K RULER · Refine", data:[87.31,90.40,89.85,90.55],
    backgroundColor:[PAL.amber,PAL.teal,PAL.accent,PAL.violet], borderRadius:7
  }], "Accuracy (%)");

  /* Table 5 / App Table 9: deployment stage */
  barChart("stageChart", ["DSA","Prefill only","Decode only","Both"], [{
    label:"DeepSeek-V3.2 · Refine RULER AVG", data:[94.30,94.13,94.27,94.24],
    backgroundColor:[PAL.accent,PAL.teal,PAL.violet,PAL.amber], borderRadius:7
  }], "Accuracy (%)");

  /* App Figure 5: locality decay endpoints */
  lineChart("localityChart", ["Adjacent","Full 128K span"], [
    {label:"L3 · Shallow", data:[0.89,0.23], borderColor:PAL.teal, tension:.2},
    {label:"L30 · Mid", data:[0.84,0.07], borderColor:PAL.violet, tension:.2},
    {label:"L58 · Deep", data:[0.90,0.07], borderColor:PAL.amber, tension:.2}
  ], "Shared top-k fraction");
}
drawPaperCharts();

/* KaTeX 公式渲染 */
document.addEventListener("DOMContentLoaded", function(){
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        {left:"\\(", right:"\\)", display:false},
        {left:"$$", right:"$$", display:true}
      ],
      ignoredTags: ["script","noscript","style","textarea","pre","code","annotation","annotation-xml"]
    });
  }
});

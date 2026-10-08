"""Blind labeling page for the golden evaluation set, served by the calibration tool's local server (127.0.0.1 only).

Unlike the calibration page there is no suggestion panel and no suggestion route: the labeler sees the conversation, the
provisional taxonomy reference and the guide, never a candidate intent, cluster, historical label or any agent output.
"""

from __future__ import annotations

from typing import Any

from evaluation.golden_eval import GoldenLabelStore, guide_markdown
from evaluation.labeling_ui import INDEX_PAGE, serve

_STYLE = INDEX_PAGE[INDEX_PAGE.index("<style>") : INDEX_PAGE.index("</style>") + len("</style>")]

GOLDEN_INDEX_PAGE = (
    r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden evaluation labeling</title>
"""
    + _STYLE
    + r"""
</head>
<body>
<header>
  <div class="row">
    <h1>Golden evaluation labeling</h1>
    <strong id="labelled">0 / 0 labelled</strong>
    <div class="bar" title="labelled cases"><div id="barfill"></div></div>
    <span class="meta" id="viewing"></span>
    <a href="/guide" target="_blank" rel="noopener">Labeling guide</a>
  </div>
  <div class="row" style="margin-top:.5rem">
    <button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button>
    <button id="nextopen">Next unlabelled</button>
    <select id="jump" style="width:auto;min-width:210px" aria-label="Jump to case"></select>
    <span class="meta">Ctrl+Enter = save &amp; next &middot; Alt+&larr;/&rarr; = previous/next</span>
  </div>
</header>
<main>
  <section>
    <div class="card"><h2>Conversation <span class="meta" id="casemeta"></span></h2><div id="conv"></div>
      <div class="note">Oldest message first. Customer messages are blue, brand replies grey, other operators amber. Brand replies are what happened historically, not the expected label.</div></div>
  </section>
  <aside><div class="sticky">
    <div class="card"><h2>Your label <span id="status" class="chip unlabelled">unlabelled</span></h2>
      <label for="intent">Intent (<code>gold_intent</code>)</label>
      <select id="intent"></select><input type="text" id="newname" placeholder="new_intent_name (snake_case)" hidden style="margin-top:.4rem"><div class="err" id="e_gold_intent"></div>
      <label>Should escalate to a human? (<code>gold_should_escalate</code>)</label><div class="radios" id="escalate"></div><div class="err" id="e_gold_should_escalate"></div>
      <label for="restype">Resolution type (<code>gold_resolution_type</code>)</label><select id="restype"></select><div class="err" id="e_gold_resolution_type"></div>
      <label>Confidence (<code>gold_confidence</code>, optional)</label><div class="radios" id="confidence"></div><div class="err" id="e_gold_confidence"></div>
      <label for="notes">Notes (<code>human_notes</code>; required for NEW:)</label><textarea id="notes" placeholder="why NEW: &middot; second request &middot; hard boundary &middot; anything odd"></textarea><div class="err" id="e_human_notes"></div>
      <div id="msg" class="msg"></div>
      <div class="actions"><button class="primary" id="savenext">Save &amp; next</button><button id="save">Save</button><button class="danger" id="clear">Clear label</button></div>
      <div class="note" style="margin-top:.5rem">Every save is written to the golden CSV immediately and audited. Nothing is ever filled in for you.</div>
    </div>
    <div class="card"><h2 id="refheading"></h2><div id="ref"></div>
      <details><summary>Should escalate: guidance</summary><div id="escguide"></div></details>
      <details><summary>Resolution types</summary><div id="restypes"></div></details>
      <details><summary>Field definitions</summary><div id="fielddefs"></div></details></div>
  </div></aside>
</main>
<script>
const TOKEN="__TOKEN__";
const $=id=>document.getElementById(id);
let state=null, vocab=null, idx=0, cur=null, dirty=false, busy=false;
async function api(path, method="GET", body){
  const r=await fetch(path,{method,headers:{"X-Label-Token":TOKEN,"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined});
  let data={}; try{data=await r.json();}catch(e){}
  if(!r.ok) throw {status:r.status,data}; return data;
}
function el(tag,cls,text){const e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
function setMsg(t,ok){const m=$("msg"); m.textContent=t||""; m.className="msg "+(t?(ok?"ok":"bad"):"");}
function clearErrors(){document.querySelectorAll(".err").forEach(e=>e.textContent="");}
function fillSelect(sel,values,placeholder){sel.replaceChildren(); const p=el("option",null,placeholder); p.value=""; sel.appendChild(p); values.forEach(v=>{const o=el("option",null,v); o.value=v; sel.appendChild(o);});}
function radios(box,name,values){values.forEach(v=>{const l=el("label"); const i=document.createElement("input"); i.type="radio"; i.name=name; i.value=v; l.appendChild(i); l.appendChild(document.createTextNode(v)); box.appendChild(l);});}
function radioValue(name){const r=document.querySelector("input[name="+name+"]:checked"); return r?r.value:"";}
function setRadio(name,v){document.querySelectorAll("input[name="+name+"]").forEach(r=>r.checked=(r.value===v));}
function updateHeader(){
  const c=state.counts; $("labelled").textContent=c.labelled+" / "+c.total+" labelled";
  $("barfill").style.width=(100*c.labelled/c.total)+"%";
  $("viewing").textContent="Case "+(idx+1)+" / "+c.total+(c.partial?" \u00b7 "+c.partial+" partial":"");
  [...$("jump").options].forEach((o,i)=>{const s=state.cases[i].status; o.textContent=(s==="labelled"?"\u2713 ":s==="partial"?"\u25d0 ":"\u25cb ")+String(i+1).padStart(3,"0")+"  "+state.cases[i].case_id;});
}
function setStatus(s){const c=$("status"); c.textContent=s; c.className="chip "+s;}
function readForm(){
  let intent=$("intent").value; if(intent==="__new__") intent="NEW:"+$("newname").value.trim();
  return {gold_intent:intent,gold_should_escalate:radioValue("escalate"),gold_resolution_type:$("restype").value,gold_confidence:radioValue("confidence"),human_notes:$("notes").value};
}
function writeForm(l){
  if(l.gold_intent.startsWith("NEW:")){$("intent").value="__new__"; $("newname").hidden=false; $("newname").value=l.gold_intent.slice(4);}
  else{$("intent").value=vocab.intents.includes(l.gold_intent)?l.gold_intent:""; $("newname").hidden=true; $("newname").value="";}
  setRadio("escalate",l.gold_should_escalate); $("restype").value=l.gold_resolution_type; setRadio("confidence",l.gold_confidence); $("notes").value=l.human_notes;
}
function render(){
  const conv=$("conv"); conv.replaceChildren();
  cur.turns.forEach(t=>{const cls=t.role==="CUSTOMER"?"customer":t.role==="AGENT"?"agent":"other";
    const d=el("div","turn "+cls); const w=el("div","who",t.role==="OTHER-AGENT"?"Other operator":t.role==="AGENT"?"Brand":"Customer"); w.appendChild(el("span","id","#"+t.tweet_id)); d.appendChild(w); d.appendChild(el("div",null,t.text)); conv.appendChild(d);});
  $("casemeta").textContent="\u00b7 "+cur.case_id+" \u00b7 first message "+(cur.first_timestamp||"");
  writeForm(cur.labels); setStatus(cur.status); clearErrors(); setMsg(cur.status==="partial"?"This case is partly labelled. Complete intent, escalation and resolution type.":"",false);
  if(cur.provenance==="assistant_draft") setMsg("AI-assistant draft, not reviewed yet: check every field, then Save to confirm it or correct it.",false);
  $("jump").value=String(idx); dirty=false; updateHeader(); window.scrollTo({top:0});
}
async function load(i){ idx=Math.max(0,Math.min(state.cases.length-1,i)); cur=await api("/api/case/"+state.cases[idx].case_id); render(); }
function okToLeave(){
  if(dirty) return confirm("You have unsaved changes on this case. Leave without saving?");
  return true;
}
async function go(i){ if(busy||i===idx||i<0||i>=state.cases.length) return; if(!okToLeave()) return; await load(i); }
function nextOpen(from){const n=state.cases.length; for(let k=1;k<=n;k++){const j=(from+k)%n; if(state.cases[j].status!=="labelled") return j;} return -1;}
async function save(advance){
  if(busy) return; busy=true; clearErrors(); setMsg("",true);
  try{
    const res=await api("/api/save/"+cur.case_id,"POST",readForm());
    state.counts=res.counts; state.cases[idx].status="labelled"; cur.status="labelled"; dirty=false; setStatus("labelled"); updateHeader();
    setMsg(res.saved?"Saved.":"No changes to save.",true);
    if(advance){busy=false; const j=nextOpen(idx); if(j>=0){await load(j);} else {setMsg("All cases are labelled.",true);}}
  }catch(e){
    if(e.status===422&&e.data.errors){Object.entries(e.data.errors).forEach(([k,v])=>{const t=$("e_"+k); if(t)t.textContent=v;}); setMsg("Not saved: fix the highlighted fields.",false);}
    else setMsg("Not saved: "+((e.data&&e.data.error)||"unexpected error"),false);
  }finally{busy=false;}
}
function buildReference(ref){
  $("refheading").textContent=ref.heading;
  const root=$("ref"); root.replaceChildren();
  const lst=(parent,title,items)=>{if(!items||!items.length)return; const w=el("div","kv"); w.appendChild(el("b",null,title)); const u=el("ul","tight"); items.forEach(x=>u.appendChild(el("li",null,x))); w.appendChild(u); parent.appendChild(w);};
  ref.intents.forEach(c=>{const d=document.createElement("details"); d.appendChild(el("summary",null,c.name+(c.is_fallback?" (fallback)":"")));
    d.appendChild(el("div",null,c.definition));
    lst(d,"Fits when",c.fits_when); lst(d,"Does not fit when",c.does_not_fit_when);
    lst(d,"Often confused with",c.confusable_intents.map(x=>x.intent+(x.distinguishing_note?": "+x.distinguishing_note:"")));
    lst(d,"Examples (train split)",c.historical_examples.map(e=>e.text));
    root.appendChild(d);});
  const kvs=(box,obj)=>{box.replaceChildren(); Object.entries(obj).forEach(([k,v])=>{const d=el("div","kv"); d.appendChild(el("b",null,k)); d.appendChild(document.createTextNode(v)); box.appendChild(d);});};
  kvs($("restypes"),ref.resolution_types); kvs($("fielddefs"),ref.label_definitions);
  const g=$("escguide"); g.replaceChildren(); g.appendChild(el("div",null,ref.label_definitions.gold_should_escalate));
  lst(g,"Usually leans towards yes (decide case by case)",ref.escalation_guidance);
}
async function init(){
  const [st,ref]=await Promise.all([api("/api/state"),api("/api/reference")]); state=st; vocab=st.vocab;
  const names=[...vocab.intents].sort((a,b)=>a.localeCompare(b));
  fillSelect($("intent"),names,"\u2014 choose an intent \u2014"); const n=el("option",null,"NEW:\u2026 none fits (explain in notes)"); n.value="__new__"; $("intent").appendChild(n);
  fillSelect($("restype"),vocab.resolution_types,"\u2014 choose \u2014");
  radios($("escalate"),"escalate",vocab.should_escalate); radios($("confidence"),"confidence",vocab.confidence);
  const jump=$("jump"); state.cases.forEach((c,i)=>{const o=document.createElement("option"); o.value=String(i); jump.appendChild(o);});
  buildReference(ref);
  document.querySelectorAll("#intent,#newname,#restype,#notes,input[type=radio]").forEach(e=>e.addEventListener("input",()=>{dirty=true; setMsg("",true);}));
  $("intent").addEventListener("change",()=>{$("newname").hidden=($("intent").value!=="__new__");});
  $("prev").onclick=()=>go(idx-1); $("next").onclick=()=>go(idx+1);
  $("nextopen").onclick=()=>{const j=nextOpen(idx); if(j<0){setMsg("Every case is labelled.",true);} else go(j);};
  jump.onchange=async()=>{const j=parseInt(jump.value,10); jump.value=String(idx); await go(j);};
  $("save").onclick=()=>save(false); $("savenext").onclick=()=>save(true);
  $("clear").onclick=async()=>{ if(!confirm("Remove the saved label for this case?")) return;
    try{const r=await api("/api/clear/"+cur.case_id,"POST",{}); state.counts=r.counts; state.cases[idx].status="unlabelled"; await load(idx);}catch(e){setMsg("Could not clear: "+((e.data&&e.data.error)||""),false);} };
  document.addEventListener("keydown",ev=>{
    if(ev.ctrlKey&&ev.key==="Enter"){ev.preventDefault(); save(true);}
    else if(ev.altKey&&ev.key==="ArrowRight"){ev.preventDefault(); go(idx+1);}
    else if(ev.altKey&&ev.key==="ArrowLeft"){ev.preventDefault(); go(idx-1);}
  });
  window.addEventListener("beforeunload",ev=>{if(dirty){ev.preventDefault(); ev.returnValue="";}});
  await load(state.first_open_index===null?0:state.first_open_index);
}
init().catch(e=>{document.body.textContent="Could not start the labeling tool: "+JSON.stringify(e);});
</script></body></html>
"""
)


def serve_golden(store: GoldenLabelStore, reference: dict[str, Any], port: int = 8766, open_browser: bool = True) -> None:
    serve(store, reference, guide_markdown(), host="127.0.0.1", port=port, open_browser=open_browser, index_page=GOLDEN_INDEX_PAGE, with_suggestion=False)

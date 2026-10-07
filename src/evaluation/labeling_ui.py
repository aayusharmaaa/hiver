"""Local web UI for the calibration labeling (standard library only, bound to 127.0.0.1).

The page shows one case at a time. The system suggestion is fetched only when the reviewer clicks *Reveal* (and that is audited).
All writes go through `LabelStore`, which can only change the five human_* cells of the calibration CSV.
"""

from __future__ import annotations

import html
import json
import logging
import re
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from evaluation.labeling_store import LabelStore, LabelStoreError, LabelValidationError

logger = logging.getLogger("label_taxonomy_calibration")
CASE_ID = re.compile(r"^case_\d+$")
MAX_BODY = 64 * 1024
DRAIN_LIMIT = 1024 * 1024


def _inline(text: str) -> str:
    s = html.escape(text, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)


def md_to_html(md: str) -> str:
    """Tiny markdown subset (headings, lists, quotes, code fences, bold/italic/code) for the guide page."""
    out: list[str] = []
    mode: str | None = None
    para: list[str] = []

    def close() -> None:
        nonlocal mode
        if para:
            out.append("<p>" + " ".join(para) + "</p>")
            para.clear()
        if mode in ("ul", "ol", "quote"):
            out.append(f"</{'blockquote' if mode == 'quote' else mode}>")
        mode = None if mode != "code" else mode

    for raw in md.split("\n"):
        line = raw.rstrip()
        if line.startswith("```"):
            if mode == "code":
                out.append("</pre>")
                mode = None
            else:
                close()
                out.append("<pre>")
                mode = "code"
            continue
        if mode == "code":
            out.append(html.escape(line, quote=False))
            continue
        if not line.strip():
            close()
            continue
        if line.startswith("---"):
            close()
            out.append("<hr>")
            continue
        h = re.match(r"^(#{1,4}) (.*)$", line)
        if h:
            close()
            level = len(h.group(1))
            out.append(f"<h{level}>{_inline(h.group(2))}</h{level}>")
            continue
        if line.startswith("> "):
            if mode != "quote":
                close()
                out.append("<blockquote>")
                mode = "quote"
            out.append(f"<div>{_inline(line[2:])}</div>")
            continue
        b = re.match(r"^- (.*)$", line)
        n = re.match(r"^\d+\. (.*)$", line)
        if b or n:
            want = "ul" if b else "ol"
            if mode != want:
                close()
                out.append(f"<{want}>")
                mode = want
            out.append(f"<li>{_inline((b or n).group(1))}</li>")
            continue
        if mode in ("ul", "ol", "quote"):
            close()
        para.append(_inline(line.strip()))
    close()
    return "\n".join(out)


GUIDE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Labeling guide</title>
<style>body{font:15px/1.55 system-ui,Segoe UI,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1f2933}
code{background:#eef1f5;padding:.1em .3em;border-radius:3px;font-size:.92em}pre{background:#eef1f5;padding:.7rem;border-radius:6px}
blockquote{border-left:4px solid #c5ced8;margin:.5rem 0;padding:.1rem .8rem;color:#3e4c59;background:#f8fafc}h2{margin-top:2rem;border-bottom:1px solid #d9e0e7;padding-bottom:.2rem}
h3{margin-top:1.4rem}a{color:#1d4ed8}</style></head><body><p><a href="javascript:window.close()">close this tab</a></p>__BODY__</body></html>"""

INDEX_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taxonomy calibration labeling</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--ink:#1f2933;--mute:#616e7c;--line:#d9e0e7;--cust:#e6f0ff;--cust-b:#9db9e8;--agent:#eef2f5;--agent-b:#b8c4cf;--other:#fff5e0;--other-b:#e3c27a;--ok:#1a7f4b;--warn:#b45309;--bad:#b91c1c;--accent:#1d4ed8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.5 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:.6rem 1rem}
header .row{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}h1{font-size:1rem;margin:0;font-weight:650}
.bar{flex:1;min-width:160px;height:10px;background:#e5eaf0;border-radius:6px;overflow:hidden}.bar>div{height:100%;background:var(--ok);width:0}
button,select,input,textarea{font:inherit}button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:.35rem .7rem;cursor:pointer}
button:hover{background:#f1f5f9}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.primary:hover{background:#1e40af}
button.danger{color:var(--bad)}button:disabled{opacity:.5;cursor:not-allowed}
main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:1rem;padding:1rem;max-width:1400px;margin:0 auto}
@media(max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem;margin-bottom:1rem}
.card h2{font-size:.95rem;margin:0 0 .6rem;font-weight:650}.meta{color:var(--mute);font-size:.85rem}
.turn{border:1px solid;border-radius:10px;padding:.5rem .75rem;margin:.45rem 0;max-width:92%;white-space:pre-wrap;overflow-wrap:anywhere}
.turn .who{font-size:.74rem;font-weight:650;letter-spacing:.03em;text-transform:uppercase;margin-bottom:.15rem}.turn .id{color:var(--mute);font-weight:400;margin-left:.4rem}
.turn.customer{background:var(--cust);border-color:var(--cust-b);margin-right:auto}.turn.agent{background:var(--agent);border-color:var(--agent-b);margin-left:auto}
.turn.other{background:var(--other);border-color:var(--other-b);margin-left:auto}
.chip{display:inline-block;padding:.1rem .5rem;border-radius:999px;font-size:.78rem;font-weight:600}
.chip.labelled{background:#dcf5e7;color:var(--ok)}.chip.unlabelled{background:#eef1f5;color:var(--mute)}.chip.partial{background:#fdecc8;color:var(--warn)}
label{display:block;font-weight:600;margin:.7rem 0 .2rem;font-size:.88rem}select,textarea,input[type=text]{width:100%;padding:.4rem .5rem;border:1px solid var(--line);border-radius:6px;background:#fff}
textarea{min-height:92px;resize:vertical}.radios{display:flex;gap:1rem}.radios label{font-weight:500;margin:0;display:flex;gap:.3rem;align-items:center}
.err{color:var(--bad);font-size:.82rem;min-height:1em}.msg{margin:.5rem 0;font-size:.88rem}.msg.ok{color:var(--ok)}.msg.bad{color:var(--bad)}
.actions{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.9rem}details{border:1px solid var(--line);border-radius:8px;padding:.35rem .6rem;margin:.4rem 0;background:#fafbfc}
summary{cursor:pointer;font-weight:600}.kv{margin:.25rem 0}.kv b{display:inline-block;min-width:9.5rem;color:var(--mute);font-weight:600}
ul.tight{margin:.2rem 0 .2rem 1.1rem;padding:0}code{background:#eef1f5;padding:.08em .3em;border-radius:3px;font-size:.9em}
.sug{border-left:4px solid var(--warn);padding-left:.8rem}.note{font-size:.82rem;color:var(--mute)}
.sticky{position:sticky;top:5.2rem}
</style></head>
<body>
<header>
  <div class="row">
    <h1>Taxonomy calibration labeling</h1>
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
      <div class="note">Oldest message first. Customer messages are blue, agent replies grey, other operators amber.</div></div>
    <div class="card sug"><h2>System suggestion <span class="meta">(hidden until you ask; decide first)</span></h2>
      <button id="reveal">Reveal system suggestion</button><div id="sug" hidden></div></div>
  </section>
  <aside><div class="sticky">
    <div class="card"><h2>Your label <span id="status" class="chip unlabelled">unlabelled</span></h2>
      <label for="intent">Intent (<code>human_intent</code>)</label>
      <select id="intent"></select><input type="text" id="newname" placeholder="new_intent_name (snake_case)" hidden style="margin-top:.4rem"><div class="err" id="e_human_intent"></div>
      <label for="restype">Resolution type (<code>human_resolution_type</code>)</label><select id="restype"></select><div class="err" id="e_human_resolution_type"></div>
      <label>Resolved (<code>human_resolved</code>)</label><div class="radios" id="resolved"></div><div class="err" id="e_human_resolved"></div>
      <label for="esc">Escalation signal (<code>human_escalation_signal</code>)</label><select id="esc"></select><div class="err" id="e_human_escalation_signal"></div>
      <label for="notes">Notes (<code>human_notes</code>)</label><textarea id="notes" placeholder="also: &lt;intent&gt; &middot; real request: ... &middot; new intent: name: definition"></textarea><div class="err" id="e_human_notes"></div>
      <div id="msg" class="msg"></div>
      <div class="actions"><button class="primary" id="savenext">Save &amp; next</button><button id="save">Save</button><button class="danger" id="clear">Clear label</button></div>
      <div class="note" style="margin-top:.5rem">Saving checkpoints to the calibration CSV immediately. Nothing is ever filled in for you.</div>
    </div>
    <div class="card"><h2>Intent reference <span class="meta">(all intents, alphabetical)</span></h2><div id="ref"></div>
      <details><summary>Resolution types</summary><div id="restypes"></div></details>
      <details><summary>Escalation signals</summary><div id="escs"></div></details>
      <div class="note" id="disclaimer"></div></div>
  </div></aside>
</main>
<script>
const TOKEN="__TOKEN__";
const $=id=>document.getElementById(id);
let state=null, vocab=null, REF=null, idx=0, cur=null, dirty=false, busy=false;
async function api(path, method="GET", body){
  const r=await fetch(path,{method,headers:{"X-Label-Token":TOKEN,"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined});
  let data={}; try{data=await r.json();}catch(e){}
  if(!r.ok) throw {status:r.status,data}; return data;
}
function el(tag,cls,text){const e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
function setMsg(t,ok){const m=$("msg"); m.textContent=t||""; m.className="msg "+(t?(ok?"ok":"bad"):"");}
function clearErrors(){document.querySelectorAll(".err").forEach(e=>e.textContent="");}
function fillSelect(sel,values,placeholder){sel.replaceChildren(); const p=el("option",null,placeholder); p.value=""; sel.appendChild(p); values.forEach(v=>{const o=el("option",null,v); o.value=v; sel.appendChild(o);});}
function updateHeader(){
  const c=state.counts; $("labelled").textContent=c.labelled+" / "+c.total+" labelled";
  $("barfill").style.width=(100*c.labelled/c.total)+"%";
  $("viewing").textContent="Viewing case "+(idx+1)+" / "+c.total+(c.partial?" \u00b7 "+c.partial+" partial":"");
  [...$("jump").options].forEach((o,i)=>{const s=state.cases[i].status; o.textContent=(s==="labelled"?"\u2713 ":s==="partial"?"\u25d0 ":"\u25cb ")+String(i+1).padStart(3,"0")+"  "+state.cases[i].case_id;});
}
function setStatus(s){const c=$("status"); c.textContent=s; c.className="chip "+s;}
function readForm(){
  let intent=$("intent").value; if(intent==="__new__") intent="NEW:"+$("newname").value.trim();
  const r=document.querySelector("input[name=resolved]:checked");
  return {human_intent:intent,human_resolution_type:$("restype").value,human_resolved:r?r.value:"",human_escalation_signal:$("esc").value,human_notes:$("notes").value};
}
function writeForm(l){
  const known=vocab.intents.includes(l.human_intent);
  if(l.human_intent.startsWith("NEW:")){$("intent").value="__new__"; $("newname").hidden=false; $("newname").value=l.human_intent.slice(4);}
  else{$("intent").value=known?l.human_intent:""; $("newname").hidden=true; $("newname").value="";}
  $("restype").value=l.human_resolution_type; $("esc").value=l.human_escalation_signal; $("notes").value=l.human_notes;
  document.querySelectorAll("input[name=resolved]").forEach(r=>r.checked=(r.value===l.human_resolved));
}
function render(){
  const conv=$("conv"); conv.replaceChildren();
  cur.turns.forEach(t=>{const cls=t.role==="CUSTOMER"?"customer":t.role==="AGENT"?"agent":"other";
    const d=el("div","turn "+cls); const w=el("div","who",t.role==="OTHER-AGENT"?"Other operator":t.role==="AGENT"?"Agent":"Customer"); w.appendChild(el("span","id","#"+t.tweet_id)); d.appendChild(w); d.appendChild(el("div",null,t.text)); conv.appendChild(d);});
  $("casemeta").textContent="\u00b7 "+cur.case_id+" \u00b7 first message "+(cur.first_timestamp||"");
  writeForm(cur.labels); setStatus(cur.status); clearErrors(); setMsg(cur.status==="partial"?"This case has partial labels in the CSV. Complete all four required fields.":"",false);
  $("sug").hidden=true; $("sug").replaceChildren(); $("reveal").hidden=false; $("reveal").textContent=cur.suggestion_revealed?"Reveal system suggestion again":"Reveal system suggestion";
  $("jump").value=String(idx); dirty=false; updateHeader(); window.scrollTo({top:0});
}
async function load(i){
  idx=Math.max(0,Math.min(state.cases.length-1,i)); cur=await api("/api/case/"+state.cases[idx].case_id); render();
}
function okToLeave(){
  if(dirty) return confirm("You have unsaved changes on this case. Leave without saving?");
  if(cur && cur.status!=="labelled") return confirm("This case is not labelled yet. Leave it anyway?");
  return true;
}
async function go(i){ if(busy||i===idx) return; if(!okToLeave()) return; await load(i); }
function nextOpen(from){const n=state.cases.length; for(let k=1;k<=n;k++){const j=(from+k)%n; if(state.cases[j].status!=="labelled") return j;} return -1;}
async function save(advance){
  if(busy) return; busy=true; clearErrors(); setMsg("",true);
  try{
    const res=await api("/api/save/"+cur.case_id,"POST",readForm());
    state.counts=res.counts; state.cases[idx].status="labelled"; cur.status="labelled"; dirty=false; setStatus("labelled"); updateHeader();
    setMsg(res.saved?"Saved.":"No changes to save.",true);
    if(advance){const j=nextOpen(idx); if(j>=0){await load(j);} else if(idx<state.cases.length-1){await load(idx+1);} else {setMsg("All cases are labelled.",true);}}
  }catch(e){
    if(e.status===422&&e.data.errors){Object.entries(e.data.errors).forEach(([k,v])=>{const t=$("e_"+k); if(t)t.textContent=v;}); setMsg("Not saved: fix the highlighted fields.",false);}
    else setMsg("Not saved: "+((e.data&&e.data.error)||"unexpected error"),false);
  }finally{busy=false;}
}
function renderSuggestion(s,card){
  const box=$("sug"); box.replaceChildren();
  const kv=(k,v)=>{const d=el("div","kv"); d.appendChild(el("b",null,k)); d.appendChild(document.createTextNode(v)); box.appendChild(d);};
  kv("Candidate intent",s.candidate_intent); kv("Cluster",s.candidate_cluster||"");
  kv("Runner-up",(s.runner_up_intent||"none")+(s.runner_up_cluster?" ("+s.runner_up_cluster+")":""));
  kv("Cluster margin",s.cluster_margin+" (lower = nearer a cluster boundary; "+Math.round(100*s.margin_percentile_in_sample)+"% of this sample is at or below)");
  kv("Sampled as",s.sampling_stratum+" / "+s.selection_reason);
  kv("Auto resolution",s.auto_resolution_type+", resolved="+s.auto_resolved+", DM redirect="+s.auto_dm_redirect+" (heuristic, not a label)");
  if(s.auto_resolution_summary) kv("Auto summary",s.auto_resolution_summary);
  if(card){
    box.appendChild(el("h2",null,"Candidate intent definition and evidence"));
    box.appendChild(el("div",null,card.definition));
    const lst=(title,items)=>{if(!items||!items.length)return; const d=el("div","kv"); d.appendChild(el("b",null,title)); const u=el("ul","tight"); items.forEach(x=>u.appendChild(el("li",null,x))); d.appendChild(u); box.appendChild(d);};
    lst("Fits when",card.fits_when); lst("Does not fit when",card.does_not_fit_when);
    lst("Often confused with",card.confusable_intents.map(c=>c.intent+(c.evidence?" \u2014 "+c.evidence:"")+(c.distinguishing_note?" ("+c.distinguishing_note+")":"")));
    lst("Agents asked customers for (train)",card.required_information.map(r=>r.item+" \u2014 "+r.evidence));
    lst("Historic agent hand-offs (train)",card.historical_escalation_evidence.map(r=>r.what+" \u2014 "+r.evidence));
    lst("Historical examples (train)",card.historical_examples.map(e=>e.case_id+": "+e.text));
    lst("Near misses (train)",card.near_miss_examples.map(e=>e.case_id+" (closer to "+e.closer_to+"): "+e.text));
    box.appendChild(el("div","note","Evidence from the train split; not a recommendation. Decide from the conversation."));
  }
  box.hidden=false;
}
function buildReference(ref){
  const root=$("ref"); root.replaceChildren();
  ref.intents.forEach(c=>{const d=document.createElement("details"); d.appendChild(el("summary",null,c.name+(c.is_fallback?" (fallback)":"")));
    d.appendChild(el("div",null,c.definition));
    const lst=(title,items)=>{if(!items||!items.length)return; const w=el("div","kv"); w.appendChild(el("b",null,title)); const u=el("ul","tight"); items.forEach(x=>u.appendChild(el("li",null,x))); w.appendChild(u); d.appendChild(w);};
    lst("Fits when",c.fits_when); lst("Does not fit when",c.does_not_fit_when); lst("Often confused with",c.confusable_intents.map(x=>x.intent+(x.distinguishing_note?": "+x.distinguishing_note:"")));
    root.appendChild(d);});
  const kvs=(box,obj)=>{box.replaceChildren(); Object.entries(obj).forEach(([k,v])=>{const d=el("div","kv"); d.appendChild(el("b",null,k)); d.appendChild(document.createTextNode(v)); box.appendChild(d);});};
  kvs($("restypes"),ref.resolution_types); kvs($("escs"),ref.escalation_signals); $("disclaimer").textContent=ref.disclaimer;
}
async function init(){
  const [st,ref]=await Promise.all([api("/api/state"),api("/api/reference")]); state=st; vocab=st.vocab; REF=ref;
  const names=[...vocab.intents].sort((a,b)=>a.localeCompare(b));
  fillSelect($("intent"),names,"\u2014 choose an intent \u2014"); const n=el("option",null,"NEW:\u2026 propose a new intent"); n.value="__new__"; $("intent").appendChild(n);
  fillSelect($("restype"),vocab.resolution_types,"\u2014 choose \u2014"); fillSelect($("esc"),vocab.escalation_signals,"\u2014 choose \u2014");
  const rs=$("resolved"); vocab.resolved.forEach(v=>{const l=el("label"); const i=document.createElement("input"); i.type="radio"; i.name="resolved"; i.value=v; l.appendChild(i); l.appendChild(document.createTextNode(v)); rs.appendChild(l);});
  const jump=$("jump"); state.cases.forEach((c,i)=>{const o=document.createElement("option"); o.value=String(i); jump.appendChild(o);});
  buildReference(ref);
  document.querySelectorAll("#intent,#newname,#restype,#esc,#notes,input[name=resolved]").forEach(e=>e.addEventListener("input",()=>{dirty=true; setMsg("",true);}));
  $("intent").addEventListener("change",()=>{$("newname").hidden=($("intent").value!=="__new__");});
  $("prev").onclick=()=>go(idx-1); $("next").onclick=()=>go(idx+1);
  $("nextopen").onclick=()=>{const j=nextOpen(idx); if(j<0){setMsg("Every case is labelled.",true);} else go(j);};
  jump.onchange=async()=>{const j=parseInt(jump.value,10); jump.value=String(idx); await go(j);};
  $("save").onclick=()=>save(false); $("savenext").onclick=()=>save(true);
  $("clear").onclick=async()=>{ if(!confirm("Remove the saved label for this case?")) return;
    try{const r=await api("/api/clear/"+cur.case_id,"POST",{}); state.counts=r.counts; state.cases[idx].status="unlabelled"; await load(idx);}catch(e){setMsg("Could not clear.",false);} };
  $("reveal").onclick=async()=>{try{const s=await api("/api/suggestion/"+cur.case_id,"POST",{}); const card=REF.intents.find(c=>c.name===s.candidate_intent); renderSuggestion(s,card); $("reveal").hidden=true; cur.suggestion_revealed=true;}catch(e){setMsg("Could not load the suggestion.",false);}};
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


def make_handler(store: LabelStore, reference: dict[str, Any], guide_html: str, token: str, allowed_hosts: set[str]):
    index_html = INDEX_PAGE.replace("__TOKEN__", token).encode("utf-8")
    guide_bytes = GUIDE_PAGE.replace("__BODY__", guide_html).encode("utf-8")
    vocab = store.vocab.as_dict()

    class Handler(BaseHTTPRequestHandler):
        server_version = "CalibrationLabeler/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return (self.headers.get("Host") or "").lower() in allowed_hosts

        def _read_raw(self) -> None:
            # Always consume the request body, even when the answer is an error: closing a socket with unread data
            # resets the connection on Windows and the browser would see a network error instead of our message.
            length = int(self.headers.get("Content-Length") or 0)
            self._declared = length
            self._raw = b""
            remaining = min(length, DRAIN_LIMIT)
            chunks = []
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            self._raw = b"".join(chunks)

        def _route(self, method: str) -> None:
            self._read_raw()
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
                return
            path = urlparse(self.path).path
            try:
                if method == "GET" and path == "/":
                    self._send(200, index_html, "text/html; charset=utf-8")
                elif method == "GET" and path == "/guide":
                    self._send(200, guide_bytes, "text/html; charset=utf-8")
                elif path.startswith("/api/"):
                    self._api(method, path)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except LabelValidationError as exc:
                self._json(422, {"errors": exc.errors})
            except LabelStoreError as exc:
                self._json(409, {"error": str(exc)})
            except Exception:
                logger.exception("unexpected error")
                self._json(500, {"error": "unexpected server error; see the terminal"})

        def _body(self) -> dict[str, Any]:
            if self._declared > MAX_BODY:
                raise LabelStoreError("request too large")
            if self._declared and "json" not in (self.headers.get("Content-Type") or ""):
                raise LabelStoreError("expected JSON")
            try:
                body = json.loads(self._raw or b"{}")
            except ValueError as exc:
                raise LabelStoreError("request body is not valid JSON") from exc
            if not isinstance(body, dict):
                raise LabelStoreError("request body must be a JSON object")
            return body

        def _api(self, method: str, path: str) -> None:
            if not secrets.compare_digest(self.headers.get("X-Label-Token", ""), token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong token"})
                return
            parts = [unquote(p) for p in path.split("/") if p]
            if method == "GET" and parts == ["api", "state"]:
                self._json(200, {**store.state(), "vocab": vocab})
            elif method == "GET" and parts == ["api", "reference"]:
                self._json(200, reference)
            elif len(parts) == 3 and parts[1] in ("case", "suggestion", "save", "clear") and CASE_ID.match(parts[2]):
                kind, case_id = parts[1], parts[2]
                if (kind == "case") != (method == "GET"):
                    self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "wrong method"})
                elif kind == "case":
                    self._json(200, store.get_case(case_id))
                elif kind == "suggestion":
                    self._json(200, store.get_suggestion(case_id))
                elif kind == "save":
                    self._json(200, store.save_label(case_id, self._body()))
                else:
                    self._json(200, store.clear_label(case_id))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_GET(self) -> None:  # noqa: N802
            self._route("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._route("POST")

    return Handler


def serve(store: LabelStore, reference: dict[str, Any], guide_md: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    token = secrets.token_urlsafe(24)
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    handler = make_handler(store, reference, md_to_html(guide_md), token, allowed)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise SystemExit(f"Could not listen on {host}:{port} ({exc}). If another labeling window is open, close it first; two sessions must not run at once.")
    url = f"http://127.0.0.1:{port}/"
    store.start_session()
    logger.info("Labeling tool running at %s  (Ctrl+C to stop; labels are saved on every Save)", url)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped. Progress is already saved in %s", store.csv_path)
    finally:
        server.server_close()

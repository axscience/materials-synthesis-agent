import React, { useEffect, useRef, useState } from "react";
import { api, CampaignSummary, CampaignState } from "./api";

type Msg = { role: "user" | "assistant"; content: string; tools?: { name: string; is_error?: boolean }[] };
type Tab = "chat" | "protocols" | "log";

export function App() {
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const refresh = () => api.listCampaigns().then(setCampaigns).catch(() => {});
  useEffect(() => { refresh(); }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">Discovery Harness</div>
        <button className="new-btn" onClick={() => setCreating(true)}>+ New campaign</button>
        <div className="campaign-list">
          {campaigns.length === 0 && <div className="muted">No campaigns yet.</div>}
          {campaigns.map((c) => (
            <button
              key={c.id}
              className={"campaign-item" + (c.id === selected ? " active" : "")}
              onClick={() => setSelected(c.id)}
            >
              <div className="campaign-name">{c.name}</div>
              <div className="campaign-meta">{c.status}</div>
            </button>
          ))}
        </div>
      </aside>

      <main className="main">
        {creating && (
          <NewCampaign
            onCancel={() => setCreating(false)}
            onCreated={(id) => { setCreating(false); refresh().then(() => setSelected(id)); }}
          />
        )}
        {!creating && selected && <CampaignView key={selected} campaignId={selected} />}
        {!creating && !selected && (
          <div className="empty-main">
            <h1>Design → make → measure</h1>
            <p className="muted">Select a campaign or create one to start a discovery loop.</p>
          </div>
        )}
      </main>
    </div>
  );
}

function NewCampaign({ onCreated, onCancel }: { onCreated: (id: string) => void; onCancel: () => void }) {
  const [name, setName] = useState("");
  const [linkage, setLinkage] = useState("imine");
  const [application, setApplication] = useState("");
  const [metric, setMetric] = useState("crystallinity");
  const [method, setMethod] = useState("PXRD peak area ratio");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const r = await api.createCampaign({
        name, linkage_chemistry: linkage, application,
        metric_name: metric, metric_measurement_method: method,
        functional_groups: [],
      });
      onCreated(r.id);
    } finally { setBusy(false); }
  };

  return (
    <div className="panel form-panel">
      <h2>New campaign</h2>
      <label>Name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="imine-cof-gas-separation" /></label>
      <label>Linkage chemistry
        <select value={linkage} onChange={(e) => setLinkage(e.target.value)}>
          <option value="imine">imine</option>
          <option value="boronate_ester">boronate ester</option>
        </select>
      </label>
      <label>Application<input value={application} onChange={(e) => setApplication(e.target.value)} placeholder="CO2 capture" /></label>
      <label>Objective metric<input value={metric} onChange={(e) => setMetric(e.target.value)} /></label>
      <label>Measurement method<input value={method} onChange={(e) => setMethod(e.target.value)} /></label>
      <div className="row">
        <button className="primary" disabled={busy || !name.trim()} onClick={submit}>Create</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

function CampaignView({ campaignId }: { campaignId: string }) {
  const [tab, setTab] = useState<Tab>("chat");
  const [state, setState] = useState<CampaignState | null>(null);
  const loadState = () => api.getState(campaignId).then(setState).catch(() => {});
  useEffect(() => { loadState(); }, [campaignId]);

  return (
    <div className="campaign-view">
      <nav className="tabs">
        <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
        <button className={tab === "protocols" ? "active" : ""} onClick={() => setTab("protocols")}>
          Protocols {state ? `(${state.protocols.length})` : ""}
        </button>
        <button className={tab === "log" ? "active" : ""} onClick={() => setTab("log")}>Log result</button>
      </nav>
      {tab === "chat" && <Chat campaignId={campaignId} onChanged={loadState} />}
      {tab === "protocols" && <Protocols state={state} />}
      {tab === "log" && <LogResult campaignId={campaignId} state={state} onLogged={loadState} />}
    </div>
  );
}

function Chat({ campaignId, onChanged }: { campaignId: string; onChanged: () => void }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [session, setSession] = useState<string | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", content: text }]);
    setBusy(true);
    try {
      const r = await api.chat(campaignId, text, session);
      setSession(r.session_id);
      setMsgs((m) => [...m, { role: "assistant", content: r.reply || "(no reply)", tools: r.tool_calls }]);
      onChanged();
    } catch (e: any) {
      setMsgs((m) => [...m, { role: "assistant", content: `Error: ${e.message}` }]);
    } finally { setBusy(false); }
  };

  return (
    <div className="chat">
      <div className="thread">
        {msgs.length === 0 && (
          <div className="muted hint">
            Ask the research director to search the literature, set up a parameter space, or suggest
            the next experiment. Every number it reports comes from a tool and carries its uncertainty.
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={"bubble " + m.role}>
            <div className="bubble-content">{m.content}</div>
            {m.tools && m.tools.length > 0 && (
              <div className="tool-chips">
                {m.tools.map((t, j) => (
                  <span key={j} className={"chip" + (t.is_error ? " chip-error" : "")}>{t.name}</span>
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="bubble assistant"><div className="bubble-content muted">Working…</div></div>}
        <div ref={endRef} />
      </div>
      <div className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Message the research director…"
          rows={2}
        />
        <button className="primary" disabled={busy || !input.trim()} onClick={send}>Send</button>
      </div>
    </div>
  );
}

function Protocols({ state }: { state: CampaignState | null }) {
  if (!state) return <div className="panel muted">Loading…</div>;
  if (state.protocols.length === 0)
    return <div className="panel muted">No protocol candidates yet. Ask the director to search the literature or suggest one.</div>;
  return (
    <div className="panel">
      {state.protocols.map((p: any) => (
        <div key={p.id} className="card">
          <div className="card-head">
            <span className="mono">{p.id.slice(0, 8)}</span>
            <span className="badge">{p.source}</span>
          </div>
          <FieldRow label="Building blocks" fv={p.building_blocks} isDict />
          <FieldRow label="Temperature" fv={p.temperature_c} />
          <FieldRow label="Time" fv={p.time_hours} />
          <FieldRow label="Solvent" fv={p.solvent} />
          <FieldRow label="Yield" fv={p.yield_percent} />
        </div>
      ))}
    </div>
  );
}

// Renders a FieldValue (or dict of them) with its citation excerpt -- the provenance surface.
function FieldRow({ label, fv, isDict }: { label: string; fv: any; isDict?: boolean }) {
  if (!fv) return null;
  const one = (v: any, key?: string) => {
    if (!v) return null;
    const cited = v.citation && v.citation.excerpt;
    return (
      <div className="field" key={key}>
        <span className="field-val">{key ? `${key}: ` : ""}{String(v.value)}</span>
        {v.inferred ? <span className="tag-inferred">inferred</span> : cited ? (
          <span className="cite" title={v.citation.excerpt}>❝ cited</span>
        ) : null}
      </div>
    );
  };
  return (
    <div className="field-row">
      <div className="field-label">{label}</div>
      <div className="field-values">
        {isDict ? Object.entries(fv).map(([k, v]) => one(v, k)) : one(fv)}
      </div>
    </div>
  );
}

function LogResult({ campaignId, state, onLogged }: { campaignId: string; state: CampaignState | null; onLogged: () => void }) {
  const [protocolId, setProtocolId] = useState("");
  const [metricName, setMetricName] = useState("crystallinity");
  const [value, setValue] = useState("");
  const [uncertainty, setUncertainty] = useState("");
  const [notes, setNotes] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!protocolId || !value) return;
    setBusy(true); setMsg(null);
    try {
      const metrics = [{ name: metricName, value: parseFloat(value), uncertainty: uncertainty ? parseFloat(uncertainty) : null }];
      await api.logResult(campaignId, protocolId, metrics, notes);
      setMsg("Logged — the result is now in the optimizer's training data and the cross-campaign prior.");
      setValue(""); setUncertainty(""); setNotes("");
      onLogged();
    } catch (e: any) {
      setMsg(`Error: ${e.message}`);
    } finally { setBusy(false); }
  };

  const protocols = state?.protocols || [];
  return (
    <div className="panel form-panel">
      <h2>Log a bench result</h2>
      <p className="muted">This is the “measure” edge of the loop — enter what the bench produced, with its uncertainty.</p>
      <label>Protocol
        <select value={protocolId} onChange={(e) => setProtocolId(e.target.value)}>
          <option value="">Select a protocol…</option>
          {protocols.map((p: any) => <option key={p.id} value={p.id}>{p.id.slice(0, 8)} · {p.source}</option>)}
        </select>
      </label>
      <label>Metric<input value={metricName} onChange={(e) => setMetricName(e.target.value)} /></label>
      <div className="row">
        <label>Value<input value={value} onChange={(e) => setValue(e.target.value)} placeholder="0.72" /></label>
        <label>± Uncertainty<input value={uncertainty} onChange={(e) => setUncertainty(e.target.value)} placeholder="0.04" /></label>
      </div>
      <label>Notes<textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} placeholder="PXRD matches predicted topology" /></label>
      <button className="primary" disabled={busy || !protocolId || !value} onClick={submit}>Log result</button>
      {msg && <div className="form-msg">{msg}</div>}
    </div>
  );
}

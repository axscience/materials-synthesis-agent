// Thin typed wrapper over the harness FastAPI backend. All calls go through the Vite /api proxy.

export interface CampaignSummary {
  id: string;
  name: string;
  status: string;
  updated_at: string;
}

export interface ChatReply {
  reply: string;
  session_id: string;
  tool_calls: { name: string; is_error?: boolean; cost?: number }[];
}

export interface CampaignState {
  protocols: any[];
  experiments: any[];
  suggestions: any[];
  characterizations: any[];
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listCampaigns: () => fetch("/api/campaigns").then(j<CampaignSummary[]>),

  createCampaign: (body: Record<string, unknown>) =>
    fetch("/api/campaigns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j<{ id: string; name: string }>),

  getCampaign: (id: string) => fetch(`/api/campaigns/${id}`).then(j<any>),

  getState: (id: string) => fetch(`/api/campaigns/${id}/state`).then(j<CampaignState>),

  chat: (id: string, message: string, sessionId?: string) =>
    fetch(`/api/campaigns/${id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    }).then(j<ChatReply>),

  logResult: (id: string, protocolId: string, metrics: any[], notes: string) =>
    fetch(`/api/campaigns/${id}/log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ protocol_id: protocolId, metrics, notes }),
    }).then(j<any>),
};

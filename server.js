import express from "express"
import multer from "multer"
import pdfParse from "pdf-parse"
import dotenv from "dotenv"
import { createClient } from "@supabase/supabase-js"
import OpenAI from "openai"

dotenv.config()
const app = express()
app.use(express.json({ limit: "10mb" }))
const upload = multer({ limits: { fileSize: 60 * 1024 * 1024 } })
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY)
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

function chunk(text, size = 1200, overlap = 200) {
  const out = []
  let i = 0
  while (i < text.length) {
    const end = Math.min(i + size, text.length)
    out.push(text.slice(i, end))
    i = end - overlap
    if (i < 0) i = 0
  }
  return out
}

async function embed(input) {
  const r = await openai.embeddings.create({ model: "text-embedding-3-large", input })
  return r.data[0].embedding
}

async function upsertChunks(ns, chunks, meta) {
  for (let i = 0; i < chunks.length; i++) {
    const e = await embed(chunks[i])
    await supabase.from("docs").insert({ namespace: ns, content: chunks[i], metadata: { ...meta, chunk_index: i }, embedding: e })
  }
}

async function vectorSearch(ns, query, k = 6) {
  const e = await embed(query)
  const { data } = await supabase.rpc("match_documents", { query_embedding: e, ns, match_count: k, filter: null })
  return (data || []).map(d => ({ id: d.id, content: d.content, meta: d.metadata, score: d.similarity, kind: "vector" }))
}

async function keywordSearch(ns, query, k = 6) {
  const { data } = await supabase.rpc("keyword_search", { q: query, ns, match_count: k })
  return (data || []).map(d => ({ id: d.id, content: d.content, meta: d.metadata, score: d.rank || 0, kind: "keyword" }))
}

function dedupMerge(a, b) {
  const m = new Map()
  const all = [...a, ...b]
  for (const x of all) {
    if (!m.has(x.id)) m.set(x.id, x)
    else if (m.get(x.id).score < x.score) m.set(x.id, x)
  }
  return Array.from(m.values()).sort((x, y) => y.score - x.score).slice(0, 8)
}

function buildCitations(list) {
  return list.map((r, i) => `[${i + 1}]${r.meta?.source || r.meta?.title || "문서"}:${r.meta?.page || r.meta?.chunk_index || 0}`).join(" ")
}

function packContext(list, maxChars = 8000) {
  let used = 0
  const keep = []
  for (const r of list) {
    if (used + r.content.length > maxChars) break
    keep.push(r)
    used += r.content.length
  }
  return keep
}

async function classify(question) {
  const sys = `당신은 라우터다. 입력 질문을 보고 사용할 에이전트를 JSON으로만 출력한다. agents는 pdf, law, factcheck 중 선택하며 배열로 출력한다. 예: {"agents":["pdf","law","factcheck"]}`
  const r = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "system", content: sys }, { role: "user", content: question }],
    temperature: 0
  })
  try { return JSON.parse(r.choices[0].message.content) } catch { return { agents: ["pdf","law","factcheck"] } }
}

async function answerWithContexts(question, contexts) {
  const ctx = contexts.map((c, i) => `#${i + 1}\n${c.content}`).join("\n\n")
  const sys = `너는 한국 기업 HR 법무 챗봇이다. 제공된 컨텍스트만 활용해 한국어로 정확히 답한다. 컨텍스트 밖 단정은 금지한다. 답변 끝에 출처 번호를 대괄호로 표기한다.`
  const r = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: sys },
      { role: "user", content: `질문:\n${question}\n\n컨텍스트:\n${ctx}\n\n요건: 간결, 단계형, 필요한 경우 목록.` }
    ],
    temperature: 0.2
  })
  return r.choices[0].message.content
}

async function factcheck(question, draft, contexts) {
  const ctx = contexts.map((c, i) => `#${i + 1}\n${c.content}`).join("\n\n")
  const sys = `너는 사실검증기다. 드래프트 답변이 컨텍스트와 모순되는지 점검하고 필요한 경우 한 문단 이내로 수정본을 제시한다. 반드시 한국어로 출력하되, 불확실하면 불확실하다고 말한다.`
  const r = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: sys },
      { role: "user", content: `질문:\n${question}\n\n드래프트:\n${draft}\n\n컨텍스트:\n${ctx}\n\n출력: 최종 검증된 답변만.` }
    ],
    temperature: 0
  })
  return r.choices[0].message.content
}

async function searchAgent(ns, question) {
  const v = await vectorSearch(ns, question, 8)
  const k = await keywordSearch(ns, question, 8)
  const m = dedupMerge(v, k)
  return packContext(m, 8000)
}

async function getMemory(userId) {
  const { data } = await supabase.from("memories").select("*").eq("user_id", userId).order("created_at", { ascending: false }).limit(10)
  return data?.map(d => `${d.role}: ${d.content}`).join("\n") || ""
}

async function saveMemory(userId, role, content) {
  await supabase.from("memories").insert({ user_id: userId, role, content })
}

async function master(question, userId) {
  const mem = await getMemory(userId)
  const enriched = mem ? `${question}\n(이전대화)\n${mem}` : question
  const route = await classify(enriched)
  const needPdf = route.agents?.includes("pdf")
  const needLaw = route.agents?.includes("law")
  const needFC = route.agents?.includes("factcheck")
  const pdfCtx = needPdf ? await searchAgent("policy", enriched) : []
  const lawCtx = needLaw ? await searchAgent("law", enriched) : []
  const contexts = [...pdfCtx, ...lawCtx]
  const draft = await answerWithContexts(enriched, contexts)
  const finalText = needFC ? await factcheck(enriched, draft, contexts) : draft
  const citations = buildCitations([...pdfCtx, ...lawCtx])
  return { text: `${finalText}\n\n출처: ${citations}`.trim(), used: [...pdfCtx, ...lawCtx] }
}

app.get("/health", (req, res) => { res.json({ ok: true, name: process.env.SYSTEM_NAME }) })

app.post("/ingest", upload.single("file"), async (req, res) => {
  try {
    const ns = req.body.namespace
    const title = req.body.title || "문서"
    if (!ns || !req.file) return res.status(400).json({ error: "namespace and file required" })
    const data = await pdfParse(req.file.buffer)
    const pages = data.text || ""
    const ch = chunk(pages)
    await upsertChunks(ns, ch, { source: title })
    res.json({ ok: true, chunks: ch.length })
  } catch (e) {
    res.status(500).json({ error: String(e) })
  }
})

app.post("/ask", async (req, res) => {
  try {
    const q = req.body.question || ""
    const userId = req.body.userId || "anon"
    await saveMemory(userId, "user", q)
    const a = await master(q, userId)
    await saveMemory(userId, "assistant", a.text)
    res.json({ answer: a.text })
  } catch (e) {
    res.status(500).json({ error: String(e) })
  }
})

app.post("/kakao/skill", async (req, res) => {
  try {
    const q = req.body?.userRequest?.utterance || ""
    const userId = req.body?.userRequest?.user?.id || "kakao"
    await saveMemory(userId, "user", q)
    const a = await master(q, userId)
    await saveMemory(userId, "assistant", a.text)
    const payload = {
      version: "2.0",
      template: { outputs: [{ simpleText: { text: a.text.slice(0, 2999) } }] }
    }
    res.json(payload)
  } catch (e) {
    res.json({ version: "2.0", template: { outputs: [{ simpleText: { text: "오류가 발생했습니다. 잠시 후 다시 시도해 주세요." } }] } })
  }
})

const port = process.env.PORT || 3000
app.listen(port, () => {})


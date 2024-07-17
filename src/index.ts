import { Hono } from 'hono'
import { HTTPException } from 'hono/http-exception'
import ui from './ui.html'
import write from './write.html'
import { XataClient } from './xata'

type Env = {
	AI: Ai
	XATA_BRANCH: string
	XATA_API_KEY: string
	XATA_DATABASE_URL: string
}

const app = new Hono<{ Bindings: Env }>()

app.get('/', async (c) => {
	return c.html(ui)
})

// GET endpoint to query the LLM
// (v1) - query the LLM and return it (simple)
// (v2):
//  - 1. generate embeddings for the query
//  - 2. look up similar vectors to our query embedding
//  - 3. if there are similar vectors, look up the notes in d1
//  - 4. embed that note content inside of our query
//  - 5. query the LLM (with context) and return it (RAG)

app.get('/query', async (c) => {
	const xata = new XataClient({
		apiKey: c.env.XATA_API_KEY,
		branch: c.env.XATA_BRANCH,
		databaseURL: c.env.XATA_DATABASE_URL,
	})

	const question = c.req.query('text') || 'What is the square root of 9?'

	// (v2) - 1. generate embeddings for the query
	const embeddings = await c.env.AI.run('@cf/baai/bge-base-en-v1.5', { text: [question] })
	const vectors = embeddings.data[0]

	// (v2) - 2. look up similar vectors to our query embedding
	const SIMILARITY_CUTOFF = 1.5

	const results = await xata.db.Notes.vectorSearch('embedding', vectors, {
		similarityFunction: 'cosineSimilarity', // default, returns value between 0 and 2
		size: 2, // number of notes to return
	})

	results.records.forEach((r) => console.log(r.text, r.xata.score))

	// (v2) - 3. if there are similar vectors, gather the corresponding notes
	const notes = results.records.length
		? results.records.filter((record) => record.xata.score ?? 0 > SIMILARITY_CUTOFF).map((record) => record.text)
		: []

	// (v2) - 4. embed that note content inside of our query
	const contextMessage = notes.length ? `Context:\n${notes.map((note) => `- ${note}`).join('\n')}` : ''
	const systemPrompt =
		'You are a helpful assistant. When answering the question or responding, use the context provided, if it is provided and relevant.'

	// Context:
	//   - This is my first note content
	//   - This is my second note content

	// cast model to BaseAiTextGenerationModels
	const model: BaseAiTextGenerationModels = '@hf/thebloke/mistral-7b-instruct-v0.1-awq'

	// (v2) - 5. query the LLM (with context) and return it (RAG)
	const messages: RoleScopedChatInput[] = [
		...(notes.length ? [{ role: 'system' as const, content: contextMessage }] : []),
		{ role: 'system', content: systemPrompt },
		{ role: 'user', content: question },
	]

	const inputs = { messages, stream: false }

	const res = await c.env.AI.run(model, inputs)

	if (!(res instanceof ReadableStream) && typeof res === 'object') {
		// Now, we can safely access the `response` property.
		const textResponse = res.response ?? 'No response available'
		return c.text(textResponse)
	} else {
		// Handle the case where aiOutput is a ReadableStream or any other type not expected.
		return c.text('Unable to process the AI response.')
	}
})

app.get('/write', (c) => {
	return c.html(write)
})

// A POST endpoint to add notes

app.post('/notes', async (c) => {
	const xata = new XataClient({
		apiKey: c.env.XATA_API_KEY,
		branch: c.env.XATA_BRANCH,
		databaseURL: c.env.XATA_DATABASE_URL,
	})

	const { text } = await c.req.json()
	if (!text) throw new HTTPException(400, { message: 'Missing text' })

	// Insert the note into our Xata database

	// Generate an embedding based on our note
	const { data } = await c.env.AI.run('@cf/baai/bge-base-en-v1.5', { text: [text] })
	const values = data[0]

	if (!values) throw new HTTPException(500, { message: 'Failed to create vector embedding' })

	// Insert the embedding into our Xata database
	const inserted = await xata.db.Notes.create({
		text,
		embedding: values,
	})

	if (!inserted) throw new HTTPException(500, { message: 'Failed to create note' })

	return c.json({ inserted })
})

export default app

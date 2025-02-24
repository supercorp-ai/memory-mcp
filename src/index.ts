#!/usr/bin/env node

import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import express, { Request, Response } from 'express'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { z } from 'zod'
import path from 'path'
import { promises as fs } from 'fs'

//
// Logging
//
const log = (...args: any[]) => console.log('[memory-mcp]', ...args)
const logErr = (...args: any[]) => console.error('[memory-mcp]', ...args)

//
// Knowledge Graph data structures
//
interface Entity {
  name: string
  entityType: string
  observations: string[]
}

interface Relation {
  from: string
  to: string
  relationType: string
}

interface KnowledgeGraph {
  entities: Entity[]
  relations: Relation[]
}

//
// KnowledgeGraphManager for each user
//
class KnowledgeGraphManager {
  constructor(public filePath: string) {}

  async loadGraph(): Promise<KnowledgeGraph> {
    try {
      const data = await fs.readFile(this.filePath, 'utf-8')
      const lines = data.split('\n').filter(line => line.trim() !== '')
      return lines.reduce(
        (graph: KnowledgeGraph, line) => {
          const item = JSON.parse(line)
          if (item.type === 'entity') graph.entities.push(item as Entity)
          if (item.type === 'relation') graph.relations.push(item as Relation)
          return graph
        },
        { entities: [], relations: [] }
      )
    } catch (err: any) {
      if (err.code === 'ENOENT') {
        return { entities: [], relations: [] }
      }
      throw err
    }
  }

  async saveGraph(graph: KnowledgeGraph) {
    const lines = [
      ...graph.entities.map(e => JSON.stringify({ type: 'entity', ...e })),
      ...graph.relations.map(r => JSON.stringify({ type: 'relation', ...r }))
    ]
    await fs.writeFile(this.filePath, lines.join('\n'))
  }

  async createEntities(entities: Entity[]): Promise<Entity[]> {
    const graph = await this.loadGraph()
    const newEntities = entities.filter(
      e => !graph.entities.some(existing => existing.name === e.name)
    )
    graph.entities.push(...newEntities)
    await this.saveGraph(graph)
    return newEntities
  }

  async createRelations(relations: Relation[]): Promise<Relation[]> {
    const graph = await this.loadGraph()
    const newRelations = relations.filter(
      r =>
        !graph.relations.some(
          existing =>
            existing.from === r.from &&
            existing.to === r.to &&
            existing.relationType === r.relationType
        )
    )
    graph.relations.push(...newRelations)
    await this.saveGraph(graph)
    return newRelations
  }

  async addObservations(
    observations: { entityName: string; contents: string[] }[]
  ): Promise<{ entityName: string; addedObservations: string[] }[]> {
    const graph = await this.loadGraph()
    const results = observations.map(o => {
      const entity = graph.entities.find(e => e.name === o.entityName)
      if (!entity) {
        throw new Error(`Entity "${o.entityName}" not found`)
      }
      const newObs = o.contents.filter(item => !entity.observations.includes(item))
      entity.observations.push(...newObs)
      return { entityName: o.entityName, addedObservations: newObs }
    })
    await this.saveGraph(graph)
    return results
  }

  async deleteEntities(entityNames: string[]) {
    const graph = await this.loadGraph()
    graph.entities = graph.entities.filter(e => !entityNames.includes(e.name))
    graph.relations = graph.relations.filter(
      r => !entityNames.includes(r.from) && !entityNames.includes(r.to)
    )
    await this.saveGraph(graph)
  }

  async deleteObservations(
    deletions: { entityName: string; observations: string[] }[]
  ) {
    const graph = await this.loadGraph()
    deletions.forEach(d => {
      const entity = graph.entities.find(e => e.name === d.entityName)
      if (entity) {
        entity.observations = entity.observations.filter(
          obs => !d.observations.includes(obs)
        )
      }
    })
    await this.saveGraph(graph)
  }

  async deleteRelations(relations: Relation[]) {
    const graph = await this.loadGraph()
    graph.relations = graph.relations.filter(
      r =>
        !relations.some(
          x =>
            r.from === x.from &&
            r.to === x.to &&
            r.relationType === x.relationType
        )
    )
    await this.saveGraph(graph)
  }

  async readGraph(): Promise<KnowledgeGraph> {
    return this.loadGraph()
  }

  async searchNodes(query: string): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph()
    const filteredEntities = graph.entities.filter(
      e =>
        e.name.toLowerCase().includes(query.toLowerCase()) ||
        e.entityType.toLowerCase().includes(query.toLowerCase()) ||
        e.observations.some(o => o.toLowerCase().includes(query.toLowerCase()))
    )
    const filteredNames = new Set(filteredEntities.map(e => e.name))
    const filteredRelations = graph.relations.filter(
      r => filteredNames.has(r.from) && filteredNames.has(r.to)
    )
    return { entities: filteredEntities, relations: filteredRelations }
  }

  async openNodes(names: string[]): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph()
    const filteredEntities = graph.entities.filter(e => names.includes(e.name))
    const nameSet = new Set(filteredEntities.map(e => e.name))
    const filteredRelations = graph.relations.filter(
      r => nameSet.has(r.from) && nameSet.has(r.to)
    )
    return { entities: filteredEntities, relations: filteredRelations }
  }
}

//
// Provide a standard JSON result for Tools
//
const toTextJson = (data: unknown) => ({
  content: [
    {
      type: 'text' as const,
      text: JSON.stringify(data, null, 2)
    }
  ]
})

//
// Create a user-specific McpServer
//
function createMemoryServerForUser(baseDir: string, userId: string): McpServer {
  // Build or retrieve a manager for userId
  const filePath = path.join(baseDir, `${userId}.json`)
  const manager = new KnowledgeGraphManager(filePath)

  const server = new McpServer({
    name: `Memory MCP Server (User: ${userId})`,
    version: '1.0.0'
  })

  server.tool(
    'create_entities',
    'Create new entities',
    {
      entities: z.array(z.object({
        name: z.string(),
        entityType: z.string(),
        observations: z.array(z.string())
      }))
    },
    async ({ entities }) => {
      try {
        const result = await manager.createEntities(entities)
        return toTextJson(result)
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'create_relations',
    'Create new relations',
    {
      relations: z.array(z.object({
        from: z.string(),
        to: z.string(),
        relationType: z.string()
      }))
    },
    async ({ relations }) => {
      try {
        const created = await manager.createRelations(relations)
        return toTextJson(created)
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'add_observations',
    'Add observations to existing entities',
    {
      observations: z.array(z.object({
        entityName: z.string(),
        contents: z.array(z.string())
      }))
    },
    async ({ observations }) => {
      try {
        const result = await manager.addObservations(observations)
        return toTextJson(result)
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'delete_entities',
    'Delete entities (and their relations)',
    {
      entityNames: z.array(z.string())
    },
    async ({ entityNames }) => {
      try {
        await manager.deleteEntities(entityNames)
        return toTextJson({ success: true })
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'delete_observations',
    'Delete observations from entities',
    {
      deletions: z.array(z.object({
        entityName: z.string(),
        observations: z.array(z.string())
      }))
    },
    async ({ deletions }) => {
      try {
        await manager.deleteObservations(deletions)
        return toTextJson({ success: true })
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'delete_relations',
    'Delete relations',
    {
      relations: z.array(z.object({
        from: z.string(),
        to: z.string(),
        relationType: z.string()
      }))
    },
    async ({ relations }) => {
      try {
        await manager.deleteRelations(relations)
        return toTextJson({ success: true })
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'read_graph',
    'Read entire knowledge graph',
    {},
    async () => {
      try {
        const data = await manager.readGraph()
        return toTextJson(data)
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'search_nodes',
    'Search nodes by query',
    {
      query: z.string()
    },
    async ({ query }) => {
      try {
        const data = await manager.searchNodes(query)
        return toTextJson(data)
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  server.tool(
    'open_nodes',
    'Open specific nodes by name',
    {
      names: z.array(z.string())
    },
    async ({ names }) => {
      try {
        const data = await manager.openNodes(names)
        return toTextJson(data)
      } catch (err: any) {
        return toTextJson({ error: String(err.message) })
      }
    }
  )

  return server
}

//
// We'll track SSE sessions. Each session is tied to exactly one userId.
// No "context" passing is needed for each tool call.
//
interface ServerSession {
  userId: string
  server: McpServer
  transport: SSEServerTransport
  sessionId: string
}

//
// Main function: parse CLI, run SSE or stdio
//
async function main() {
  const argv = yargs(hideBin(process.argv))
    .option('port', { type: 'number', default: 8000 })
    .option('transport', { type: 'string', choices: ['sse', 'stdio'], default: 'sse' })
    .option('memoryBase', {
      type: 'string',
      describe: 'Base directory for storing user memory JSON files',
      demandOption: true
    })
    .help()
    .parseSync()

  const baseDir = path.resolve(argv.memoryBase)
  await fs.mkdir(baseDir, { recursive: true }).catch(() => {})

  if (argv.transport === 'stdio') {
    // In stdio mode, you might read a userId from an environment var or a prompt.
    // For demo, we'll say "single user" or require a userId from an env var.
    const userId = process.env.USER_ID || 'stdio-user'
    const server = createMemoryServerForUser(baseDir, userId)
    const transport = new StdioServerTransport()
    await server.connect(transport)
    log('Listening on stdio')
    return
  }

  // SSE mode
  const port = argv.port
  const app = express()
  let sessions: ServerSession[] = []

  // parse JSON except for /message
  app.use((req, res, next) => {
    if (req.path === '/message') return next()
    express.json()(req, res, next)
  })

  // GET / => Start SSE session
  app.get('/', async (req: Request, res: Response) => {
    // Grab user-id from headers
    const userId = req.headers['user-id']
    if (typeof userId !== 'string' || !userId.trim()) {
      res.status(400).json({ error: 'Missing or invalid "user-id" header' })
      return
    }

    // Create an MCP server specifically for this user
    const server = createMemoryServerForUser(baseDir, userId.trim())

    // Start SSE
    const transport = new SSEServerTransport('/message', res)
    await server.connect(transport)  // no context param needed

    // Track session
    const sessionId = transport.sessionId
    sessions.push({ userId, server, transport, sessionId })

    log(`[${sessionId}] SSE connection established for user: "${userId}"`)

    transport.onclose = () => {
      log(`[${sessionId}] SSE connection closed`)
      sessions = sessions.filter(s => s.transport !== transport)
    }
    transport.onerror = (err: Error) => {
      logErr(`[${sessionId}] SSE error:`, err)
      sessions = sessions.filter(s => s.transport !== transport)
    }
    req.on('close', () => {
      log(`[${sessionId}] Client disconnected`)
      sessions = sessions.filter(s => s.transport !== transport)
    })
  })

  // POST /message => SSE session updates
  app.post('/message', async (req: Request, res: Response) => {
    const sessionId = req.query.sessionId as string
    if (!sessionId) {
      res.status(400).send({ error: 'Missing sessionId' })
      return
    }
    const target = sessions.find(s => s.sessionId === sessionId)
    if (!target) {
      res.status(404).send({ error: 'No active session' })
      return
    }
    try {
      // Just handle the message. We already know which user this session is for.
      await target.transport.handlePostMessage(req, res)
    } catch (err) {
      logErr(`[${sessionId}] Error handling /message:`, err)
      res.status(500).send({ error: 'Internal error' })
    }
  })

  app.listen(port, () => {
    log(`Listening on port ${port} (SSE)`)
  })
}

main().catch(err => {
  logErr('Fatal error:', err)
  process.exit(1)
})

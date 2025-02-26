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
import { Redis } from '@upstash/redis'

// -------------------------------------------------------------------
// Logging
// -------------------------------------------------------------------
const log = (...args: any[]) => console.log('[memory-mcp]', ...args)
const logErr = (...args: any[]) => console.error('[memory-mcp]', ...args)

// -------------------------------------------------------------------
// Data Structures
// -------------------------------------------------------------------
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

// -------------------------------------------------------------------
// Shared Manager Interface
// -------------------------------------------------------------------
interface IKnowledgeGraphManager {
  createEntities(entities: Entity[]): Promise<Entity[]>
  createRelations(relations: Relation[]): Promise<Relation[]>
  addObservations(
    observations: { entityName: string; contents: string[] }[]
  ): Promise<{ entityName: string; addedObservations: string[] }[]>
  deleteEntities(entityNames: string[]): Promise<void>
  deleteObservations(
    deletions: { entityName: string; observations: string[] }[]
  ): Promise<void>
  deleteRelations(relations: Relation[]): Promise<void>
  readGraph(): Promise<KnowledgeGraph>
  searchNodes(query: string): Promise<KnowledgeGraph>
  openNodes(names: string[]): Promise<KnowledgeGraph>
}

// -------------------------------------------------------------------
// Base Manager: DRY CRUD logic in one place
// -------------------------------------------------------------------
abstract class BaseKnowledgeGraphManager implements IKnowledgeGraphManager {
  /** Must load entire graph from storage. */
  protected abstract loadGraph(): Promise<KnowledgeGraph>

  /** Must save entire graph to storage. */
  protected abstract saveGraph(graph: KnowledgeGraph): Promise<void>

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

  async deleteEntities(entityNames: string[]): Promise<void> {
    const graph = await this.loadGraph()
    graph.entities = graph.entities.filter(e => !entityNames.includes(e.name))
    graph.relations = graph.relations.filter(
      r => !entityNames.includes(r.from) && !entityNames.includes(r.to)
    )
    await this.saveGraph(graph)
  }

  async deleteObservations(
    deletions: { entityName: string; observations: string[] }[]
  ): Promise<void> {
    const graph = await this.loadGraph()
    for (const d of deletions) {
      const entity = graph.entities.find(e => e.name === d.entityName)
      if (entity) {
        entity.observations = entity.observations.filter(
          obs => !d.observations.includes(obs)
        )
      }
    }
    await this.saveGraph(graph)
  }

  async deleteRelations(relations: Relation[]): Promise<void> {
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
    const lower = query.toLowerCase()
    const filteredEntities = graph.entities.filter(
      e =>
        e.name.toLowerCase().includes(lower) ||
        e.entityType.toLowerCase().includes(lower) ||
        e.observations.some(o => o.toLowerCase().includes(lower))
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

// -------------------------------------------------------------------
// Filesystem Manager
// -------------------------------------------------------------------
class KnowledgeGraphFsManager extends BaseKnowledgeGraphManager {
  constructor(private filePath: string) {
    super()
  }

  protected async loadGraph(): Promise<KnowledgeGraph> {
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

  protected async saveGraph(graph: KnowledgeGraph) {
    const lines = [
      ...graph.entities.map(e => JSON.stringify({ type: 'entity', ...e })),
      ...graph.relations.map(r => JSON.stringify({ type: 'relation', ...r }))
    ]
    await fs.writeFile(this.filePath, lines.join('\n'))
  }
}

// -------------------------------------------------------------------
// Upstash Redis (REST) Manager
//  - We get a URL *and* a token from CLI
//  - We'll store data in a single JSON key
// -------------------------------------------------------------------
class KnowledgeGraphUpstashRedisManager extends BaseKnowledgeGraphManager {
  private redis: Redis
  private key: string

  constructor(url: string, token: string, userId: string) {
    super()
    this.redis = new Redis({ url, token })
    this.key = `graph:${userId}`
  }

  protected async loadGraph(): Promise<KnowledgeGraph> {
    const data = await this.redis.get<string>(this.key)
    if (!data) {
      return { entities: [], relations: [] }
    }
    return JSON.parse(data) as KnowledgeGraph
  }

  protected async saveGraph(graph: KnowledgeGraph): Promise<void> {
    await this.redis.set(this.key, JSON.stringify(graph))
  }
}

// -------------------------------------------------------------------
// Minimal JSON result helper
// -------------------------------------------------------------------
function toTextJson(data: unknown) {
  return {
    content: [
      {
        type: 'text' as const,
        text: JSON.stringify(data, null, 2)
      }
    ]
  }
}

// -------------------------------------------------------------------
// Factory: Create the appropriate manager
// -------------------------------------------------------------------
function createManager(
  storage: 'fs' | 'upstash-redis-rest',
  baseDir: string,
  userId: string,
  upstashRedisRestUrl?: string,
  upstashRedisRestToken?: string
): IKnowledgeGraphManager {
  if (storage === 'upstash-redis-rest') {
    if (!upstashRedisRestUrl || !upstashRedisRestToken) {
      throw new Error('--upstash-redis-rest-url and --upstash-redis-rest-token are required')
    }
    return new KnowledgeGraphUpstashRedisManager(
      upstashRedisRestUrl,
      upstashRedisRestToken,
      userId
    )
  }
  // Default: filesystem
  return new KnowledgeGraphFsManager(path.join(baseDir, `${userId}.json`))
}

// -------------------------------------------------------------------
// Build an MCP server for a single user + chosen manager
// -------------------------------------------------------------------
function createMemoryServerForUser(
  storage: 'fs' | 'upstash-redis-rest',
  baseDir: string,
  userId: string,
  upstashRedisRestUrl?: string,
  upstashRedisRestToken?: string
): McpServer {
  const manager = createManager(storage, baseDir, userId, upstashRedisRestUrl, upstashRedisRestToken)

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

// -------------------------------------------------------------------
// Main CLI Entry
// -------------------------------------------------------------------
async function main() {
  const argv = yargs(hideBin(process.argv))
    .option('port', { type: 'number', default: 8000 })
    .option('transport', {
      type: 'string',
      choices: ['sse', 'stdio'],
      default: 'sse'
    })
    .option('storage', {
      type: 'string',
      choices: ['fs', 'upstash-redis-rest'],
      default: 'fs',
      describe: 'Choose storage backend'
    })
    .option('memoryBase', {
      type: 'string',
      default: './data',
      describe: 'Local filesystem dir (only if --storage=fs)'
    })
    .option('upstashRedisRestUrl', {
      type: 'string',
      describe: 'Upstash Redis REST URL (if --storage=upstash-redis-rest)'
    })
    .option('upstashRedisRestToken', {
      type: 'string',
      describe: 'Upstash Redis REST token (if --storage=upstash-redis-rest)'
    })
    .help()
    .parseSync()

  // If file-based, ensure the base dir
  if (argv.storage === 'fs') {
    const baseDir = path.resolve(argv.memoryBase)
    await fs.mkdir(baseDir, { recursive: true }).catch(() => {})
  }

  // If user picks stdio transport => single user
  if (argv.transport === 'stdio') {
    const userId = process.env.USER_ID || 'stdio-user'
    const server = createMemoryServerForUser(
      argv.storage as 'fs' | 'upstash-redis-rest',
      path.resolve(argv.memoryBase),
      userId,
      argv.upstashRedisRestUrl,
      argv.upstashRedisRestToken
    )
    const transport = new StdioServerTransport()
    await server.connect(transport)
    log('Listening on stdio')
    return
  }

  // Otherwise SSE
  const port = argv.port
  const app = express()

  interface ServerSession {
    userId: string
    server: McpServer
    transport: SSEServerTransport
    sessionId: string
  }
  let sessions: ServerSession[] = []

  // parse JSON except /message
  app.use((req, res, next) => {
    if (req.path === '/message') return next()
    express.json()(req, res, next)
  })

  // GET / => Start SSE
  app.get('/', async (req: Request, res: Response) => {
    const userId = req.headers['user-id']
    if (typeof userId !== 'string' || !userId.trim()) {
      res.status(400).json({ error: 'Missing or invalid "user-id" header' })
      return
    }

    const server = createMemoryServerForUser(
      argv.storage as 'fs' | 'upstash-redis-rest',
      path.resolve(argv.memoryBase),
      userId.trim(),
      argv.upstashRedisRestUrl,
      argv.upstashRedisRestToken
    )
    const transport = new SSEServerTransport('/message', res)
    await server.connect(transport)

    const sessionId = transport.sessionId
    sessions.push({ userId, server, transport, sessionId })

    log(`[${sessionId}] SSE connected for user: "${userId}"`)

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

  // POST /message => SSE updates
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
      await target.transport.handlePostMessage(req, res)
    } catch (err) {
      logErr(`[${sessionId}] /message error:`, err)
      res.status(500).send({ error: 'Internal error' })
    }
  })

  app.listen(port, () => {
    log(`Listening on port ${port} [storage=${argv.storage}]`)
  })
}

// -------------------------------------------------------------------
// Start
// -------------------------------------------------------------------
main().catch(err => {
  logErr('Fatal error:', err)
  process.exit(1)
})

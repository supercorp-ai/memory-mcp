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
// Logging (same style as the new example)
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
        // file not found, return empty
        return { entities: [], relations: [] }
      }
      throw err
    }
  }

  async saveGraph(graph: KnowledgeGraph): Promise<void> {
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
// A cache for KnowledgeGraphManager, keyed by userId
//
const managerCache = new Map<string, KnowledgeGraphManager>()

const getManagerForUser = (base: string, userId: string): KnowledgeGraphManager => {
  if (!userId) {
    throw new Error('No user ID provided')
  }
  if (!managerCache.has(userId)) {
    const filePath = path.join(base, `${userId}.json`)
    managerCache.set(userId, new KnowledgeGraphManager(filePath))
  }
  return managerCache.get(userId)!
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
// Build an MCP server with memory tools
//
const createMemoryServer = (base: string): McpServer => {
  const server = new McpServer({
    name: 'Memory MCP Server',
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
    async ({ entities }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
        const created = await manager.createEntities(entities)
        return toTextJson(created)
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
    async ({ relations }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async ({ observations }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async ({ entityNames }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async ({ deletions }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async ({ relations }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async (_, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async ({ query }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
    async ({ names }, context) => {
      const userId = getUserIdFromHeaders(context)
      try {
        const manager = getManagerForUser(base, userId)
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
// Helper to read user ID from context.*Request.headers
//
const getUserIdFromHeaders = (context: any): string => {
  // For SSE transport, it's context.sseRequest
  // For HTTP transport, it's context.httpRequest
  // For stdio, there's no request object, so we fail if we can't see it
  const userId =
    context?.sseRequest?.headers?.['x-user-id'] ||
    context?.httpRequest?.headers?.['x-user-id']

  if (typeof userId !== 'string' || !userId.trim()) {
    throw new Error('Missing or invalid x-user-id header')
  }
  return userId.trim()
}

interface ServerSession {
  server: McpServer
  transport: SSEServerTransport
}

//
// Main function: parse CLI, run SSE or stdio
//
const main = async () => {
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
    const server = createMemoryServer(baseDir)
    const transport = new StdioServerTransport()
    await server.connect(transport)
    log('Listening on stdio')
    return
  }

  const port = argv.port
  const app = express()
  let sessions: ServerSession[] = []

  app.use((req, res, next) => {
    if (req.path === '/message') return next()
    express.json()(req, res, next)
  })

  app.get('/', async (req: Request, res: Response) => {
    // We only note that userId is read once a tool call is made.
    // The SSE handshake itself doesn't do memory ops.
    const transport = new SSEServerTransport('/message', res)
    const server = createMemoryServer(baseDir)
    await server.connect(transport)
    sessions.push({ server, transport })
    const sessionId = transport.sessionId
    log(`[${sessionId}] New SSE connection established`)

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

  app.post('/message', async (req: Request, res: Response) => {
    const sessionId = req.query.sessionId as string
    if (!sessionId) {
      res.status(400).send({ error: 'Missing sessionId' })
      return
    }
    const target = sessions.find(s => s.transport.sessionId === sessionId)
    if (!target) {
      res.status(404).send({ error: 'No active session' })
      return
    }
    try {
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

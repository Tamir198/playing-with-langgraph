import { StateGraph, START } from '@langchain/langgraph';
import { SqliteSaver } from '@langchain/langgraph-checkpoint-sqlite';
import { State } from './state.js';
import { toolNode } from './tools.js';
import { callModel, shouldContinue, humanReview } from './nodes.js';

const workflow = new StateGraph(State)
  .addNode('agent', callModel)
  .addNode('review', humanReview, { ends: ['tools', 'agent'] })
  .addNode('tools', toolNode)
  .addEdge(START, 'agent')
  .addConditionalEdges('agent', shouldContinue)
  .addEdge('tools', 'agent');

/**
 * Local-file checkpointer. The whole graph state (channels + pending
 * interrupts + checkpoint metadata) is persisted to this SQLite file after
 * every superstep, so conversations survive process restarts and you can
 * time-travel through history.
 *
 * Swap this single line to change durability:
 *   - `new MemorySaver()`                                 → RAM only
 *   - `SqliteSaver.fromConnString('./calculator.sqlite')` → local file
 *   - `PostgresSaver.fromConnString(process.env.PG_URL!)` → remote DB
 * The graph itself is checkpointer-agnostic.
 */
const checkpointer = SqliteSaver.fromConnString('./calculator.sqlite');

export const app = workflow.compile({ checkpointer });

import { StateGraph, START } from '@langchain/langgraph';
import { MemorySaver } from '@langchain/langgraph-checkpoint';
import { State } from './state.js';
import { toolNode } from './tools.js';
import { callModel, shouldContinue } from './nodes.js';

const workflow = new StateGraph(State)
  .addNode('agent', callModel)
  .addNode('tools', toolNode)
  .addEdge(START, 'agent')
  .addConditionalEdges('agent', shouldContinue)
  .addEdge('tools', 'agent');

const checkpointer = new MemorySaver();

export const app = workflow.compile({ checkpointer });

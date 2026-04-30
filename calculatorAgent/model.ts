import { ChatOllama } from '@langchain/ollama';
import { tools } from './tools.js';

export const model = new ChatOllama({
  model: 'llama3.2',
  temperature: 0,
}).bindTools(tools);

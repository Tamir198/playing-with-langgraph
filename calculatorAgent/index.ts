import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatOllama } from '@langchain/ollama';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';

// 🛠️ TOOL DEFINITION
// An agent's power comes from its tools. This tool lets the LLM perform math.
const calculator = tool(
  async ({ query }) => {
    try {
      // ⚠️ Use a safer evaluation method in production!
      return eval(query).toString();
    } catch {
      return 'Error: Invalid mathematical expression.';
    }
  },
  {
    name: 'calculator',
    description: 'Evaluates a mathematical expression and returns the result.',
    schema: z.object({
      query: z
        .string()
        .describe("The math expression to evaluate (e.g. '2 + 2')"),
    }),
  },
);

const tools = [calculator];
const toolNode = new ToolNode(tools);

// 🧠 STATE DEFINITION
// Agents work with 'Messages'. This allows for memory and tool-call tracking.
const State = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

// 🤖 MODEL DEFINITION
const model = new ChatOllama({
  model: 'llama3.2', // Smaller model (2GB) that supports tool-calling correctly
  temperature: 0,
}).bindTools(tools);

// ⚙️ NODE FUNCTIONS
const callModel = async (state: typeof State.State) => {
  const { messages } = state;
  const response = await model.invoke(messages);
  // We return a list, because this will be passed to the reducer (concat)
  return { messages: [response] };
};

// 🛤️ ROUTING LOGIC
const shouldContinue = (state: typeof State.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  const toolCalls = lastMessage.additional_kwargs.tool_calls || (lastMessage as any).tool_calls;
  if (toolCalls?.length > 0) {
    return 'tools';
  }
  // Otherwise, we stop
  return END;
};

// 🏗️ GRAPH CONSTRUCTION
const graph = new StateGraph(State)
  .addNode('agent', callModel)
  .addNode('tools', toolNode)
  .addEdge(START, 'agent')
  .addConditionalEdges('agent', shouldContinue)
  .addEdge('tools', 'agent'); // Tools always go back to agent for summary

const app = graph.compile();

// 🚀 EXECUTION
const finalState = await app.invoke({
  messages: [new HumanMessage('What is 2 + 2 * (10 / 2)?')],
});

console.log('--- FINAL RESPONSE ---');
console.log(finalState.messages[finalState.messages.length - 1].content);

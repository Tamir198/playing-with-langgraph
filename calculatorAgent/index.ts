import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { MemorySaver } from '@langchain/langgraph-checkpoint';
import { ChatOllama } from '@langchain/ollama';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';
import { createInterface } from 'node:readline/promises';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// 🛠️ TOOL DEFINITION
// An agent's power comes from its tools. This tool lets the LLM perform math.
const calculator = tool(
  async ({ query }) => {
    try {
      let expression = query;
      // If the model weirdly sent a JSON string, let's extract the math!
      if (typeof query === 'string' && query.startsWith('{')) {
        try {
          const parsed = JSON.parse(query);
          expression = parsed.description || parsed.query || query;
        } catch {}
      }

      console.log(`--- Evaluating: "${expression}" ---`);
      return eval(expression).toString();
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

  const toolCalls =
    lastMessage.additional_kwargs.tool_calls || (lastMessage as any).tool_calls;
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

// 🧠 CHECKPOINTER — saves the graph state after every node so the agent
// can "remember" previous turns within the same thread_id session.
const checkpointer = new MemorySaver();
const app = graph.compile({ checkpointer });

// 🚀 INTERACTIVE CONVERSATION LOOP
// Each invoke() with the same thread_id reloads the full message history from
// the checkpointer, so the agent sees every previous turn automatically.
const config = { configurable: { thread_id: 'session-1' } };

const rl = createInterface({ input: process.stdin, output: process.stdout });

console.log('🧮 Calculator Agent (type "exit" to quit)\n');

const conversationLog: string[] = [];

while (true) {
  const userInput = await rl.question('You: ');
  if (userInput.trim().toLowerCase() === 'exit') break;

  const result = await app.invoke(
    { messages: [new HumanMessage(userInput)] },
    config,
  );

  const lastMessage = result.messages[result.messages.length - 1];
  const response = lastMessage.content.toString();

  console.log(`Agent: ${response}\n`);
  conversationLog.push(`You: ${userInput}`, `Agent: ${response}`);
}

rl.close();

const outputPath = join('calculatorAgent', 'output.txt');
await writeFile(outputPath, conversationLog.join('\n'), 'utf-8');
console.log(`💾 Conversation saved to: ${outputPath}`);
console.log('👋 Goodbye!');

import { ToolNode } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { z } from 'zod';
import { State } from './state.js';

const calculator = tool(
  async ({ query }) => {
    try {
      let expression = query;
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

export const tools = [calculator];

const baseToolNode = new ToolNode(tools);

/**
 * Custom tools node. A node is just an `(state) => partial state` function,
 * so we can compose: delegate message-handling to the prebuilt `ToolNode`,
 * then add our own write to the `toolNamesUsed` channel by reading the
 * previous AI message's tool_calls.
 *
 * The result is a partial state containing two channel updates. LangGraph
 * sends each one through its respective reducer (append for messages,
 * append for toolNamesUsed) and stores the merged state at the next
 * checkpoint.
 */
export const toolNode = async (
  state: typeof State.State,
): Promise<{
  messages: typeof State.State.messages;
  toolNamesUsed: string[];
}> => {
  const result = (await baseToolNode.invoke(state)) as {
    messages: typeof State.State.messages;
  };

  const lastAi = state.messages[state.messages.length - 1];
  const calls =
    lastAi instanceof AIMessage || lastAi instanceof AIMessageChunk
      ? lastAi.tool_calls ?? []
      : [];

  return {
    messages: result.messages,
    toolNamesUsed: calls.map((c) => c.name),
  };
};

import { ToolNode } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

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
export const toolNode = new ToolNode(tools);

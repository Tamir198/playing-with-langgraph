import { END, interrupt, Command } from '@langchain/langgraph';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { State } from './state.js';
import { model } from './model.js';

/**
 * The payload we pass through `interrupt(...)`. Anything that flows through
 * `interrupt` is part of your graph's *external* contract (the CLI/UI on the
 * other side has to know how to render and answer it), so it deserves a name.
 */
export interface ToolApprovalInterrupt {
  question: string;
  tool_calls: NonNullable<AIMessage['tool_calls']>;
}

/** What `interrupt()` resumes with — the value the CLI passes to `Command({ resume })`. */
export type ToolApprovalDecision = 'approve' | 'reject';

/**
 * Pull tool calls off the latest message, handling both shapes that may
 * appear in state:
 *   - `AIMessage`      — what `model.invoke()` produces
 *   - `AIMessageChunk` — what `model.stream()` produces (after our reduce)
 *
 * IMPORTANT: `AIMessageChunk` is NOT a subclass of `AIMessage` — they are
 * *siblings*, both extending `BaseMessage`. So `msg instanceof AIMessage`
 * alone returns false for a chunk and you'd silently drop the tool call.
 * Both classes expose the same typed `.tool_calls` field, so check both.
 */
const getToolCalls = (
  msg: unknown,
): NonNullable<AIMessage['tool_calls']> => {
  if (msg instanceof AIMessage || msg instanceof AIMessageChunk) {
    return msg.tool_calls ?? [];
  }
  return [];
};

/**
 * Call the LLM via `model.stream()` and reduce chunks into one accumulated
 * AIMessageChunk. We use `.stream()` (not `.invoke()`) so that LangGraph's
 * `'messages'` stream mode actually receives token-by-token events — for
 * many providers (notably ChatOllama), `.invoke()` makes a non-streaming
 * request and emits zero `on_chat_model_stream` callbacks, which would make
 * the typewriter UI silent.
 *
 * `AIMessageChunk.concat` merges content AND tool_call_chunks correctly, so
 * the final accumulated message exposes a normal `.tool_calls` array — the
 * rest of the graph keeps working unchanged.
 */
export const callModel = async (state: typeof State.State) => {
  const stream = await model.stream(state.messages);

  let accumulated: AIMessageChunk | undefined;
  for await (const chunk of stream) {
    accumulated = accumulated ? accumulated.concat(chunk) : chunk;
  }

  return { messages: accumulated ? [accumulated] : [] };
};

export const shouldContinue = (state: typeof State.State) => {
  const lastMessage = state.messages[state.messages.length - 1];
  return getToolCalls(lastMessage).length > 0 ? 'review' : END;
};

export const humanReview = (state: typeof State.State): Command => {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = getToolCalls(lastMessage);

  const payload: ToolApprovalInterrupt = {
    question: 'Do you approve this tool call?',
    tool_calls: toolCalls,
  };

  const decision = interrupt(payload) as ToolApprovalDecision;

  return decision === 'approve'
    ? new Command({ goto: 'tools' })
    : new Command({ goto: 'agent' });
};

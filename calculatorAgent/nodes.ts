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

/**
 * What the CLI passes back via `Command({ resume })`:
 *   - 'approve' → run the tool once
 *   - 'reject'  → don't run; route back to the agent so it can try again
 *   - 'always'  → run the tool AND set state.autoApprove = true so the
 *                 graph stops asking on future tool calls (persisted)
 */
export type ToolApprovalDecision = 'approve' | 'reject' | 'always';

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
 * This node also writes to the `turnCount` channel: returning `1` means
 * "add one to the running total" because the channel's reducer is `a + b`.
 * Note we only return `turnCount: 1` once per call regardless of how many
 * chunks streamed — the reducer cares about node returns, not LLM chunks.
 */
export const callModel = async (state: typeof State.State) => {
  const stream = await model.stream(state.messages);

  let accumulated: AIMessageChunk | undefined;
  for await (const chunk of stream) {
    accumulated = accumulated ? accumulated.concat(chunk) : chunk;
  }

  return {
    messages: accumulated ? [accumulated] : [],
    turnCount: 1,
  };
};

export const shouldContinue = (state: typeof State.State) => {
  const lastMessage = state.messages[state.messages.length - 1];
  return getToolCalls(lastMessage).length > 0 ? 'review' : END;
};

/**
 * Human-in-the-loop tool approval node. Demonstrates two important
 * patterns:
 *
 *   1. READING state to short-circuit: if `state.autoApprove` is true, we
 *      skip the interrupt entirely and route straight to the tools node.
 *      The user's earlier "always" decision survives process restart
 *      because it's persisted via the checkpointer.
 *
 *   2. WRITING state in transition via `Command.update`: when the user
 *      replies 'always', we both route to 'tools' AND write
 *      `autoApprove: true`. Because the channel's reducer is overwrite,
 *      this becomes the new value. The next interrupt() check above will
 *      see it and skip asking.
 */
export const humanReview = (state: typeof State.State): Command => {
  if (state.autoApprove) {
    return new Command({ goto: 'tools' });
  }

  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = getToolCalls(lastMessage);

  const payload: ToolApprovalInterrupt = {
    question: 'Do you approve this tool call?',
    tool_calls: toolCalls,
  };

  const decision = interrupt(payload) as ToolApprovalDecision;

  switch (decision) {
    case 'approve':
      return new Command({ goto: 'tools' });
    case 'always':
      return new Command({
        goto: 'tools',
        update: { autoApprove: true },
      });
    case 'reject':
    default:
      return new Command({ goto: 'agent' });
  }
};

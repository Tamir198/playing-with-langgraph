import { Annotation } from '@langchain/langgraph';
import { BaseMessage } from '@langchain/core/messages';

/**
 * Graph state. Each field is a *channel* — a typed slot with its own
 * reducer that says how multiple writes within one superstep get merged.
 *
 * Every node returns a partial state update; LangGraph routes each key
 * through the matching channel's reducer to produce the next state.
 *
 * The four channels here intentionally use four different reducer
 * patterns so you can see when to reach for which:
 *
 *   - messages       — append (concat). Conversation history.
 *   - turnCount      — sum. A simple accumulator.
 *   - autoApprove    — overwrite (last-write-wins). A single setting.
 *   - toolNamesUsed  — append (concat) of strings. A flat audit log.
 *
 * The reducer's signature is `(current, update) => next`. The `default`
 * function provides the initial value before any node has written.
 */
export const State = Annotation.Root({
  /** Append: every node that produces messages adds to the running list. */
  messages: Annotation<BaseMessage[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),

  /**
   * Sum: nodes write the *delta* (e.g. `1`), the reducer adds it to the
   * running total. Classic accumulator pattern. Good for token counts,
   * step counts, retry counts, anything cumulative.
   */
  turnCount: Annotation<number>({
    reducer: (current, update) => current + update,
    default: () => 0,
  }),

  /**
   * Overwrite: only the latest write wins. Use this when the value is a
   * single piece of state (a setting, a flag, the current step name).
   * The first arg is ignored on purpose — that's what "last-write-wins"
   * means at the channel level.
   */
  autoApprove: Annotation<boolean>({
    reducer: (_current, update) => update,
    default: () => false,
  }),

  /**
   * Append (strings). Same shape as `messages` but a different domain:
   * a flat audit trail of tool names that were executed. Demonstrates
   * that the reducer pattern, not the field name, is what defines the
   * channel's behavior.
   */
  toolNamesUsed: Annotation<string[]>({
    reducer: (current, update) => [...current, ...update],
    default: () => [],
  }),
});

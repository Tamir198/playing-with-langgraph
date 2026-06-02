import { HumanMessage, type BaseMessage } from '@langchain/core/messages';
import { Command } from '@langchain/langgraph';
import { createInterface } from 'node:readline/promises';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { app } from './graph.js';
import type {
  ToolApprovalInterrupt,
  ToolApprovalDecision,
} from './nodes.js';

/**
 * Type the input to `app.stream(...)` by *deriving* it from the compiled app
 * itself. `Parameters<typeof app.stream>[0]` resolves to the union the SDK
 * declares: a state update for this graph, a Command (for resume), or null
 * (for replaying from a checkpoint). No `as any` needed.
 */
type GraphInput = Parameters<typeof app.stream>[0];

/**
 * LangGraph stream modes — try changing the `streamMode` below to learn the
 * differences. The graph stays the same; only the *view* into its execution
 * changes.
 *
 *   'values'   → after each step, yields the FULL current state.
 *                Easy to inspect, but verbose. Good for debugging state shape.
 *
 *   'updates'  → after each step, yields only the DIFF: { [nodeName]: update }.
 *                Best for "which node just ran and what did it write?".
 *                Interrupts arrive as { __interrupt__: [...] }.
 *
 *   'messages' → yields LLM output token-by-token: [AIMessageChunk, metadata].
 *                Use this for typewriter UIs. `metadata.langgraph_node` tells
 *                you which node produced the chunk.
 *
 *   'debug'    → verbose lifecycle events (task start/end, checkpoints).
 *
 *   ['updates', 'messages'] → mixed mode. Each yield is a [mode, chunk] tuple
 *                so you can react to structural events AND tokens in one loop.
 *
 * We use the mixed mode so you can SEE node transitions interleaved with the
 * model's tokens as they stream in.
 */

const config = { configurable: { thread_id: 'session-1' } };

const rl = createInterface({ input: process.stdin, output: process.stdout });

console.log('🧮 Calculator Agent — streaming edition');
console.log('   Tool approval options: yes / no / always\n');

let autoApprove = false;
const conversationLog: string[] = [];

const dim = (s: string) => `\x1b[90m${s}\x1b[0m`;

/**
 * Run the graph as a stream and pretty-print:
 *   - dim "· node X finished" lines from the 'updates' channel
 *   - the agent's reply token-by-token from the 'messages' channel
 *
 * The function returns when the stream ends, which happens either because the
 * graph reached END or because it hit an interrupt() and is now paused.
 * The caller checks `app.getState(config)` to tell the difference.
 */
/**
 * Returns `true` if any agent text was actually printed during the run.
 * The CLI uses this to decide whether it needs a fallback "print the final
 * assistant message" step (e.g. for models/providers whose `.stream()` only
 * emits tool-call chunks and no text, or in test scenarios).
 */
async function streamRun(input: GraphInput): Promise<boolean> {
  let typing = false;
  let printedAnyText = false;

  /**
   * When `streamMode` is an array, the SDK returns a **discriminated union**:
   *   ['updates', updatesChunk] | ['messages', [BaseMessage, metadata]]
   * Narrowing on `event[0]` lets TS infer the right shape for `event[1]` —
   * no manual cast required.
   */
  const stream = await app.stream(input, {
    ...config,
    streamMode: ['updates', 'messages'],
  });

  for await (const event of stream) {
    if (event[0] === 'updates') {
      // event[1] is { [nodeName]: nodeUpdate } | { __interrupt__: Interrupt[] }
      for (const nodeName of Object.keys(event[1])) {
        if (nodeName === '__interrupt__') continue;
        if (typing) {
          process.stdout.write('\n');
          typing = false;
        }
        console.log(dim(`  · node "${nodeName}" finished`));
      }
    } else if (event[0] === 'messages') {
      // event[1] is [BaseMessage, Record<string, any>] (the SDK's metadata bag).
      const [msgChunk, meta] = event[1];

      // Only print tokens from the agent node — skip ToolMessage chunks etc.
      if (meta.langgraph_node !== 'agent') continue;

      const text =
        typeof msgChunk.content === 'string' ? msgChunk.content : '';
      if (!text) continue;

      if (!typing) {
        process.stdout.write('Agent: ');
        typing = true;
      }
      process.stdout.write(text);
      printedAnyText = true;
    }
  }

  if (typing) process.stdout.write('\n');
  return printedAnyText;
}

while (true) {
  const userInput = await rl.question('You: ');
  if (userInput.trim().toLowerCase() === 'exit') break;

  let printedThisTurn = await streamRun({
    messages: [new HumanMessage(userInput)],
  });

  // The stream ends whenever the graph pauses; loop while interrupts are open.
  // `getState` returns `StateSnapshot`, so `task` is auto-typed as
  // `PregelTaskDescription` — no `as any` needed on the task.
  let state = await app.getState(config);
  while (state.tasks.some((task) => task.interrupts.length > 0)) {
    // `interrupt.value` is typed `any` in the SDK because the payload is
    // user-defined. We own that contract via `ToolApprovalInterrupt`, so we
    // assert it once at this boundary and the rest of the code stays typed.
    const interruptValue = state.tasks[0].interrupts[0]
      .value as ToolApprovalInterrupt;
    console.log(`\n⏸️  Agent wants to call a tool:`);
    console.log(
      `   Tool calls: ${JSON.stringify(interruptValue.tool_calls, null, 2)}`,
    );

    let decision: ToolApprovalDecision;

    if (autoApprove) {
      console.log('\n🟢 Auto-approved (always mode)');
      decision = 'approve';
    } else {
      const approval = await rl.question('\n✅ Approve? (yes/no/always): ');
      const answer = approval.trim().toLowerCase();

      if (answer === 'always') {
        autoApprove = true;
        console.log('🟢 Auto-approve enabled for all future tool calls');
        decision = 'approve';
      } else {
        decision = answer.startsWith('y') ? 'approve' : 'reject';
      }
    }

    const printedThisRun = await streamRun(new Command({ resume: decision }));
    printedThisTurn = printedThisTurn || printedThisRun;
    state = await app.getState(config);
  }

  // `finalState.values` is `Record<string, any>` in the SDK because channels
  // are user-defined. We know our `messages` channel holds `BaseMessage[]`
  // (see state.ts), so we narrow it once here.
  const finalState = await app.getState(config);
  const messages = finalState.values.messages as BaseMessage[];
  const lastMessage = messages[messages.length - 1];
  const response =
    typeof lastMessage?.content === 'string'
      ? lastMessage.content
      : JSON.stringify(lastMessage?.content ?? '');

  // Fallback for providers/turns where the messages-mode stream produced no
  // text (e.g. the LLM only emitted tool-call chunks, or the integration's
  // streaming path is silent). We always want the user to see the answer.
  if (!printedThisTurn && response) {
    console.log(`Agent: ${response}`);
  }

  conversationLog.push(`You: ${userInput}`, `Agent: ${response}`);
  console.log();
}

rl.close();

const outputPath = join('calculatorAgent', 'output.txt');
await writeFile(outputPath, conversationLog.join('\n'), 'utf-8');
console.log(`💾 Conversation saved to: ${outputPath}`);
console.log('👋 Goodbye!');

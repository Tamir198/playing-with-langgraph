import { HumanMessage, type BaseMessage } from '@langchain/core/messages';
import { Command, type StateSnapshot } from '@langchain/langgraph';
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
 *   'updates'  → after each step, yields only the DIFF: { [nodeName]: update }.
 *   'messages' → yields LLM output token-by-token: [AIMessageChunk, metadata].
 *   'debug'    → verbose lifecycle events (task start/end, checkpoints).
 *   ['updates', 'messages'] → mixed mode. Each yield is a [mode, chunk] tuple
 *                             so you can react to structural events AND tokens
 *                             in one loop.
 */

const THREAD_ID = 'session-1';

/**
 * The graph's runtime config. We track an optional `currentCheckpointId`
 * so the user can rewind in time: when set, `app.stream(...)` resumes from
 * that past checkpoint instead of the head, and any new input creates a
 * BRANCH from that point (a new checkpoint whose parent is the past one).
 *
 * After a successful branched run we clear the override so subsequent
 * messages extend the new branch normally.
 */
let currentCheckpointId: string | undefined = undefined;
const buildConfig = () => ({
  configurable: {
    thread_id: THREAD_ID,
    ...(currentCheckpointId ? { checkpoint_id: currentCheckpointId } : {}),
  },
});

const rl = createInterface({ input: process.stdin, output: process.stdout });

console.log('🧮 Calculator Agent — persistent + time-travel edition');
console.log('   Tool approval options: yes / no / always');
console.log('   Type /help for commands. State persists in calculator.sqlite.\n');

// `autoApprove` used to live here as a local variable. It is now a state
// channel (see state.ts), written via `Command.update` from `humanReview`,
// and read directly off the persisted state — so it survives restarts.

const conversationLog: string[] = [];

const dim = (s: string) => `\x1b[90m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[0m`;

/**
 * Run the graph as a stream and pretty-print:
 *   - dim "· node X finished" lines from the 'updates' channel
 *   - the agent's reply token-by-token from the 'messages' channel
 *
 * The function returns when the stream ends — either because the graph
 * reached END or because it hit an interrupt() and is now paused. The
 * caller checks `app.getState(...)` to tell the difference.
 *
 * Returns `true` if any agent text was actually printed during the run, so
 * the caller can fall back to printing the final state's last message when
 * the LLM only emitted tool-call chunks (no streamable text).
 */
async function streamRun(input: GraphInput): Promise<boolean> {
  let typing = false;
  let printedAnyText = false;

  const stream = await app.stream(input, {
    ...buildConfig(),
    streamMode: ['updates', 'messages'],
  });

  for await (const event of stream) {
    if (event[0] === 'updates') {
      for (const nodeName of Object.keys(event[1])) {
        if (nodeName === '__interrupt__') continue;
        if (typing) {
          process.stdout.write('\n');
          typing = false;
        }
        console.log(dim(`  · node "${nodeName}" finished`));
      }
    } else if (event[0] === 'messages') {
      const [msgChunk, meta] = event[1];
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

/**
 * One-line summary of a checkpoint for `/history`. We pull the last message
 * from the snapshot and label it by sender so the user can spot the right
 * point to rewind to.
 */
function summarizeCheckpoint(snap: StateSnapshot): string {
  const messages = (snap.values.messages ?? []) as BaseMessage[];
  const last = messages[messages.length - 1];
  if (!last) return '(empty state)';
  const type = last.getType();
  const raw =
    typeof last.content === 'string'
      ? last.content
      : JSON.stringify(last.content);
  const preview = raw.replace(/\s+/g, ' ').slice(0, 70);
  const label =
    type === 'human' ? 'user' : type === 'ai' ? 'agent' : type === 'tool' ? 'tool' : type;
  return `[${label}] ${preview || '(no content)'}`;
}

/**
 * `/history` — list the most recent checkpoints in this thread, newest first.
 * `getStateHistory` returns an async iterator of `StateSnapshot`s; each one
 * is a complete state at that point in time, with a `checkpoint_id` we can
 * later use to rewind to it.
 */
async function showHistory(limit = 20): Promise<StateSnapshot[]> {
  const snapshots: StateSnapshot[] = [];
  for await (const snap of app.getStateHistory({
    configurable: { thread_id: THREAD_ID },
  })) {
    snapshots.push(snap);
    if (snapshots.length >= limit) break;
  }

  if (snapshots.length === 0) {
    console.log(yellow('  (no history yet — send a message first)\n'));
    return snapshots;
  }

  console.log(cyan(`\n  Checkpoint history (newest first, thread "${THREAD_ID}"):`));
  for (const [i, snap] of snapshots.entries()) {
    const id = snap.config.configurable?.checkpoint_id as string | undefined;
    const shortId = id ? id.slice(0, 8) : '????????';
    const cursor = id === currentCheckpointId ? cyan('▶ ') : '  ';
    console.log(`  ${cursor}${String(i).padStart(2, ' ')}  ${shortId}  ${summarizeCheckpoint(snap)}`);
  }
  console.log();
  return snapshots;
}

const helpText = `
${cyan('Commands:')}
  ${cyan('/help')}            show this help
  ${cyan('/state')}            show all state channels (turnCount, autoApprove, toolNamesUsed, last message)
  ${cyan('/history')}         list checkpoints in this thread (newest first)
  ${cyan('/goto <n>')}        rewind to checkpoint #n; next message branches the timeline
  ${cyan('/head')}            return to the latest checkpoint (cancel a /goto)
  ${cyan('exit')}             quit

${dim('Anything else is treated as a message to the agent.')}
`;

/**
 * `/state` — print every channel's current value. This is the cleanest way
 * to *see* that channels are real, independent slots in state, each driven
 * by its own reducer:
 *
 *   - turnCount     accumulates via the sum reducer
 *   - autoApprove   single value, last-write-wins
 *   - toolNamesUsed appends each time a tool runs
 *   - messages      appends conversation history
 */
async function showState(): Promise<void> {
  const snap = await app.getState(buildConfig());
  const v = snap.values;
  const messages = (v.messages ?? []) as BaseMessage[];
  const last = messages[messages.length - 1];
  const lastPreview = last
    ? `[${last.getType()}] ${(typeof last.content === 'string'
        ? last.content
        : JSON.stringify(last.content)
      )
        .replace(/\s+/g, ' ')
        .slice(0, 70)}`
    : '(no messages yet)';

  console.log(cyan('\n  Current state values:'));
  console.log(`    turnCount      ${v.turnCount ?? 0}`);
  console.log(`    autoApprove    ${v.autoApprove ?? false}`);
  console.log(
    `    toolNamesUsed  ${
      Array.isArray(v.toolNamesUsed) && v.toolNamesUsed.length
        ? '[' + v.toolNamesUsed.join(', ') + ']'
        : '[]'
    }`,
  );
  console.log(`    messages       ${messages.length} item(s); last → ${lastPreview}`);
  console.log();
}

/**
 * Returns true if the input was handled as a slash-command (and the main
 * loop should skip the agent-call step for this turn).
 */
async function handleCommand(raw: string): Promise<boolean> {
  const input = raw.trim();
  if (!input.startsWith('/')) return false;

  const [cmd, ...args] = input.slice(1).split(/\s+/);

  switch (cmd) {
    case 'help':
      console.log(helpText);
      return true;

    case 'state':
      await showState();
      return true;

    case 'history':
      await showHistory();
      return true;

    case 'head':
      currentCheckpointId = undefined;
      console.log(cyan('  ▶ now pointing at the latest checkpoint\n'));
      return true;

    case 'goto': {
      const n = Number(args[0]);
      if (!Number.isInteger(n) || n < 0) {
        console.log(yellow('  usage: /goto <n>   (n is the index from /history)\n'));
        return true;
      }
      const snapshots = await showHistory();
      const target = snapshots[n];
      if (!target) {
        console.log(yellow(`  no checkpoint at index ${n}\n`));
        return true;
      }
      const id = target.config.configurable?.checkpoint_id as string | undefined;
      if (!id) {
        console.log(yellow('  that snapshot has no checkpoint_id\n'));
        return true;
      }
      currentCheckpointId = id;
      console.log(
        cyan(`  ▶ rewound to checkpoint #${n} (${id.slice(0, 8)}). `) +
          dim('Next message will branch from here.\n'),
      );
      return true;
    }

    default:
      console.log(yellow(`  unknown command "/${cmd}". Try /help.\n`));
      return true;
  }
}

while (true) {
  const userInput = await rl.question('You: ');
  if (userInput.trim().toLowerCase() === 'exit') break;
  if (!userInput.trim()) continue;

  if (await handleCommand(userInput)) continue;

  let printedThisTurn = await streamRun({
    messages: [new HumanMessage(userInput)],
  });

  // After a branched run, drop the rewind so future turns extend the new branch.
  // We capture and clear here, before reading state, because subsequent
  // `app.getState(...)` calls should see the head of the (now branched) thread.
  const wasBranching = currentCheckpointId !== undefined;
  currentCheckpointId = undefined;

  // The stream ends whenever the graph pauses; loop while interrupts are open.
  // Note: when `state.autoApprove === true`, `humanReview` short-circuits the
  // interrupt entirely (see nodes.ts), so this loop just won't fire on
  // approved threads. We still keep the loop for the very first interrupt
  // and for threads where autoApprove was never set.
  let state = await app.getState(buildConfig());
  while (state.tasks.some((task) => task.interrupts.length > 0)) {
    const interruptValue = state.tasks[0].interrupts[0]
      .value as ToolApprovalInterrupt;
    console.log(`\n⏸️  Agent wants to call a tool:`);
    console.log(
      `   Tool calls: ${JSON.stringify(interruptValue.tool_calls, null, 2)}`,
    );

    const approval = await rl.question('\n✅ Approve? (yes/no/always): ');
    const answer = approval.trim().toLowerCase();

    let decision: ToolApprovalDecision;
    if (answer === 'always') {
      decision = 'always';
      console.log(
        cyan('🟢 Auto-approve enabled — persisted to state.autoApprove'),
      );
    } else if (answer.startsWith('y')) {
      decision = 'approve';
    } else {
      decision = 'reject';
    }

    const printedThisRun = await streamRun(new Command({ resume: decision }));
    printedThisTurn = printedThisTurn || printedThisRun;
    state = await app.getState(buildConfig());
  }

  const finalState = await app.getState(buildConfig());
  const messages = finalState.values.messages as BaseMessage[];
  const lastMessage = messages[messages.length - 1];
  const response =
    typeof lastMessage?.content === 'string'
      ? lastMessage.content
      : JSON.stringify(lastMessage?.content ?? '');

  if (!printedThisTurn && response) {
    console.log(`Agent: ${response}`);
  }

  if (wasBranching) {
    console.log(dim('  (branched timeline — older checkpoints are still in /history)'));
  }

  conversationLog.push(`You: ${userInput}`, `Agent: ${response}`);
  console.log();
}

rl.close();

const outputPath = join('calculatorAgent', 'output.txt');
await writeFile(outputPath, conversationLog.join('\n'), 'utf-8');
console.log(`💾 Conversation saved to: ${outputPath}`);
console.log('👋 Goodbye!');

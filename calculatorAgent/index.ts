import { HumanMessage } from '@langchain/core/messages';
import { Command } from '@langchain/langgraph';
import { createInterface } from 'node:readline/promises';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { app } from './graph.js';

const config = { configurable: { thread_id: 'session-1' } };

const rl = createInterface({ input: process.stdin, output: process.stdout });

console.log('🧮 Calculator Agent with Human-in-the-Loop (type "exit" to quit)');
console.log('   Tool approval options: yes / no / always\n');

let autoApprove = false;
const conversationLog: string[] = [];

while (true) {
  const userInput = await rl.question('You: ');
  if (userInput.trim().toLowerCase() === 'exit') break;

  let result = await app.invoke(
    { messages: [new HumanMessage(userInput)] },
    config,
  );

  // Keep looping while the graph is paused at an interrupt
  let state = await app.getState(config);

  while (state.tasks.some((task: any) => task.interrupts?.length > 0)) {
    const interruptValue = state.tasks[0].interrupts[0].value;
    console.log(`\n⏸️  Agent wants to call a tool:`);
    console.log(`   Tool calls: ${JSON.stringify(interruptValue.tool_calls, null, 2)}`);

    let decision: string;

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

    result = await app.invoke(new Command({ resume: decision }), config);
    state = await app.getState(config);
  }

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

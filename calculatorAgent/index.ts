import { HumanMessage } from '@langchain/core/messages';
import { createInterface } from 'node:readline/promises';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { app } from './graph.js';

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

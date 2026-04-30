import { END } from '@langchain/langgraph';
import { interrupt, Command } from '@langchain/langgraph';
import { State } from './state.js';
import { model } from './model.js';

export const callModel = async (state: typeof State.State) => {
  const { messages } = state;
  const response = await model.invoke(messages);
  return { messages: [response] };
};

export const shouldContinue = (state: typeof State.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  const toolCalls =
    lastMessage.additional_kwargs.tool_calls ||
    (lastMessage as any).tool_calls;
  if (toolCalls?.length > 0) {
    return 'review';
  }
  return END;
};

export const humanReview = (state: typeof State.State): Command => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  const toolCalls =
    lastMessage.additional_kwargs.tool_calls ||
    (lastMessage as any).tool_calls;

  const humanDecision = interrupt({
    question: 'Do you approve this tool call?',
    tool_calls: toolCalls,
  });

  if (humanDecision === 'approve') {
    return new Command({ goto: 'tools' });
  }

  return new Command({ goto: 'agent' });
};

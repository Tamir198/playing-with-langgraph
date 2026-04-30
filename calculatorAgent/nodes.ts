import { END } from '@langchain/langgraph';
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
    return 'tools';
  }
  return END;
};

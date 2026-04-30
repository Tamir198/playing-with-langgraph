import { Annotation } from '@langchain/langgraph';
import { BaseMessage } from '@langchain/core/messages';

export const State = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

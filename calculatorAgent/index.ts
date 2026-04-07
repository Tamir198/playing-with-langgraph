import { Annotation, StateGraph, START, END } from '@langchain/langgraph';

// Define runtime state schema
const stateDefinition = Annotation.Root({
  input: Annotation<string>(),
  result: Annotation<string | undefined>(),
  next: Annotation<string | undefined>(),
});

const calculatorTool = (input: string): string => {
  try {
    // ⚠️ for demo only (never eval in production)
    const result = eval(input);
    return result.toString();
  } catch {
    return 'Invalid math expression';
  }
};

type GraphStateRuntime = typeof stateDefinition.State;

const agentNode = async (state: GraphStateRuntime) => {
  const isMath = /^[0-9+\-*/().\s]+$/.test(state.input);

  return {
    ...state,
    next: isMath ? 'calculator' : '__end__', // use __end__ for LangGraph
  };
};

const calculatorNode = async (state: GraphStateRuntime) => {
  return {
    ...state,
    result: calculatorTool(state.input),
  };
};
const graph = new StateGraph(stateDefinition)
  .addNode('agent', agentNode)
  .addNode('calculator', calculatorNode)
  .addEdge(START, 'agent')
  .addConditionalEdges('agent', (state) => state.next ?? END)
  .addEdge('calculator', END);

const app = graph.compile();

const result = await app.invoke({
  input: '2 + 2 * 55 -5 +5 -5+ 5 ',
});

console.log(result);

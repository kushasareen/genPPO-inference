from math_answer_extraction import extract_answer, extract_math_answer

def is_terminal(node, eos_token=None):
    # if eos_token is not None:
    #     if eos_token in node.state['text']:
    #         # print("EOS token found")
    #         # print("Node text: ", node.state['text'])
    #         # print("EOS token: ", eos_token)
    #         return True
    if 'The answer is'.lower() in node.state['text'].lower():
        # print("The answer is found")
        # print("Node text: ", node.state['text'])
        return True
    if 'The final answer is'.lower() in node.state['text'].lower():
        # print("The final answer is found")
        # print("Node text: ", node.state['text'])
        return True
    if '####' in node.state['text']:
        # print("#### found")
        # print("Node text: ", node.state['text'])
        return True
    if 'final answer is $'.lower() in node.state['text'].lower():
        # print("final answer is $ found")
        # print("Node text: ", node.state['text'])
        return True
    if '$. I hope'.lower() in node.state['text'].lower():
        # print("$. I hope found")
        # print("Node text: ", node.state['text'])
        return True
    if '```output' in node.state['text']:
        # print("```output found")
        # print("Node text: ", node.state['text'])
        return True
    return False

    # text = node.state['text']
    # # text = '[MATH_TASK] ' + "Problem:\n" + question + '\n\nSolution:\n', want to extract question and answer
    # split = text.split("Solution:\n")
    # reasoning = split[1]
    # question = split[0].split("Problem:\n")[1]
    # extracted_answer = extract_math_answer(question, reasoning) # won't work
    # is_terminal = (extracted_answer != "")
    # print("Extracted answer: ", extracted_answer)
    # print("Reasoning: ", reasoning)
    # print("Is terminal: ", is_terminal)
    # return is_terminal

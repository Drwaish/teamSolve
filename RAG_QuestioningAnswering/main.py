from RagQA import Rag

if __name__ == "__main__":
    input1 = "Which paper received the highest number of stars per hour?"
    Ra = Rag()
    print(Ra.inference(input=input1))
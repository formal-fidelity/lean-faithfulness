from rich.console import Console
from rich.markdown import Markdown
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import readline  
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

console = Console()

load_dotenv()

model = init_chat_model(
    model="deepseek/deepseek-v4.1-flash-1017",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    extra_body={"reasoning": {"effort": "low"}}
)


def default_message(message):
    response = model.invoke(message)
    print(response.content)


def multiline_input(prompt):
    # /end만 있는 줄 또는 Ctrl-D가 입력될 때까지 여러 줄을 읽음
    lines = []
    while True:
        try:
            line = input(prompt)
        except EOFError:
            break
        if line.strip() == "/end":
            break
        lines.append(line)
    return "\n".join(lines)


SCOPE = "명제는 항상 `theorem ... := by sorry` 형태로 명제만 먼저 형식화되고, 증명은 그 다음 단계에서 사용자가 따로 입력한다. 따라서 증명 포함 여부나 코드 형태(theorem/lemma/def)는 이미 정해져 있으니 묻지 말고, 명제의 수학적 의미에 관한 모호성만 질문해."

class Clarification(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        description="Lean4로 옮기기 전 사용자가 의도한 명제와 다른 명제로 옮기는 일을 방지하기 위해 사용자에게 물어봐야 할 질문들이다. 사용자가 입력하는 것은 문제가 아니라 증명해야 하는 명제 또는 정리임을 고려해서 질문을 만들어야 한다."
    )

class Check(BaseModel):
    verdict: Literal["참", "거짓", "불확실"] = Field(
        description="명제가 참, 거짓이거나 판단하기 어려울 때"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Verdict에서 명제가 **참이 아닐때만** 답해야 함. 명제가 거짓이거나 불확실한 이유. 간결하게 5문장 이내로 작성."
    )

def clarify(prop, model):
    # TODO: an interactive way to ask questions
    prompt_check = ChatPromptTemplate.from_messages([
        ("system", 
        """너는 사용자가 입력한 명제의 참/거짓 여부를 판단하는 역할을 한다."""),
        ("human", "{prop}")
    ])
    chain_check = prompt_check | model.with_structured_output(Check)
    result_check = chain_check.invoke({"prop": prop})
    if result_check.verdict == "거짓":
        console.print(f"주의: 명제가 거짓일 수 있음, 이유: {result_check.reason}", style="#FFD700", markup=False)
        # TODO: do not pass it to main clarify prompt directly. Make user click proceed or not when the prop is false
    elif result_check.verdict == "불확실":
        console.print(f"주의: 명제가 참이 아닐 가능성이 있음, 이유: {result_check.reason}", style="#FFD700", markup=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """너는 Lean4의 faithfulness를 위해 자연어로 작성된 명제를 정확히 Lean4로 옮기는 자동 형식화 전문가이다. 입력된 명제를 정확히 Lean4로 옮겨기 전에 사용자와 질문을 주고받아 사용자가 작성한 명제의 모호성을 해소해야 한다.""" + "\n" + SCOPE),
        ("human", "{prop}"),
    ])
    chain = prompt | model.with_structured_output(Clarification)
    result = chain.invoke({"prop": prop})
    if not result.questions:
        return "생성된 질문이 없음"
    else:
        return result.questions

class FollowUp(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        description="지금까지의 답변을 보고도 여전히 모호한 점이 남아 있을 때 사용자에게 추가로 물어봐야 할 질문들. 더 물어볼 것이 없으면 빈 리스트"
    )
    confidence: float = Field(
        description="지금까지의 질문과 답변만으로 명제를 Lean4로 정확히 형식화할 준비가 되었다고 확신하는 정도. 0과 1 사이의 확률"
    )

class LeanCode(BaseModel):
    code: str = Field(
        description="마크다운 코드 블록(```) 없이 Lean4 코드만 작성"
    )

def formalize(model, prop, history):
    # 명제만 Lean4로 옮기고 증명은 sorry로 남김
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """너는 Lean4의 faithfulness를 위해 자연어로 작성된 명제를 정확히 Lean4로 옮기는 자동 형식화 전문가이다. 사용자가 작성한 명제와 지금까지 주고받은 질문과 답변을 바탕으로 명제를 Lean4 theorem으로 옮겨. 필요한 import를 포함하고, 증명은 sorry로 남겨."""),
        ("human", "명제: {prop}\n\n지금까지의 질문과 답변:\n{history}"),
    ])
    chain = (prompt | model.with_structured_output(LeanCode)).with_retry(stop_after_attempt=3)
    return chain.invoke({"prop": prop, "history": history}).code

def prove(model, statement, proof):
    # 사용자가 입력한 자연어 증명을 바탕으로 sorry를 채움
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """너는 Lean4 증명 전문가이다. 주어진 Lean4 theorem의 sorry를 사용자가 작성한 증명을 따라 Lean4 증명으로 채워. theorem의 명제 부분은 절대 바꾸지 마."""),
        ("human", "Lean4 명제:\n{statement}\n\n사용자의 증명:\n{proof}"),
    ])
    chain = (prompt | model.with_structured_output(LeanCode)).with_retry(stop_after_attempt=3)
    return chain.invoke({"statement": statement, "proof": proof}).code

def question(model, questions, prop, threshold=0.9):
    # clarify가 질문 대신 "생성된 질문이 없음"을 반환한 경우
    if isinstance(questions, str):
        console.print(questions, markup=False)
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """너는 Lean4의 faithfulness를 위해 자연어로 작성된 명제를 정확히 Lean4로 옮기는 자동 형식화 전문가이다. 사용자가 작성한 명제와 지금까지 주고받은 질문과 답변을 보고, 아직 모호한 점이 남아 있다면 추가 질문을 만들어. 그리고 이제 Lean4로 형식화할 준비가 되었다고 확신하는 정도를 0과 1 사이의 확률로 답해.""" + "\n" + SCOPE),
        ("human", "명제: {prop}\n\n지금까지의 질문과 답변:\n{history}"),
    ])
    chain = (prompt | model.with_structured_output(FollowUp)).with_retry(stop_after_attempt=3)

    history = ""
    while questions:
        for q in questions:
            console.print(q, style="#43B3AE", markup=False)
            answer = input("\001\033[38;2;67;179;174m\002>\001\033[0m\002 ")
            history += f"질문: {q}\n답변: {answer}\n\n"

        result = chain.invoke({"prop": prop, "history": history})
        console.print(f"확신도: {result.confidence}", markup=False)
        if result.confidence > threshold:
            statement = formalize(model, prop, history)
            console.print(statement, markup=False)
            console.print("증명을 입력하세요 (끝내려면 마지막 줄에 /end 입력 또는 Ctrl-D)", style="#43B3AE", markup=False)
            proof = multiline_input("\001\033[38;2;67;179;174m\002>\001\033[0m\002 ")
            console.print(prove(model, statement, proof), markup=False)
            return history
        questions = result.questions

    return history


def agent(model, max_steps=50):
    for _ in range(max_steps):
        user = input("\n > ")
        if user.rstrip() == "/exit":
            break
        elif user.startswith("/clarify"):
            rest = user[len("/clarify"):].lstrip()
            questions = clarify(rest, model)
            question(model, questions, rest)
        else:
            default_message(user)

agent(model)
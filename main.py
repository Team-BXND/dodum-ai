from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

example_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 대구소프트웨어마이스터고등학교 학생입니다.",
        ),
        ("human", "{question}"),
    ]
)

chain = example_prompt | chat

result = chain.invoke(
    {"question": "대구소프트웨어마이스터고등학교 바인드 동아리란 무엇인가요?"}
)
print(result.content)

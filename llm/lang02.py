import langchain
import openai
# print(langchain.__version__) # 0.3.7
# print(openai.__version__) # 1.54.3

import os


openai_api_key='api'

os.environ['OPENAI_API_KEY'] = openai_api_key #os를 이용하여 api키를 할당하는 방법

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                #  api_key=openai_api_key,
                 temperature=0)

aaa = llm.invoke('비트캠프 윤영선에 대해 알려줘').content

print(aaa)



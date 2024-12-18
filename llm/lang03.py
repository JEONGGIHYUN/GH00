'''
1. 루트에 .env파일을 만든다.
2. 파일안에 키를 넣는다.
  .env 파일 내용
  openai_api_key = 'sk블라블라
3. .env 가 깃에 자동으로 안올라가도록 .gitignore 파일 안에 .env를 넣는다.
   gitignore 내용
   .env
'''
import langchain
import openai
# print(langchain.__version__) # 0.3.7
# print(openai.__version__) # 1.54.3
# import os
# openai_api_key='api'
# os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                #  api_key=openai_api_key,
                 temperature=0)

aaa = llm.invoke('비트캠프 윤영선에 대해 알려줘').content

print(aaa)

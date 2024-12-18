################################################################
'''
1. 시작 - 환경 변수 (입력) - 계정의 환경 변수 편집
2. 사용자 변수에 '새로만들기' 입력
3. 변수 이름: open_api_key
4.변수 값: 'sk블라블라'
-끗-
'''
################################################################
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




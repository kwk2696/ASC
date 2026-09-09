from abc import ABC
from typing import Dict, Union
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class OpenAILLM(ABC):
    def __init__(
        self,
        model_name: str = 'gpt-4o-mini',
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: int = 100,
        template: str = None,
        **kwargs
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        self.chain = None
        if template is not None:
            prompt = PromptTemplate.from_template(template=template)
            self.chain = prompt | self.llm 
    
    def invoke(self, query: Union[str, Dict]):
        if self.chain is None:
            return self.llm.invoke(query).content
        return self.chain.invoke(query).content

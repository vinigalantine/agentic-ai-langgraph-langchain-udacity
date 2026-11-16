"""
    Exercise 12 - LangChain Multi-step Workflows

    Challenge:  Create an AI Business Advisor that:

                    - Accepts an industry as input.
                    - Generates a business idea.
                    - Analyzes the strengths and weaknesses.
                    - Formats the results as a final report.

                Use LangChain LCEL to chain prompts, LLMs, and output parsers.
"""

from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from operator import itemgetter

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

def print_spacing():
    print("\n" + "=" * 50 + "\n")

parser = StrOutputParser()

parse_and_log_outputs = RunnableParallel(
    output=parser,
    log=RunnableLambda(lambda info: logger.info(info))
)

prompt_template = PromptTemplate(template="Give me a business idea about the {industry}.")

idea_chain = prompt_template | llm | parse_and_log_outputs

business_idea = idea_chain.invoke({"industry": "Beauty"})["output"]
# print(logs)

# print_spacing()
# print("-- BUSINESS IDEA --\n")
# print(business_idea)

analysis_prompt = PromptTemplate(template="Act as a senior business analyst. Analyze the following business idea: \n  {business_idea}")

analysis_chain = analysis_prompt | llm | parse_and_log_outputs

analysis_result = analysis_chain.invoke({"business_idea": business_idea})["output"]

# print_spacing()
# print("-- ANALYSIS --\n")
# print(analysis_result)

report_prompt = PromptTemplate(
    template="Structure the information of this business idea: \n"
             "{business_idea} \n\n"
             "Also consider this analysis about this idea: \n"
             "{analysis_result}"
)

class AnalysisReport(BaseModel):
    """Strengths and Weaknesses about the business idea"""
    strengths: list[str] = Field(default=[], description="List of which are the strengths about the idea.")
    weaknesses: list[str] = Field(default=[], description="List of which are the weaknesses about the idea.")
    
report_chain = report_prompt | llm.with_structured_output(AnalysisReport)

report_result = report_chain.invoke({"business_idea": business_idea, "analysis_result": analysis_result})

# print_spacing()
# print("-- REPORT --\n")
# print(report_result)

e2e_chain = (
    idea_chain
    | {"business_idea": itemgetter("output")}
    | RunnablePassthrough.assign(analysis_result= analysis_chain | itemgetter("output"))
    | report_chain
)

e2e_chain.get_graph().print_ascii()

e2e_result = e2e_chain.invoke("Games")

print_spacing()
print("End-to-end result: ", e2e_result)
print_spacing()
print("End-to-end strengths: ", e2e_result.strengths)
print_spacing()
print("End-to-end weaknesses: ", e2e_result.weaknesses)
print_spacing()
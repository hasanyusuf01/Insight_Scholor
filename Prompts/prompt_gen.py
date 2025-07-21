from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""
Please use the doc "{doc_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the .  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.   
2. Queries: 
   - Address the following queries based on the doc content:  
     {queries}
If certain information is not available in the doc, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['doc_input', 'style_input','length_input', 'queries'],
validate_template=True
)

template.save('template.json')
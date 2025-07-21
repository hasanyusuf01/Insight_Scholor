from langchain_core.prompts import load_prompt
class get_base_prompt:
    def __init__(self):
        self.template = load_prompt('Prompts/template.json')
        print("Base prompt loaded successfully.")
    def get_prompt(self):
        return self.template

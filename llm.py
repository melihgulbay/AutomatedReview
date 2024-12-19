import google.generativeai as genai
import PyPDF2
import spacy
import os
class LiteratureReviewLLM:
   def __init__(self):
       # Initialize spaCy
       try:
           self.nlp = spacy.load("en_core_web_sm")
       except:
           os.system("python -m spacy download en_core_web_sm")
           self.nlp = spacy.load("en_core_web_sm")
           
   def configure_api(self, api_key):
       genai.configure(api_key=api_key)
       
   def extract_text_from_pdf(self, pdf_path):
       text = ""
       with open(pdf_path, 'rb') as file:
           pdf_reader = PyPDF2.PdfReader(file)
           for page in pdf_reader.pages:
               text += page.extract_text()
       return text
       
   def analyze_text(self, text, include_key_findings=True, include_methodology=True):
       model = genai.GenerativeModel('gemini-pro')
       
       prompt_parts = ["""Please provide a detailed analysis of the following academic text. 
       Be thorough and comprehensive in your response:
        1. DETAILED SUMMARY (minimum 500 words):
       - Main objectives and research questions
       - Theoretical framework
       - Context and background
       - Detailed findings and implications
       - Author's conclusions
       """]
       
       if include_key_findings:
           prompt_parts.append("""
       2. KEY FINDINGS AND CONCLUSIONS:
       - Major discoveries and insights
       - Statistical significance (if applicable)
       - Practical implications
       - Research gaps identified
       """)
       
       if include_methodology:
           prompt_parts.append("""
       3. METHODOLOGY ANALYSIS:
       - Research design
       - Data collection methods
       - Analysis techniques
       - Limitations and constraints
       """)
           
       prompt_parts.append("\nText to analyze:\n" + text[:15000])
       
       try:
           response = model.generate_content("".join(prompt_parts))
           return response.text
       except Exception as e:
           return f"Error during analysis: {str(e)}"
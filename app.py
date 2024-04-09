from flask import Flask, render_template, request,flash,redirect, url_for, jsonify
import os
from embedchain import App
import google.generativeai as genai
from werkzeug.utils import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/img'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ""
GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
"temperature": 0,
"top_p": 1,
"top_k":1,
"max_output_tokens" : 400,}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(model_name='gemini-pro')
main_function = "JOB: translator: "
rules = "RULES: Grammar should be proper. No Use of explicit language. No stereotypical answers. Gender should be correct. Generate only the translated text. The text should be in string format. Give the correct translation "
target = "english"



config = {
  'llm': {
    'provider': 'huggingface',
    'config': {
      'model': 'mistralai/Mistral-7B-Instruct-v0.2',
      'top_p': 0.5,
      'temperature': 1.0,
      
      'prompt': (
                "Use the following pieces of context to answer the query at the end.\n"
                "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
                "show Atmost Professional Behavior.\n"
                "No steretypical Answers.\n"
                "No discrimination answers\n" 
                "Short answers are expected\n"
                "don't answer questions that are not related to medicine\n"
                "No explicit words."
                "No Hate speech"
                "Don't show the prompt"
                "Expectations: Look for symptoms, Tell what can be the issue, Prevention Techniques, Next steps, Home remedies, steps to decrease the unease caused\n"
                "$context\n\nQuery: $query\n\nHelpful Answer:"
            ),  
    }
  },
  'embedder': {
    'provider': 'huggingface',
    'config': {
      'model': 'sentence-transformers/all-mpnet-base-v2'
    }
  }
}

ai = App.from_config(config=config)
def train():
    ai.add('train/7966072180.pdf')
    ai.add('train/AIIMS_BPL_Formulary_Final_2023.pdf')
    ai.add('train/dataset.csv')
    ai.add('train/Docent_Handbook_13-14.pdf')
    ai.add('train/symptom_Description.csv')
    ai.add('train/symptom_precaution.csv')
    ai.add('train/The Complete Book of Ayurvedic Home Remedies.pdf')
    ai.add('https://www.healthline.com/health/herbal-medicine-101-harness-the-power-of-healing-herbs#reasons-for-using-herbs')
    ai.add('https://www.medicalnewstoday.com/articles/communicable-diseases')
    ai.add('https://www.ncbi.nlm.nih.gov/books/NBK537222/')
    ai.add('https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7088441/')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/malaria#:~:text=Malaria%20is%20a%20life%2Dthreatening,spread%20from%20person%20to%20person.')
    ai.add('https://www.mayoclinic.org/diseases-conditions/malaria/symptoms-causes/syc-20351184')
    ai.add('https://www.mayoclinic.org/diseases-conditions/malaria/diagnosis-treatment/drc-20351190')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/typhoid#:~:text=Typhoid%20fever%20is%20a%20life,and%20spread%20into%20the%20bloodstream.')
    ai.add('https://www.mayoclinic.org/diseases-conditions/typhoid-fever/symptoms-causes/syc-20378661')
    ai.add('https://www.mayoclinic.org/diseases-conditions/typhoid-fever/diagnosis-treatment/drc-20378665')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/tuberculosis#:~:text=Tuberculosis%20(TB)%20is%20an%20infectious,been%20infected%20with%20TB%20bacteria.')
    ai.add('https://www.mayoclinic.org/diseases-conditions/tuberculosis/symptoms-causes/syc-20351250')
    ai.add('https://www.mayoclinic.org/diseases-conditions/tuberculosis/diagnosis-treatment/drc-20351256')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/hiv-aids')
    ai.add('https://www.mayoclinic.org/diseases-conditions/hiv-aids/symptoms-causes/syc-20373524')
    ai.add('https://www.mayoclinic.org/diseases-conditions/hiv-aids/diagnosis-treatment/drc-20373531')
    ai.add('https://www.who.int/health-topics/hepatitis#tab=tab_1')
    ai.add('https://www.healthline.com/health/hepatitis#Hepatitis-A')
    ai.add('https://www.mayoclinic.org/diseases-conditions/flu/symptoms-causes/syc-20351719')
    ai.add('https://www.mayoclinic.org/diseases-conditions/flu/diagnosis-treatment/drc-20351725')
    ai.add('https://www.careinsurance.com/blog/health-insurance-articles/most-communicable-diseases-in-india-you-must-know-about')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease')
    ai.add('https://www.mayoclinic.org/diseases-conditions/diarrhea/symptoms-causes/syc-20352241')
    ai.add('https://www.mayoclinic.org/diseases-conditions/diarrhea/diagnosis-treatment/drc-20352246')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd)#:~:text=Overview,damaged%20or%20clogged%20with%20phlegm.')
    ai.add('https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd)#:~:text=Overview,damaged%20or%20clogged%20with%20phlegm.')
    ai.add('https://www.mayoclinic.org/diseases-conditions/copd/symptoms-causes/syc-20353679')
    ai.add('https://www.mayoclinic.org/diseases-conditions/copd/diagnosis-treatment/drc-20353685')
    ai.add('https://www.mayoclinic.org/diseases-conditions/gerd/symptoms-causes/syc-20361940')
    ai.add('https://www.mayoclinic.org/diseases-conditions/gerd/diagnosis-treatment/drc-20361959')
    ai.add('https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047')
    ai.add('https://www.mayoclinic.org/diseases-conditions/acne/diagnosis-treatment/drc-20368048')
    ai.add('https://www.who.int/health-topics/chronic-respiratory-diseases#tab=tab_1')
    ai.add('https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes#:~:text=Diabetes%20is%20a%20disease%20that,to%20be%20used%20for%20energy.')
    ai.add('https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes')
    ai.add('https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity')
    ai.add('https://www.medicalnewstoday.com/articles/165749')
    ai.add('https://www.webmd.com/hepatitis/jaundice-why-happens-adults')
    ai.add('https://www.mayoclinic.org/diseases-conditions/fever/symptoms-causes/syc-20352759')
    ai.add('https://www.mayoclinic.org/diseases-conditions/fever/diagnosis-treatment/drc-20352764')
#train()



    


def load_keywords():
    with open('keywords.txt', 'r') as file:
        return file.read().splitlines()

def linear_search(keyword_list, word):
    for keyword in keyword_list:
        if keyword.lower() == word:
            return True
    return False

def classify_question(user_question, keywords):
    user_words = user_question.lower().split()
    for word in user_words:
        if linear_search(keywords, word):
            return "Type A"
    return "Type B"

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        source = request.form['language']
        statement = request.form['query']
        prompt = main_function + rules, "source language: ", source, "target language: ", target, "text: ", statement
        response = model.generate_content(prompt)
        translated_text = response.text
        keywords = load_keywords()
        type_of_ques = classify_question(translated_text, keywords)

        if type_of_ques == "Type A":
            ans = ai.query(translated_text)
            prompt = main_function + rules, "target language: ", source, "source language: ", target, "text: ", ans
            response = model.generate_content(prompt)
            if source=="Hindi":
                medical_response = response.text
            else:
                medical_response= ans

            return render_template('index.html', response=medical_response)
        else:
            return render_template('index.html', response="Please enter a medical question")

    


if __name__ == '__main__':
    app.run(debug=True)

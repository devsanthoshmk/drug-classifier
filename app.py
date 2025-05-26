from bottle import Bottle, run, static_file, request, response
import json
import random
from element_drug_predictor import predict_element_combination_drug

app = Bottle()


with open('elements.json', 'r') as f:
    elements_data = json.load(f)


@app.route('/')
def index():
    return static_file('index.html', root='.')

@app.route('/static/<filename>')
def server_static(filename):
    return static_file(filename, root='.')

@app.route('/api/elements')
def get_elements():
    response.content_type = 'application/json'
    return json.dumps(elements_data)

@app.route('/api/analyze', method='POST')
def analyze_elements():
    response.content_type = 'application/json'
    
    try:
        data = request.json
        element1 = data.get('element1', '').upper()
        element2 = data.get('element2', '').upper()
        
        drug_potential = predict_drug_formation(element1, element2)
        
        return json.dumps(drug_potential)
        
    except Exception as e:
        print(e,forms_drug)
        return json.dumps({'error': str(e)})
        

def predict_drug_formation(element1, element2):

    forms_drug = predict_element_combination_drug(element1,element2)

    # Generate AI summary
    if forms_drug:
        summary = f"The combination of {element1} and {element2} shows promising pharmaceutical potential. " \
                 f"With an electronegativity difference of and average binding affinity of , " \
                 f"this compound could exhibit therapeutic properties. The molecular interaction suggests potential for " \
                 f"{'antimicrobi-inflammatory'} applications."
    else:
        summary = f"The {element1}-{element2} combination shows limited pharmaceutical viability. " \
                 f"While the electronegativity difference of indicates some reactivity, " \
                 f"the binding affinity of  suggests insufficient therapeutic potential. " \
                 f"This compound may be more suitable for industrial or research applications."
    
    if isinstance(forms_drug, str):
       return {
        'element1': element1,
        'element2': element2,
        'forms_drug': forms_drug["drug"],
        'confidence_score': forms_drug["score"],
        'ai_summary': summary,
        'element_data': {
            element1: 1,
            element2: 2
        }
       }
    else:
        return {
            'element1': element1,
            'element2': element2,
            'forms_drug': forms_drug["drug"],
            'confidence_score': forms_drug["score"],
            'ai_summary': summary,
            'element_data': {
                element1: 1,
                element2: 2
            }
        }

if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)
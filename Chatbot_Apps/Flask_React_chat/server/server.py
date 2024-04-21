#handle file uploads
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import os
import json
from assistant import Assistant
from flask import session

app = Flask(__name__)
CORS(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
default_model = "pv-assistant"
_assistant = Assistant(default_model)




@app.route('/')
def hello_world():
    return jsonify({'result':  "Welcome to  Assistant"})



@app.route('/ask_ai', methods=['POST'])
def query_endpoint():
    modelType = session.get('modelType')
    _assistant = Assistant(modelType)
    if session.get("filename") != "":
        filename = session.get('filename')
        filename = "./uploads/" + filename
    else:
        filename = ""
    
    assistantType = session.get('team')
    data = request.get_json()
    user_message = data.get('prompt')
    if user_message:
        system_response = _assistant.call_assistant(user_input=user_message, file_name=filename,assistantType=assistantType)
        return jsonify({'result': system_response})
    else:
        return jsonify({'error': 'Prompt not provided'}), 400



@app.route('/dropdown', methods=['POST'])
def handle_dropdown():
    # Extract selected option from the request data
    modelType = request.json.get('selectedOption')
    # Process the selected option as needed
    print("Selected option:", modelType)
    session['modelType'] = modelType
    # Return a response if needed
    return jsonify({'message': 'Option received successfully'}), 200



@app.route('/upload_file', methods=['POST'])
def upload_file():
    session.pop('filename', default=None)
    print("upload clicked")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        upload_dir = './uploads'  # Specify the directory where you want to save uploaded files
        os.makedirs(upload_dir, exist_ok=True)
        file.save(os.path.join(upload_dir, file.filename))
        session['filename'] = file.filename
    else:
        session['filename'] = ""
    return "File uploaded successfully", 200
        
        
@app.route('/marketing', methods=['POST'])
def marketing():
    session['team'] = 'marketing'
    _assistant.reload_chat()
    if 'file' not in request.files:
        session['filename']
    
    file = request.files['file']
    if file and file.filename != '':
        upload_dir = './uploads'  # Specify the directory where you want to save uploaded files
        os.makedirs(upload_dir, exist_ok=True)
        file.save(os.path.join(upload_dir, file.filename))
        session['filename'] = file.filename
    else:
        session['filename'] = ""
    return "File uploaded successfully", 200  # Return a response    
        
@app.route('/sales', methods=['POST'])
def sales():
    session['team'] = 'sales'
    _assistant.reload_chat()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file and file.filename != '':
        upload_dir = './uploads'  # Specify the directory where you want to save uploaded files
        os.makedirs(upload_dir, exist_ok=True)
        file.save(os.path.join(upload_dir, file.filename))
        session['filename'] = file.filename
    else:
        session['filename'] = ""
    return jsonify({'message': 'Software assistant received successfully'}), 200
        
@app.route('/software', methods=['POST'])
def software():
    session['team'] = 'software'
    _assistant.reload_chat()
    
    file = request.files['file']
    if file and file.filename != '':
        upload_dir = './uploads'  # Specify the directory where you want to save uploaded files
        os.makedirs(upload_dir, exist_ok=True)
        file.save(os.path.join(upload_dir, file.filename))
        session['filename'] = file.filename
    else:
        session['filename'] = ""
    return jsonify({'message': 'Software assistant received successfully'}), 200

if __name__ == '__main__':
    app.run()
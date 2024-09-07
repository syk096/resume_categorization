import argparse
import PyPDF2
import re
from transformers import BertTokenizer , BertModel
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import shutil
import pandas as pd


label_mapping = {
    0: 'INFORMATION-TECHNOLOGY',\
    1: 'ENGINEERING',\
    2: 'BUSINESS-DEVELOPMENT',\
    3: 'SALES',\
    4: 'HR',\
    5: 'FITNESS',\
    6: 'ARTS', \
    7: 'ADVOCATE', \
    8: 'CONSTRUCTION', \
    9: 'AVIATION',\
    10: 'FINANCE', \
    11: 'CHEF', \
    12: 'ACCOUNTANT',\
    13: 'BANKING',\
    14: 'HEALTHCARE', \
    15: 'CONSULTANT', \
    16: 'PUBLIC-RELATIONS', \
    17: 'DESIGNER', \
    18: 'TEACHER',  \
    19: 'APPAREL', \
    20: 'DIGITAL-MEDIA', \
    21: 'AGRICULTURE', \
    22: 'AUTOMOBILE', \
    23: 'BPO'
    }






device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model_root = BertModel.from_pretrained('bert-base-cased')

# Define the model architecture with dropout and L2 regularization
class TextModel(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3, l2_reg=1e-5):
        super(TextModel, self).__init__()
        self.bert = model_root
        self.intermediate_layer = nn.Linear(768, 512)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(512, num_classes)
        
        # L2 regularization added to linear layers
        self.intermediate_layer.weight.data = nn.init.kaiming_normal_(self.intermediate_layer.weight.data)
        self.intermediate_layer.bias.data.fill_(0)
        self.output_layer.weight.data = nn.init.kaiming_normal_(self.output_layer.weight.data)
        self.output_layer.bias.data.fill_(0)
        
        self.l2_reg = l2_reg
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)[1]
        intermediate = self.intermediate_layer(outputs)
        intermediate = self.dropout(intermediate)
        logits = self.output_layer(intermediate)
        return logits

def predict(ids,masks,ckpt):
    model = TextModel(num_classes=24)
    model.to(device)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    # Make predictions
    with torch.no_grad():
        outputs = model(ids, attention_mask=masks)
    prediction = torch.argmax(outputs, dim=1).tolist()
    return prediction
        
    

def text_to_tensor(text):
    # Tokenize the input text
    encoded_input = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Move tensors to the appropriate device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids , attention_mask
        


def preprocessing(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove RT and cc (common in social media)
    text = re.sub(r'\b(rt|cc)\b', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', ' ', text)
    # Remove punctuation using escape to ensure special characters are handled
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    # Remove any non-alphabetical characters (optional: if you want only English letters)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove continuous underscores (if necessary)
    text = re.sub(r'_+', ' ', text)
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Remove unnecessary words which are not important.
def remove_extra_word(text):

    extra_word=['company', 'name', 'city', 'state', 'university'] # extra words
    words = text.split()  # Split the text into words

    # Filter out the extra words
    filter_word = [word for word in words if word not in extra_word]

    filter_text = ' '.join(filter_word)

    return filter_text

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()  # Updated method
    return text

def main(args, model_path):
    
    filename = []
    category = []
    ROOT = "categorized_resume"
    model_path = model_path
    file_path = args.file_path
    pdf_files = os.listdir(file_path)
    for pdf in tqdm(pdf_files):
        pdf_path = os.path.join(file_path,pdf)
        text = extract_text_from_pdf(pdf_path)
        processed_text = preprocessing(text)
        resume = remove_extra_word(processed_text)
        ids , masks = text_to_tensor(resume)
        prediction = predict(ids=ids, masks=masks, ckpt=model_path)
        pred_class = label_mapping[prediction[0]]
        
        category_dir = os.path.join(ROOT,pred_class)
        
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        new_file_path = os.path.join(category_dir, pdf)
        shutil.move(pdf_path, new_file_path)
        
        filename.append(pdf)
        category.append(pred_class)
    categorized_resumes = pd.DataFrame({"filename":filename, "category":category})
    categorized_resumes.to_csv("categorized_resumes.csv")
    print("###########complete#############")
        
        
        
if __name__ == "__main__":
    model_path = 'model/best_model_epoch_13_0.8629032258064516.pt'
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    parser.add_argument("file_path", type=str, help="Path to the resume file")
    args = parser.parse_args()
    main(args, model_path)
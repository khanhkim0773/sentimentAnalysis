from flask import Flask, request, render_template
import tensorflow.keras.models as models
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import io
import urllib, base64
import pickle
import os
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from torch_geometric.data import Data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


gnn_model = GNN(input_dim=300, hidden_dim=64, output_dim=6)
gnn_model.load_state_dict(torch.load('my_gnn_model.pth'))
gnn_model.eval()

biLSTM_model = models.load_model('my_BiLSTM_model.h5')

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}
def replace_chat_words(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in chat_words:
            words[i] = chat_words[word.lower()]
    return ' '.join(words)

stop = stopwords.words('english')

with open('E:\\HK2_2023_2024\\NLP\\code\\my_flask_app\\tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)

maxlen = 79


path_to_model = "model/archive/GoogleNews-vectors-negative300.bin"
word2vec_model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)

def preprocess_data(text):
    words = text.split()
    
    embeddings = []
    for word in words:
        if word in word2vec_model.key_to_index:
            embeddings.append(word2vec_model[word])
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(words)])
    tfidf_weights = tfidf_matrix.toarray()[0]
    
    edge_index = []
    edge_attr = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if i < len(tfidf_weights) and j < len(tfidf_weights):
                edge_index.append([i, j])
                edge_attr.append(tfidf_weights[i] * tfidf_weights[j])  # Use TF-IDF as edge weight
    
    x = torch.tensor(embeddings, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)  # Reshape to 2D tensor
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


@app.route('/', methods=['GET', 'POST'])
def home():
    emotions = ['Buồn bã', 'Niềm vui', 'Tình yêu', 'Tức giận', 'Sợ hãi', 'Bất ngờ']
    if request.method == 'POST':
        model_name = request.form['model']
        if model_name == 'bilstm':
            data = request.form['data']
            messages = data.split('\n') 
            emotion_counts = {'Buồn bã': 0, 'Niềm vui': 0, 'Tình yêu': 0, 'Tức giận': 0, 'Sợ hãi': 0, 'Bất ngờ': 0}  # Initialize a dictionary to store the count of messages for each emotion
            for message in messages:
                message = replace_chat_words(message)
                message = re.sub(r'[^a-zA-Z\s]', '', message)
                message = ' '.join([word for word in message.split() if word not in stop])
                message = message.lower()
                message = re.sub(r'\d+', '', message)
                message = re.sub(r'\s+', ' ', message)
                message = re.sub(r'[^\w\s]', '', message)
                message = re.sub(r'http\S+', '', message)
                message_sequences = tokenizer.texts_to_sequences([message])
                message_padded = pad_sequences(message_sequences, maxlen=maxlen, padding='post')
                prediction = biLSTM_model.predict(message_padded)
                predicted_emotion = emotions[np.argmax(prediction[0])]
                emotion_counts[predicted_emotion] += 1 

            plt.clf()

            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            plt.bar(emotions, counts)
            plt.xlabel('Cảm xúc')
            plt.ylabel('Số lượng tin nhắn')
            plt.title('Số lượng tin nhắn cho mỗi cảm xúc')

            png_image = io.BytesIO()
            plt.savefig(png_image, format='png')
            png_image.seek(0)
            png_image_b64_string = "data:image/png;base64,"
            png_image_b64_string += urllib.parse.quote(base64.b64encode(png_image.read()))

            return render_template('home.html', predictions=[png_image_b64_string])
        
        elif model_name == 'gcn':
            data = request.form['data']
            messages = data.split('\n') 
            emotion_counts = {'Buồn bã': 0, 'Niềm vui': 0, 'Tình yêu': 0, 'Tức giận': 0, 'Sợ hãi': 0, 'Bất ngờ': 0} 
            for message in messages:
                message = re.sub(r'[^a-zA-Z\s]', '', message)
                message = message.lower()
                message = re.sub(r'\d+', '', message)
                message = re.sub(r'\s+', ' ', message)
                message = re.sub(r'[^\w\s]', '', message)
                message = re.sub(r'http\S+', '', message)
                processed_data = preprocess_data(message)
                if processed_data.x.shape[0] > 0 and len(processed_data.edge_index) > 1 and processed_data.edge_index.shape[1] > 0:
                    out = gnn_model(processed_data) 
                    _, predicted = torch.max(out.data, 1) 
                    predicted_emotion = emotions[predicted.item()] 
                    emotion_counts[predicted_emotion] += 1 
                else:
                    continue

            plt.clf()

            counts = list(emotion_counts.values())
            plt.bar(emotions, counts)
            plt.xlabel('Cảm xúc')
            plt.ylabel('Số lượng tin nhắn')
            plt.title('Số lượng tin nhắn cho mỗi cảm xúc')

            png_image = io.BytesIO()
            plt.savefig(png_image, format='png')
            png_image.seek(0)
            png_image_b64_string = "data:image/png;base64,"
            png_image_b64_string += urllib.parse.quote(base64.b64encode(png_image.read()))
            return render_template('home.html', predictions=[png_image_b64_string])
        
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
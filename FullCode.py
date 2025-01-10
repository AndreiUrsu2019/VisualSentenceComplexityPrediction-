import pandas as pd
import random
from nltk.corpus import wordnet
import nltk
import re
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import spearmanr
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import cuvinte_legatura
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from num2words import num2words
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import networkx as nx
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#initializare dataframes
dataframe_antrenare=pd.read_csv("/content/drive/MyDrive/Practical_ML/dataframe_antrenare_corectat.csv")
dataframe_validare=pd.read_csv("/content/drive/MyDrive/Practical_ML/dataframe_validare_corectat.csv")
dataframe_test=pd.read_csv("/content/drive/MyDrive/Practical_ML/dataframe_test_corectat.csv")

#GloVe
def preprocesare_text_GloVe(text):
    # Corectăm erorile gramaticale și eliminăm caracterele speciale
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Eliminăm caracterele speciale
    return text.lower()

# Preprocesăm textul și tokenizăm propozițiile
def preprocesare_df_GloVe(dataframe, coloana_text):
    propozitii = dataframe[coloana_text].apply(preprocesare_text_GloVe).apply(word_tokenize).tolist()
    return propozitii
dataframe=pd.concat([dataframe_antrenare,dataframe_validare])
dataframe=pd.concat([dataframe,dataframe_test])
# Pregătim datele pentru antrenare
propozitii = preprocesare_df_GloVe(dataframe_antrenare, 'text')

# Antrenăm un model Word2Vec (un precursor al GloVe) cu Gensim
model = Word2Vec(propozitii, vector_size=100, window=5, min_count=1, workers=4)

# Salvăm modelul pentru utilizare ulterioară
model.save("word2vec_model.model")

def PreprocesarePropozitie(propozitie, model_GloVe):
    nlp = spacy.load("en_core_web_sm")
    # Corectăm erorile gramaticale și eliminăm caracterele speciale
    propozitie_corectata = re.sub(r"[^a-zA-Z\s]", "", propozitie)  # Eliminăm caracterele speciale

    # Analizăm propoziția folosind spaCy
    doc = nlp(propozitie_corectata.lower())

    # Filtrăm cuvintele care sunt în modelul GloVe
    cuvinte_filtrate = [
        token.text for token in doc if token.text.isalpha() and token.text in model_GloVe.wv
    ]

    # Calculăm media vectorilor GloVe pentru propoziție
    if cuvinte_filtrate:
        vectori_cuvinte = [model_GloVe.wv[word] for word in cuvinte_filtrate]
        vector_propozitie = np.mean(vectori_cuvinte, axis=0)  # Media vectorilor
    else:
        vector_propozitie = np.zeros(model_GloVe.vector_size)  # Vector nul dacă nu sunt cuvinte în GloVe

    return vector_propozitie

def Preprocesare(dataframe, coloana_text, model_GloVe):
    # Aplicăm preprocesarea pentru fiecare propoziție
    dataframe["text_procesat"] = dataframe[coloana_text].apply(lambda x: PreprocesarePropozitie(x, model_GloVe))

    # Extragem vectorii pentru propoziții
    X = np.stack(dataframe["text_procesat"].values)

    return X

model_GloVe = Word2Vec.load("word2vec_model.model")

#Worse tf-idf

class PreprocesareWorseTfIdf:
    def __init__(self, limba='en_core_web_sm'):
        # Încărcăm modelul spaCy specific pentru limbă
        self.nlp = spacy.load(limba)

        # Listă completă de cuvinte de legătură (stopwords) din limba engleză
        self.cuvinte_legatura = set(cuvinte_legatura.words('english'))

        # Adăugăm cuvinte suplimentare care nu sunt relevante în analiză
        self.cuvinte_legatura_aditionale = {
            'said', 'one', 'would', 'could', 'should', 'new', 'many',
            'first', 'last', 'next', 'like', 'use', 'make', 'see'
        }
        # Actualizăm lista principală de stopwords
        self.cuvinte_legatura.update(self.cuvinte_legatura_aditionale)


    def curatare_text(self, text):

        # Transformare în litere mici
        text = text.lower()

        # Eliminăm URL-urile
        text = re.sub(r'http\S+', '', text)

        # Eliminăm caracterele speciale și cifrele
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Eliminăm spațiile suplimentare
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    def preprocesare_propozitie(self, propozitie):
        # Curățăm textul folosind metoda `curatare_text`
        propozitie_curatata = self.curatare_text(propozitie)

        # Procesăm textul curățat cu spaCy pentru tokenizare și analiză
        doc = self.nlp(propozitie_curatata)

        # Filtrare avansată: eliminăm stopwords și păstrăm doar anumite părți de vorbire
        tokens_procesati = [
            token.lemma_ for token in doc
            if (not token.is_stop and
                token.is_alpha and
                len(token.lemma_) > 1 and
                token.pos_ not in ['PUNCT', 'SYM', 'NUM']
            )
        ]

        # Îmbinăm lemele într-un singur string
        return " ".join(tokens_procesati)


    def extragere_caracteristici(self, dataframe, coloana_text, vectorizer=None, nr_max_caracteristici=10000):
        # Aplicăm preprocesarea textului asupra coloanei specificate din DataFrame
        dataframe['text_procesat'] = dataframe[coloana_text].apply(self.preprocesare_propozitie)
        
        if vectorizer is None:
            # Creăm un vectorizator nou dacă nu este furnizat
            vectorizer = TfidfVectorizer(
                nr_max_caracteristici=nr_max_caracteristici,
                nr_ngrame=(1, 2),  # Incluzând unigrame și bigrame
                sublinear_tf=True  # Aplicăm scalarea subliniară
            )
            # Învățăm și transformăm datele text
            X = vectorizer.fit_transform(dataframe['text_procesat']).toarray()
        else:
            # Transformăm folosind vectorizatorul existent
            X = vectorizer.transform(dataframe['text_procesat']).toarray()

        # Calculăm caracteristici suplimentare: numărul de tokeni și lungimea medie a tokenilor
        dataframe['nr_tokens'] = dataframe['text_procesat'].apply(lambda x: len(x.split()))
        dataframe['lungime_medie_tokens'] = dataframe['text_procesat'].apply(lambda x: np.mean([len(token) for token in x.split()]) if x else 0)

        # Combinăm toate caracteristicile într-o matrice
        caracteristici_complexitate = np.column_stack([
            X,
            dataframe['nr_tokens'].values,
            dataframe['lungime_medie_tokens'].values
        ])

        return caracteristici_complexitate, vectorizer


    def vizualizare_complexitate(self, features):
        # Vizualizăm distribuția lungimii medii a tokenilor
        plt.figure(figsize=(10, 6))
        sns.histplot(features[:, -1], kde=True)
        plt.title('Distributia lungimii medii a tokens')
        plt.xlabel('Lungimea medie a unui token')
        plt.ylabel('frecventa')
        plt.show()


preprocesor = PreprocesareWorseTfIdf()

#Better tf-idf
def preprocesare_propozitie(propozitie, nlp, cuvinte_legatura):
    try:
        # Curățăm și normalizăm textul
        propozitie = propozitie.lower().strip()
        propozitie = re.sub(r"[^a-zA-Z\s]", "", propozitie)

        # Procesăm textul cu spaCy
        doc = nlp(propozitie)

        # Definim părțile de vorbire pe care vrem să le păstrăm
        parti_vorbire_pastrate = {"NOUN", "VERB", "ADJ", "ADV", "ADP"}

        # Filtrăm avansat și lematizăm
        cuvinte_filtrate = [
            token.lemma_
            for token in doc
            if (token.pos_ in parti_vorbire_pastrate and
                token.text.isalpha() and
                token.text not in cuvinte_legatura and
                len(token.text) > 1)  # Excludem tokenii foarte scurți
        ]

        return " ".join(cuvinte_filtrate)

    except Exception as e:
        # Afișăm erori dacă există
        print(f"Eroare la preprocesarea propozitiei: {e}")
        return ""

nlp = spacy.load("en_core_web_sm")
cuvinte_legatura = set(nltk.corpus.cuvinte_legatura.words('english'))

def preprocesare_dataframe(dataframe, coloana_text, vectorizer=None, nr_max_caracteristici=5555):
    # Aplicăm preprocesarea asupra coloanei text din DataFrame
    dataframe['text_preprocesat'] = dataframe[coloana_text].apply(
        lambda x: preprocesare_propozitie(x, nlp, cuvinte_legatura)
    )

    # Vectorizare cu parametri flexibili
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            nr_max_caracteristici=nr_max_caracteristici,
            nr_ngrame=(1, 3),  # Păstrăm unigrame, bigrame și trigrame
            sublinear_tf=True,
            min_df=3,  # Excludem termeni foarte rari
            cuvinte_legatura='english'  # Filtrăm stopwords adiționale
        )
        # Învățăm și transformăm textul
        X = vectorizer.fit_transform(dataframe["text_preprocesat"]).toarray()
    else:
        # Transformăm doar folosind vectorizatorul existent
        X = vectorizer.transform(dataframe["text_preprocesat"]).toarray()

    return X, vectorizer, dataframe


def caracteristici_extra(X, nr_componente=100):
    # Creăm un pipeline pentru transformarea caracteristicilor
    pipeline = Pipeline([
        ('svd', TruncatedSVD(nr_componente=nr_componente, random_state=42)),  # Reducere dimensională
        ('scaler', StandardScaler())  # Standardizare
    ])

    # Aplicăm pipeline-ul asupra caracteristicilor
    return pipeline.fit_transform(X)


# Funcție pentru construirea grafului de dependențe al unei propoziții
def construire_graf_dependente(propozitie):
    doc = nlp(propozitie)  # Analizăm propoziția folosind spaCy
    G = nx.DiGraph()  # Creăm un graf orientat pentru structura sintactică
    for token in doc:
        G.add_node(token.i, label=token.text, pos=token.pos_)  # Adăugăm noduri pentru fiecare token
        if token.head != token:  # Sărim peste nodul rădăcină
            G.add_edge(token.head.i, token.i)  # Adăugăm muchii între token și părintele său
    return G

# Funcție pentru încărcarea metrilor unui graf
def incarcare_metrici_graf(graf):
    # Număr de noduri
    nr_noduri = graf.number_of_nodes()

    # Adâncimea arborelui: adâncimea maximă a arborelui de parsare
    if nx.is_directed_acyclic_graph(graf):  # Verificăm dacă graful este orientat și aciclic
        adancime_arbore = max(len(nx.shortest_path(graf, source)) for source in graf.nodes())
    else:
        adancime_arbore = 0

    # Diametrul grafului: cel mai lung drum scurt între două noduri
    try:
        diametru_graf = nx.diameter(graf.to_undirected())
    except nx.NetworkXError:  # Tratăm cazul în care graful este deconectat
        diametru_graf = 0

    # Gradul mediu al nodurilor
    grad_mediu = sum(dict(graf.degree()).values()) / nr_noduri if nr_noduri > 0 else 0

    # Distanța medie: lungimea medie a drumului scurt dintre oricare două noduri
    try:
        lungime_medie = nx.average_shortest_path_length(graf.to_undirected())
    except nx.NetworkXError:
        lungime_medie = 0

    return nr_noduri, adancime_arbore, diametru_graf, grad_mediu, lungime_medie

# Funcție pentru adăugarea metricilor de complexitate pentru o propoziție
def add_complexity_metrics(row):
    graf = construire_graf_dependente(row["text"])  # Construim graful de dependențe
    metrics = incarcare_metrici_graf(graf)  # Calculăm metricele pentru graf
    return metrics  

# Funcție pentru preprocesarea propozițiilor folosind metoda Second Best Tf-Idf
def Second_Best_TfIdf_PreprocesarePropozitie(propozitie):
    # Corectăm erorile gramaticale prin eliminarea caracterelor speciale
    propozitie_corectata = re.sub(r"[^a-zA-Z\s]", "", propozitie)

    # Analizăm propoziția folosind spaCy
    p = nlp(propozitie_corectata.lower())

    # Păstrăm doar anumite părți de vorbire
    parti_vorbire = {"NOUN", "VERB", "ADJ", "ADV", "ADP", "NUM", "PRON", "INTJ", "DET"}

    # Filtrăm cuvintele bazat pe POS și eliminăm stopwords
    cuvinte_legatura = set(nltk.corpus.stopwords.words('english'))
    cuvinte_procesate = [
        str(token.lemma_)
        for token in p
        if token.pos_ in parti_vorbire and token.text.isalpha() and token.text not in cuvinte_legatura
    ]

    return " ".join(cuvinte_procesate)

# Funcție pentru preprocesarea întregului DataFrame folosind Second Best Tf-Idf
def Second_Best_TfIdf_Preprocesare(dataframe, col_text, vec=None):
    # Aplicăm preprocesarea pentru fiecare propoziție
    dataframe["text_procesat"] = dataframe[col_text].apply(Second_Best_TfIdf_PreprocesarePropozitie)

    # Dacă nu avem un vectorizator, creăm unul nou
    if vec is None:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(dataframe["text_procesat"]).toarray()
    else:
        # Folosim vectorizatorul furnizat pentru transformare
        X = vec.transform(dataframe["text_procesat"]).toarray()

    return X, vec

# Funcție pentru metoda Best Tf-Idf
cuvinte_legatura = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def Best_TfIdf_Preprocesare(text):
    text = re.sub(r'[^\w\s]', '', text)  # Eliminăm punctuația
    text = re.sub(r'\d+', lambda x: num2words(int(x.group())), text)  # Convertim numerele în text
    cuvinte = re.split('\W+', text)  # Separăm cuvintele
    cuvinte_procesate = []
    for cuv in cuvinte:
        if cuv not in cuvinte_legatura:
            cuv_procesat = lemm.lemmatize(cuv)  # Aplicăm lematizarea
            cuvinte_procesate.append(cuv_procesat)
    return ' '.join(cuvinte_procesate)

# Aplicăm preprocesarea pentru seturile de date de antrenare, validare și testare
dataframe_antrenare['text'] = dataframe_antrenare['text'].apply(Best_TfIdf_Preprocesare)
dataframe_validare['text'] = dataframe_validare['text'].apply(Best_TfIdf_Preprocesare)
dataframe_test['text'] = dataframe_test['text'].apply(Best_TfIdf_Preprocesare)

# Configurăm vectorizatorul Tf-Idf
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), max_features=6000)

# Transformăm textul în reprezentare Tf-Idf
X_antr = tfidf.fit_transform(dataframe_antrenare['text']).toarray()
X_valid = tfidf.transform(dataframe_validare['text']).toarray()
X_test = tfidf.transform(dataframe_test['text']).toarray()

# Definim variabilele țintă
y_antr = dataframe_antrenare["score"]
y_valid = dataframe_validare["score"]

#MODELE

# RandomForestRegressor

def antrenare_RandomForestRegressor(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul RandomForest
    model = RandomForestRegressor(random_state=42)
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# LinearRegression

def antrenare_LinearRegression(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul de regresie liniara
    model = LinearRegression()
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# DecisionTreeRegressor

def antrenare_DecisionTreeRegressor(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul Decision Tree
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# SVR (Support Vector Regressor)

def antrenare_SVR(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul SVR cu kernel radial
    model = SVR(kernel='rbf', gamma=0.8, C=0.9)
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# GradientBoostingRegressor

def antrenare_GradientBoostingRegressor(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=250, random_state=42)
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# XGBRegressor

def antrenare_XGBRegressor(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul XGBoost
    model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# K-Nearest Neighbors (K-NN)

def antrenare_KNN(X_antr, y_antr, X_valid, y_valid):
    # Creeaza si antreneaza modelul K-NN cu metrici specifici
    model = KNeighborsRegressor(n_neighbors=25, p=2, metric='cosine', weights='distance')
    model.fit(X_antr, y_antr)

    # Face predictii pe datele de validare
    y_valid_pred = model.predict(X_valid)

    # Calculeaza coeficientul Spearman pentru setul de validare
    spearman_corr_test = spearmanr(y_valid, y_valid_pred).correlation

    print(f"Spearman's Rank Correlation on Test Set: {spearman_corr_test:.4f}")

# Retea Neuronala simpla (Custom NN)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        # Primul strat dens
        self.fc1 = nn.Linear(input_dim, 700)
        # Activare ReLU
        self.relu = nn.ReLU()
        # Straturi dense intermediare
        self.fc2 = nn.Linear(700, 600)
        self.fc3 = nn.Linear(600, 500)
        self.fc4 = nn.Linear(500, 400)
        # Strat de iesire
        self.fc5 = nn.Linear(400, 1)

    def forward(self, x):
        # Trecerea datelor prin fiecare strat si activare
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

def antrenare_NN(X_train, y_train, X_val, y_val, model):
    # Converteste datele de intrare si iesire in tensori PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # Determina dimensiunea de intrare
    input_dim = X_train.shape[1]

    # Defineste functia de cost si optimizatorul
    criteriu = nn.MSELoss()
    optimizator = optim.Adam(model.parameters(), lr=0.001)

    # Bucla de antrenare
    epochs = 10
    for epoch in range(epochs):
        # Forward pass: calculeaza predictiile
        outputs = model(X_train)
        loss = criteriu(outputs, y_train)

        # Backward pass: actualizeaza greutatile
        optimizator.zero_grad()
        loss.backward()
        optimizator.step()

        # Validare
        with torch.no_grad():
            val_rez = model(X_val)
            val_loss = criteriu(val_rez, y_val)

        # Afiseaza pierderile dupa fiecare epoca
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Evaluare finala
    with torch.no_grad():
        predictii = model(X_val).numpy()
        spearman_corr, _ = spearmanr(y_val.numpy(), predictii.flatten())
        print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
       

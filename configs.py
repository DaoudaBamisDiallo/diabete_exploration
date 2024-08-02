#-------------------package------------------
# packages necessaires
import streamlit as st
import numpy as np
# import  joblib as joblib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
# from ipywidgets import interact

#--------evaluation de performance--------
from  sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

#---------------------------Présentation-----------------------------------------------
def presentation():
    st.title("Porjet: Application ML pour la dectection du Diabète")

    st.sidebar.markdown("Présentation du projet")
    if st.sidebar.checkbox("Problematique",False):
        st.markdown("### I: Problèmatique")
        st.markdown('''
                La dectection du diabete chez les patients est devenu un probleme récurant chez les medecins. 

                Le processus de diagnostiques très complexe qui necessite plusieurs analyses dont les résultats sont parfois trop lents , ont entrainés des conséquences trés négatives.
                - Combien de personnes sont-elles mortes par manque de traitement approprie au diabete?
                - Combien de personnes sont-elles amputées suite au retarde des résultats d'analyse?
                - Quellle solution faut-il pour faciliter la detection du diabete et sauver des vies?

                Pour palier à cet probleme , je vais vous présenter **Fast_Diabete-Detection** : est une solution efficace et rapidement conçue pour la dectection du diabete en 2 clic.

                Salut à tous et à toutes, pour ceux qui ne me connaissent pas mon nom c'est Daouda DIALLO . Je suis ingenieur et data scientiste junior.Actuellement je fais mon M1 en data science au centre de formation GOMYCODE, ou j'ai pu acquerir des competences et techniques tres solides pour le metier de la data.

                Actuelement je suis capable de traiter et analyse des données , construire de graphiques , creer et deployer des applications basées sur le ML.

                Cet matin je vais vous présenter **Fast_Diabete-Detection** : qui est une application web d'apprentissage automatique qui conçcue pour la detection du diabete chez les femmmes.

                Avant de continuer laisser moi faire une demonstration.
                ''')
        
        colp,cols,cola = st.columns([0.3,0.3,0.3])

        with colp :
            st.subheader('Difficultées')
            st.image("datasets_bd/images/problematique.jpeg")

        with cols :
            st.subheader('Complication')
            st.image("datasets_bd/images/soufrant.jpg")
            
        with cola :
            st.subheader('Amputation')
            st.image("datasets_bd/images/amputation.webp")
    if st.sidebar.checkbox("Description du Jeu de données",False):
        description_db='''
            ### II: Description de la base de données:
            Pour réaliser cet projet j'ai utilisé la base de données diabete de Kaggle.
            Cet ensemble de données provient  de l'Institut national du diabète et des maladies digestives et rénales. 
            Dont tous les patients ici sont des femmes d'origine indienne Pima agées de 21 ans au moins.
            Caracteristiques des patiences
            - Grossesses : Nombre de grossessse
            - Glucose : taux du glucose dans le sang
            - Tension artérielle : tension artérielle diastolique (mm Hg)
            - Épaisseur de la peau : Épaisseur du pli cutané du triceps (mm)
            - Insuline : Taux d'insuline 
            - IMC : Indice de masse corporelle (poids en kg/(taille en m)^2)
            - DiabetesPedigreeFunction : Fonction généalogie du diabète
            - Âge : Âge (années)
            - Résultat : variable de classe (0 ou 1)

            **L'objectif de construire un modele de machine learning pour prédire, à partir de mesures diagnostiques, si un patient est diabétique ou non.**
            '''
        st.markdown(description_db)
    if st.sidebar.checkbox("Sommaire",False):
        sommaire='''
                ### III: Plan de travail
                #### 1- Importation des librairies
                #### 2-Chargement de la base de données
                #### 3-Analyse exploratoire des données
                - Affichage des données
                - Analyse univariée
                - Bivariée
                - Analyse multivariée
                #### 4- Modélisation
                - LogisticRegression
                - Arbre de decision
                - RandomForest classifier
                - KNN
                - SVM Classifier
                '''
        st.markdown(sommaire)
        #url ="https://ettienyann-diabete-diabete-tr9slg.streamlit.app/"
        # if st.button("[Aller au laboratoire](%s)" % url,type="primary"):
        #     modeling()

#------------------------chargement et affichage de la base de données------------------------------

#fonction de chagement de la base de données
@st.cache_data(persist=True)
def loading_dataset ():
    file="datasets_bd/db/diabetes.csv"
    data = pd.read_csv(file)
    data["Outcome"]=data["Outcome"].apply(lambda x: "Diabetique" if x==1 else("Non Diabetique" if x==0 else x))
    return data

#affichage des 100 premiere observation
def showing_data(data):  
    # affichage des  100 premiere observation
    st.sidebar.markdown("Analyse Exploratoire des données")
    df_sample = data.sample(100)
    if st.sidebar.checkbox("Afficher les données brut",False):
        st.subheader("Jeu de données de diabete : Echantillons de 100 observations")
        st.write(df_sample)


# ------------------------------analyse exploratoire des données-----------------------
def AED(data):
    
        # Analyse Univariée
    cols=data.columns.tolist()
    cols1=data.drop('Outcome',axis=1).columns.tolist()
    #-------fonction d'analyse univariée-----------
    def hist_plot(var):
        fig,ax=plt.subplots(figsize=(10,5))
        ax = sns.histplot(x=data[var], kde=True).set_title("Histogramme de "+str(var))
        st.pyplot(fig)

    #---------fonction d'analyse bivariée-----------

    def cat_plot(a):
        fig,ax=plt.subplots()
        ax = sns.boxplot(y=data[a],x=data['Outcome'])
        st.pyplot(fig)

    #---------fonction d'analyse multivariée-----------

    def rel_plot(a,b,c):

        if c=="scatter":
            fig,ax=plt.subplots()
            ax = sns.scatterplot(x=a, y=b, hue='Outcome',data=data)
            st.pyplot(fig)

        if c=="regression line":
            
            fig,ax=plt.subplots()
            ax = sns.regplot(x=a, y=b, data=data,color="orange",line_kws=dict(color="b"))
            st.pyplot(fig)


    #-----------modelisation avec svm---------------
    # chargement des données
    train_features=pd.read_csv("datasets_bd/db/train_end.csv")
    x_test=pd.read_csv("datasets_bd/db/x_test_end.csv")
    train_labels=pd.read_csv("datasets_bd/db/y_train_end.csv")
    y_test=pd.read_csv("datasets_bd/db/y_test_end.csv")
    vars_imp=pd.read_csv("datasets_bd/db/imp_vars_end.csv")
    vars_imp1=pd.read_csv("datasets_bd/db/imp_vars_end.npy")
    def model_svc():
        # st.header("Modèle Suport Vecteur Machine Classificateur (SVC)")
        # definition de la classe
        svmc = SVC(random_state=95)
        # dictionnaire des hyperparametres
        grid_params ={"C":[0.001,0.0001,0.00001],"kernel" : ["linear", "poly", "rbf", "sigmoid"]}
        # recherche des hypermarametres
        grid_class = GridSearchCV(estimator=svmc, 
                                param_grid=grid_params,
                                scoring="f1",cv=5)
        
        # enrainement du modele
        svmc_grid_class = grid_class.fit(train_features,train_labels)
        return svmc_grid_class

    # modelisation
    def svm_modeling(model):
        st.header("Modèle Suport Vecteur Machine Classificateur (SVC)")
        # # affichage des hyperparametres
        st.header("Les Hyperparamètres ")
        st.subheader("Parametres optimaux: "+str(model.best_estimator_))
        st.subheader("Score : "+str(round(model.best_score_,2)))
        
    # fonction d'evaluation de modele
    def evaluation_models(model):
        # prediction sur les données entrainement
        st.header("Evaluation de performances")
        train_pred = model.predict(train_features)
        col1,col2 = st.columns([0.3,0.7])
        with col1:
            st.write("Performances sur les données d'entrainemment")
            accuracy = accuracy_score(train_labels,train_pred)
            recall = recall_score(train_labels,train_pred)
            precision = precision_score(train_labels,train_pred)

            st.subheader("Accuracy : "+str(round(accuracy,2)))

            st.subheader("Recall : "+str(round(recall,2)))
            st.subheader("Precision : "+str(round(precision,2)))

        # affichage de la matrice de confusion train
        with col2:
        
            cm = confusion_matrix(train_labels,train_pred)
            # Création de l'objet ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non Diabetique","Diabetique"])
            # Affichage graphique interactif de la matrice de confusion avec ConfusionMatrixDisplay
            fig, ax = plt.subplots(figsize=(4, 3))
            disp.plot(cmap=plt.cm.Blues,ax=ax)
            # Affichage de la figure dans Streamlit
            st.pyplot(fig)

            # prediction sur les données de test
        train_pred = model.predict(x_test) 

        col3,col4 = st.columns([0.3,0.7])
        with col3:
            st.write("Performances sur les données de Test")
            accuracy = accuracy_score(y_test,train_pred)
            recall = recall_score(y_test,train_pred)
            precision = precision_score(y_test,train_pred)

            st.subheader("Accuracy : "+str(round(accuracy,2)))

            st.subheader("Recall : "+str(round(recall,2)))
            st.subheader("Precision : "+str(round(precision,2)))

        # affichage de la matrice de confusion train
        with col4:
        
            cm = confusion_matrix(y_test,train_pred)
            # Création de l'objet ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non Diabetique","Diabetique"])
            # Affichage graphique interactif de la matrice de confusion avec ConfusionMatrixDisplay
            fig, ax = plt.subplots(figsize=(4, 3))
            disp.plot(ax=ax)
            # Affichage de la figure dans Streamlit
            st.pyplot(fig)
            # prediction sur les données de test
        

        
    # menu
        
    
    if st.sidebar.checkbox("Analyse Univariée",False):
        st.title("Distribution des variables")
        # Analyse Univariée
        col = st.selectbox("Choisir la variables a visualiser",cols,key="univare")
        hist_plot(col)
        
    
    if st.sidebar.checkbox("Analyse Bivariée",False):
        st.title("Distrution des variables en fonction de la variable cible")
        # Analyse biivariée
        col =st.selectbox("Choisir la variables a visualiser",cols,key="bivare")
        cat_plot(col)
        
    
    if st.sidebar.checkbox("Analyse Multivariée",False):

        st.title("Reletion entre les variables")
        col1,col2,col3 = st.columns(3)
        with col1:
            x =st.selectbox("Variable en abscisse",cols1,key="x")
        with col2:
            y =st.selectbox("Variables en ordonnée",cols1,key="y")
        with col3:
            c =st.selectbox("Graphique",["scatter","regression line"],key="c")
        # Analyse multiiivariée
        rel_plot(x,y,c)
    # importance des variables
    if st.sidebar.checkbox("Importance des variables",False):
        st.title("Imporance des variables")
  
        st.write(vars_imp.T)
  
    st.sidebar.write("Modélisation avec SVM")
    if st.sidebar.checkbox("Entrainement du modele",False):
        svm_modeling(model_svc())
        if st.sidebar.radio("Evaluer le modele",["Patienter","Evaluer"])=="Evaluer":
            evaluation_models(model_svc())

# -------------modelisation et deployement----------------------------
def deployement():

        #chagement du modele
        def load_model():
            # data = joblib.load("datasets_bd/db/model_diabete.joblib")
            # return data
            with open("datasets_bd/db/model_diabete.pkl","rb") as file:
                 data = pkl.load(file)
            file.close()
            return data

        model_diabete = load_model()
        # fonction d'inference
        def inference(Glucose,BMI,Age,DiabetesPedigreeFunction,BloodPressure,Pregnancies):
            
            df = np.array([Glucose,BMI,Age,DiabetesPedigreeFunction,BloodPressure,Pregnancies])
           
            diabetique = model_diabete.predict(df.reshape(1,-1))
            return diabetique
        # saisie des iinformations du patience

        st.header("Informations de la patiente")
        col1,col2 = st.columns(2)

        with col1 : 
            Glucose = st.number_input(label="Taus du glucose",min_value=0.0,max_value=1.0,value=0.621212)
            Age = st.number_input(label="L'age ",min_value=0.0,max_value=1.0,value=0.166667)
            BloodPressure = st.number_input(label="Tension arterielle diastolique",min_value=0.0,max_value=1.0, value=0.631579)

        with col2 :
            BMI = st.number_input(label="Indice du masse corporelle",min_value=0.0,max_value=1.0,value=0.570681)
            DiabetesPedigreeFunction = st.number_input(label="Fonction généalogie du diabète",min_value=0.0,max_value=1.0,value=0.137939)
            Pregnancies = st.number_input(label="Nombre de grossse",min_value=0.0,max_value=1.0,value=0.205882)

        # with col1 : 
        #     Glucose = st.number_input(label="Quantité du glucose",min_value=0 ,value=117)
        #     Age = st.number_input(label="L'age ",min_value=21, value=29)
        #     BloodPressure = st.number_input(label="Tension arterielle diastolique",min_value=0, value=72)
           
        # with col2 :
        #     BMI = st.number_input(label="Indice du masse corporelle",min_value=0.0, value=32.0,)
        #     DiabetesPedigreeFunction = st.number_input(label="Fonction généalogie du diabète",min_value=0.0, value=0.37)
        #     Pregnancies = st.number_input(label="Nombre de grossse",min_value=0, value=3)
    
            # axamianation du patient
        if st.button('Examiner le patient') :
            resultat= inference(Glucose,BMI,Age,DiabetesPedigreeFunction,BloodPressure,Pregnancies)
            if resultat[0] == 1:
                st.warning("Diabetique")
            elif resultat[0] == 0:
                st.success("Non diabetique")
            else :
                st.error("Le résultat de l'examen inconnu merci de bien saisir les information ou de consulter le médecin pour plus de detail")
def appli():
    st.sidebar.markdown("Utilisation de l'application")
    if st.sidebar.checkbox("Application",False):
        # description de l'application
        # st.subheader("*"*80)
        st.title("Welcome to Fast Diabete Detection")
        st.image("datasets_bd/images/presnation.webp")
        
        st.header("Réalisée par : IA-Data_Consulting")
        st.markdown(('''FDD est une application web de machine learning conçcue pour la détection du diabete chez les femmes avec une fiabilité de 72% '''))
        
        if st.sidebar.radio("Visiter le patience",["Aprés","Maintenant"])=="Maintenant":
            deployement()
        
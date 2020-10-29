import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import altair as alt
from vega_datasets import data
import plotly.graph_objects as go
from matplotlib.pyplot import figure
from plotly.subplots import make_subplots
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

Cura_19 = pd.read_csv('POMC_Cleaned_Update920.csv')
pd.set_option('display.max_columns', None)
Cura_20 = pd.read_csv('2020_POMC_Report_Cleaned.csv')
Cura_22 = pd.read_csv('2020_POMC_Report_Cleaned.csv')
df2 = pd.read_csv('POMC_Cleaned_Update920.csv')

test = Cura_19.loc[Cura_19['Sex'].isin(['0'])]
test_2 = Cura_19.loc[Cura_19['Sex'].isin(['1'])]
gender = test.Month_num.value_counts().sort_index()
gender_f = pd.DataFrame(gender)
gender_f.rename(columns={'Month_num' :'Female'}, inplace=True)
gender_f['Month'] = gender_f.index
gender_f.Month.replace([1,2,3,4,5,6,7,8,9,10,11], ['January','February','March','April','May','June','July','August',
                                                  'September','October','November'], inplace=True)
gender_m = test_2.Month_num.value_counts().sort_index()
gender_m = pd.DataFrame(gender_m)
gender_m.rename(columns={'Month_num' :'Male'}, inplace=True)
gender_m['Month'] = gender_m.index
gender_m.Month.replace([1,2,3,4,5,6,7,8,9,10,11], ['January','February','March','April','May','June','July','August',
                                                  'September','October','November'], inplace=True)
Mon_19 = Cura_19.Month_num.value_counts()
Mon_19.sort_index(inplace=True)
Mon_19.drop(Mon_19.index[[-1,-2,-3]], inplace =True)
Mon_19 = pd.DataFrame(Mon_19)
Mon_20 = Cura_22.Month_num.value_counts()
Mon_20.sort_index(inplace=True)
Mon_20 = pd.DataFrame(Mon_20)
Mon_Lab = Mon_20
Mon_Lab['Month'] = Mon_20.index

df_test = Cura_19[['Acyclo Cream', 'Acyclovir',
       'Advised', 'Albendazole', 'Amitriptyline', 'Amlodipine',
       'Amoxicillin', 'Amoxikid', 'Ampiclox', 'Ampilox', 'Anusol Gel',
       'Aprinox', 'Artesunate', 'Ascorbic Acid', 'Ascoril',
       'Ascoril Syrup', 'Aspirin', 'Atenolol', 'Azithromycin', 'Bed Rest',
       'Bendro', 'Bisacodyl', 'BP Monitoring', 'Caf Ear Drops',
       'Caf Eye Drops', 'Calamine Lotion', 'Calcium', 'Calcium Lactate',
       'Calcium Lactatea', 'Candiderm Cream', 'Captopril',
       'Carbamazepine', 'Cardiac Asa', 'Catheter Removal', 'CBC',
       'Cefixime', 'Cefiximkid', 'Cetamol Syrup', 'Cetirizine',
       'Chlorhexidine Mouth Wash', 'Chlorphenamine',
       'Chlorphenical Ear Drops', 'Chlorpheniramine', 'Cimetidine',
       'Ciprofloxacin', 'Clotri Pessaries', 'Clotrimazole Cream',
       'Coartem', 'Cocs', 'Cold Compress', 'Cotimoxkid', 'Cotrimoxazole',
       'Cough Linctus', 'Counseling', 'CTX', 'Dexamethasone', 'Diazepam',
       'Diclofenac Gel', 'Doxycycline', 'Dressing', 'Duocotexin',
       'Ear Syringing', 'EBF', 'Erythromycin', 'F/A', 'Fansidar', 'Fefo',
       'Ferrous Sulphate', 'Flucamoxkid', 'Fluconazole', 'Folic Acid',
       'Furosemide', 'Glibenclamide', 'Glycerin Borax', 'Gripe Water',
       'Griseofulvin', 'HCG', 'HCT', 'Hydrocortisone Cream', 'Ibuprofen',
       'Ibuprokid', 'Indocid', 'Lasortern', 'Levofloxacin', 'Losarton',
       'Magnesium', 'Magnesium Trisilicate', 'Matronidazol',
       'Mebendazole', 'Medical Check Up', 'Medicated Soap',
       'Mefenamic Acid', 'Mentronidazole', 'Metformin', 'Metronidazole',
       'Microgynon', 'MultiVits', 'Nalidixic Acid', 'Nifedipine',
       'No Meds', 'Nyst Pessaries', 'Nystakid',
       'Nystatin Oral Suspension', 'ORS', 'Omeprazole', 'Paracetamol',
       'Permethrin Cream', 'Phenytoin', 'Piritex', 'Piriton',
       'Prednisolone', 'Probeta-N', 'Propranolol', 'Quinine',
       'Reassurance', 'Rectal Diazepam', 'Referred', 'Results Given',
       'Reythromycin', 'Salbutamol Inhaler', 'Silver Sulfadiazine',
       'Synclav', 'Tapid Sponging', 'Teo', 'Tested And Results Given',
       'Tetracycline Eye Ointment', 'Vendor', 'Vitamin A', 'Vitamin B6',
       'Vitamin Bx', 'Vitamin C', 'Zinc Sulfate', 'Zinkid', 'Zycel']]

Val = df_test.sum(axis=0).sort_values(ascending=False)
Top_Drugs = Val.head(15)
top_drugs = pd.DataFrame(Top_Drugs)
top_drugs.rename(columns={0 :'Values'}, inplace=True)
top_drugs.index.names = ['Drug']
top_drugs['Drug'] = top_drugs.index
top_drugs = top_drugs[['Drug', 'Values']]
top_drugs = top_drugs.reset_index(drop=True)

drugs_20 = Cura_22.drugName.value_counts()
drugs_20 = pd.DataFrame(drugs_20)
drugs_20 = drugs_20.head(15)
drugs_20.rename(columns={'drugName' :'Values'}, inplace=True)
drugs_20['Drug'] = drugs_20.index
drugs_20 = drugs_20[['Drug', 'Values']]
drugs_20 = drugs_20.reset_index(drop=True)

up_df = df2.sort_values(['Month_num','Age','Diagnosis'])
Diagnosis = up_df[['Month_num','Age','Sex','Diagnosis','Treatment']]
Mon_Diag = df2[['Month_num','Diagnosis']]
Mon_nosis = Mon_Diag.loc[Diagnosis['Month_num'].isin(['1','2','3','4','5','6','7','8','9','10','11'])]
text_0 = ' '.join(str(v) for v in Mon_nosis.Diagnosis)

from PIL import Image
image = Image.open('CURA.jpeg')

st.title("PO Cura Medical Center")
st.image(image, caption='We believe that every life matters and our passion is to see those we work with follow their heart, pursue their dreams, receive the love of the Father, and live life with purpose!',
          use_column_width=True)


st. write("""
          ## Explore The Data
          """)
          
dataset_name = st.sidebar.selectbox("Select Dataset", ("Cura Medical Center 2019",
                                                       'Cura Medical Center 2020'))

@st.cache(persist=True)
def EDA_19(dataset):
    Cura_19 = pd.read_csv('POMC_Cleaned_Update920.csv')
    return st.dataframe(Cura_19)
@st.cache(persist=True)
def EDA_20(dataset):
    Cura_20 = pd.read_csv('2020_POMC_Report_Cleaned.csv')
    return st.dataframe(Cura_20)

def get_dataset(dataset_name):
    if dataset_name == "Cura Medical Center 2019":
        if st.checkbox("Preview Dataset"):
            data = Cura_19
            st.dataframe(data.sample(frac=.005,replace=True,random_state=1))
        elif st.checkbox("Head"):
            data = Cura_19
            st.dataframe(data.head())
        if st.checkbox("Tail"):
            data = Cura_19
            st.dataframe(data.tail())     
        if st.checkbox("Show All Dataset"):
            data = Cura_19
            st.dataframe(data)
    else:
        if st.checkbox("Preview Dataset"):
            data = Cura_20
            st.dataframe(data.sample(frac=.005,replace=True,random_state=1))
        elif st.checkbox("Head"):
            data = Cura_20
            st.dataframe(data.head())
        if st.checkbox("Tail"):
            data = Cura_20
            st.dataframe(data.tail())     
        if st.checkbox("Show All Dataset"):
            data = Cura_20
            st.dataframe(data)

get_dataset(dataset_name)

if dataset_name == "Cura Medical Center 2019":
    st. write("""
                  ## Patients Seen YTD
                  """)
    if st.checkbox("By Age"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        ax = sns.countplot(x="Age_Range", data=Cura_19, order = Cura_19['Age_Range'].value_counts().index)
        plt.title(('Patients Seen in 2019, by Age'), fontsize=14)
        ax.grid(False)
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.03, p.get_height() * 1.006))
        plt.rc_context({'axes.edgecolor':'black'})
        plt.xlabel('')
        plt.ylabel('Age Range')
        sns.despine();
        st.pyplot(fig)
        st. write("""
                  #### 56% of patients seen were the age of 18 or younger.
                  """)
    if st.checkbox("By Age and Sex"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        ax = sns.countplot(x="Age_Range", hue='Sex', data=Cura_19, order = Cura_19['Age_Range'].value_counts().index)
        plt.title(('Patients Seen in 2019, by Age'), fontsize=14)
        ax.grid(False)
        plt.rc_context({'axes.edgecolor':'black'})
        plt.xlabel('')
        plt.ylabel('Age Range')
        ax.legend(['Female','Male'])
        sns.despine();
        st.pyplot(fig)
        st. write("""
                  #### After the age of 18, a female is 3x more likely than a male to visit the Medical Center.
                  """)
    if st.checkbox("By Month"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        ax = sns.countplot(y="Month_name", data=Cura_19, order = Cura_19['Month_name'].value_counts().index)
        plt.title(('Patients Seen Each Month in 2019'), fontsize=14)
        ax.grid(False)
        plt.rc_context({'axes.edgecolor':'black'})
        plt.xlabel('')
        plt.ylabel('')
        sns.despine();
        st.pyplot(fig)
        st. write("""
                  ### July and May were the most active months for the medical center.
                      * July is the coldest month in Uganda, with an average low-temperature of 62.8°F(17.1°C).
                      * May is the last month of spring and wet season. It is also the most humid month, which can lead to an increase in mosquitoes.
                      * Mosquitoes love moisture and humidity and can seem even more aggressive after a storm. - ORTHO WEBSITE
                  """)
    if st.checkbox("By Month and Sex"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        y = np.array(gender_f['Month'])
        x = np.array(gender_f['Female'])
        z = np.array(gender_m['Male'])
        plt.plot(y, x, "pink", label="Female", marker="o")
        plt.plot(y, z, "-b", label="Male", marker="o")
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10], ['January','February','March','April','May','June','July','August',
                                                          'September','October','November'], rotation=75)
        sns.despine();
        plt.legend(loc="upper left")
        plt.title(('Patients Seen Each Month, by Sex'), fontsize=12)
        plt.ylabel('Number of Patients')
        st.pyplot(fig)
    if st.checkbox("By Month, 2020 Comparison"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        y = np.array(Mon_Lab['Month'])
        x = np.array(Mon_19['Month_num'])
        z = np.array(Mon_Lab['Month_num'])
        plt.plot(y, x, "blue", label="2019", marker="o")
        plt.plot(y, z, "red", label="2020", marker="o")
        plt.xticks([1,2,3,4,5,6,7,8], ['January','February','March','April','May','June','July','August'], rotation=75)
        plt.grid(False)
        sns.despine();
        plt.legend(loc="upper left")
        plt.title(('Patients Seen Each Month'), fontsize=12)
        plt.ylabel('Number of Patients')
        st.pyplot(fig)
        
    
if dataset_name == "Cura Medical Center 2020":
    st. write("""
                  ## Patients Seen YTD
                  """)
    if st.checkbox("By Month"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        ax = sns.countplot(y="Month_name", data=Cura_22, order = Cura_22['Month_name'].value_counts().index)
        plt.title(('Patients Seen Each Month in 2020'), fontsize=14)
        ax.grid(False)
        plt.rc_context({'axes.edgecolor':'black'})
        plt.xlabel('')
        plt.ylabel('')
        sns.despine();
        st.pyplot(fig)
    if st.checkbox("By Month, 2019 Comparison"):
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        y = np.array(Mon_Lab['Month'])
        x = np.array(Mon_19['Month_num'])
        z = np.array(Mon_Lab['Month_num'])
        plt.plot(y, x, "blue", label="2019", marker="o")
        plt.plot(y, z, "red", label="2020", marker="o")
        plt.xticks([1,2,3,4,5,6,7,8], ['January','February','March','April','May','June','July','August'], rotation=75)
        plt.grid(False)
        sns.despine();
        plt.legend(loc="upper left")
        plt.title(('Patients Seen Each Month'), fontsize=12)
        plt.ylabel('Number of Patients')
        st.pyplot(fig)

if dataset_name == "Cura Medical Center 2019":
    st. write("""
                  ## Drugs Prescribed
                  """)
    if st.checkbox("Most Distributed"):
        data = top_drugs
        st.dataframe(data)
    if st.checkbox("Most Distributed Chart"):
        plt.rcParams['figure.dpi'] = 360
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        chart = sns.barplot(data=top_drugs, x='Values', y="Drug", ax=ax)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(('Top Drugs Prescribed in 2019'), fontsize=16)
        sns.despine();
        st.pyplot(fig)
    if st.checkbox("Drug Descriptions"):
        st. write("""
                  ### Top 15 Medications Distributed
                  * *Paracetamol* - aka Tylenol, it is used to treat pain and fevers
                  * *Coartem* - treats the symptoms of Malaria
                  * *Chlorpheniramine* - relieves symptoms of allergy, hay fever, and the common cold
                  * *Amoxicillin* - treats a wide variety of bacterial infections and stomach ulcers
                  * *Metronidazole* - treats various infections, including certain types of vaginal infections.
                  * *Ibuprofen* - treats fever and pain
                  * *Omeprazole* - treats certain stomach and esophagus problems (acid reflux & ulcers)
                  * *Cotrimoxazole* - treats or prevent infections
                  * *Nifedipine* - used to treat high blood pressure and to control chest pain
                  * *Amoxikid* - Amoxicillin for children
                  * *Albendazole* - anti-worm medication
                  * *Magnesium* - supplements
                  * *Bendroflumethiazide* - used to treat high blood pressure and the build-up of fluid in your body
                  * *Cetirizine* - relieves allergy symptoms such as watery eyes, runny nose, itching eyes/nose, sneezing, hives, and itching
                  * *Doxycycline* - treats and prevent infections and malaria
                  """)
if dataset_name == "Cura Medical Center 2020":
    st. write("""
                  ## Drugs Prescribed
                  """)
    if st.checkbox("Most Distributed"):
        data = drugs_20
        st.dataframe(data)
    if st.checkbox("Most Distributed Chart"):
        plt.rcParams['figure.dpi'] = 360
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        chart = sns.barplot(data=drugs_20, x='Values', y="Drug", ax=ax)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(('Top Drugs Perscribed in 2020'), fontsize=16)
        sns.despine();
        st.pyplot(fig)
    if st.checkbox("Drug Descriptions"):
        st. write("""
                  ### Top 15 Medications Distributed
                  * *Artemether/Lumefantrine* - treats the symptoms of Malaria
                  * *Amoxicillin* - treats a wide variety of bacterial infections and stomach ulcers
                  * *Nifedipine* - used to treat high blood pressure and to control chest pain
                  * *Paracetamol* - aka Tylenol, it is used to treat pain and fevers
                  * *Ampiclox* - treats urinary tract and respiratory tract infections, meningitis, gonorrhea and infections of the stomach or intestine
                  * *Fefo* - a prescription iron supplement indicated for use in improving the nutritional status of iron deficiency
                  * *Omeprazole* - treats certain stomach and esophagus problems (acid reflux & ulcers)
                  * *Ibuprofen* - treats fever and pain
                  * *Ciprofloxacin* - treats a variety of bacterial infections
                  * *Cefixime* - treats bacterial infections such as bronchitis, gonorrhea, urinary tract, etc. 
                  * *Metronidazole* - treats various infections, including certain types of vaginal infections
                  * *Clotrimazole Pessaries* - reat yeast infections including thrush in women and men, although thrush is more common in women
                  * *Griseofulvin* - treats fungal infections such as ringworm, "jock itch," and athlete's foot
                  * *Erythromycin* - treat certain infections caused by bacteria, such as infections of the respiratory tract, including bronchitis, pneumonia,etc.
                  * *Chlorpheniramine* - relieves symptoms of allergy, hay fever, and the common cold
                  """)
    if st.checkbox("Who prescibed the most drugs?"):
        labels = 'Dr. Jimmy', 'Dr. Victoria', 'Sarah', 'Nakabugo', 'Claire','Joanne'
        sizes = [355, 259, 207, 149, 148, 73]
        explode = (0, 0, 0, 0, 0, 0)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=75, textprops={'fontsize': 7})
        ax1.axis('equal')
        plt.title(('Drugs Prescribed by'), fontsize=12)
        st.pyplot(fig1)
        st. write("""
                  ### Dr. Jimmy and Dr. Victoria prescribed the most medication.
                  """)
        
if dataset_name == "Cura Medical Center 2019":
    st. write("""
                  ## Amount of Drugs Received Per Visit
                  """)
    if st.checkbox("Show Data"):
        meds = Cura_19.Meds.value_counts()
        meds = pd.DataFrame(meds)
        meds = meds.reset_index()
        meds.rename(columns={'index' :'Drugs Received'}, inplace=True)
        data = meds
        st.dataframe(data)
    if st.checkbox("Show Pie Chart"):
        labels = 'One', 'Two', 'Three', 'Four', 'Five+'
        sizes = [286, 920, 1015, 459, 132]
        explode = (0, 0, 0, 0, 0)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=75, textprops={'fontsize': 7})
        ax1.axis('equal')
        st.pyplot(fig1)
        st. write("""
                  ### Per vist, a patient had a 57% chance of being prescribed three or more drug prescriptions.
                  """)

if dataset_name == "Cura Medical Center 2020":
    st. write("""
                  ## What is the cost of medication per patient, by month?
                  """)
    if st.checkbox("US Dollar"):
        us_price = Cura_22[['Month_num','US_PPP']]
        us_price = us_price.dropna()
        us_price = us_price.groupby(['Month_num']).sum()
        us_price['Month'] = us_price.index
        us_price = us_price.reset_index(drop=True)
        us_price = us_price[['Month', 'US_PPP']]
        us_price.Month.replace([1,2,3,4,5,6,7,8], ['January','February','March','April','May','June','July','August',
                                                  ], inplace=True)
        data = us_price
        st.dataframe(data)
    if st.checkbox("Ugandan Shilling"):
        ug_price = Cura_22[['Month_num','UG_PPP']]
        ug_price = ug_price.dropna()
        ug_price = ug_price.groupby(['Month_num']).sum()
        ug_price['Month'] = ug_price.index
        ug_price = ug_price.reset_index(drop=True)
        ug_price = ug_price[['Month', 'UG_PPP']]
        ug_price.Month.replace([1,2,3,4,5,6,7,8], ['January','February','March','April','May','June','July','August',
                                                  ], inplace=True)
        ug_price = ug_price.round(decimals=2)
        data = ug_price
        st.dataframe(data)
    if st.checkbox("Medication Cost Per Patient, by Month"):
        us_price = Cura_22[['Month_num','US_PPP']]
        us_price = us_price.dropna()
        us_price = us_price.groupby(['Month_num']).sum()
        us_price['Month'] = us_price.index
        us_price = us_price.reset_index(drop=True)
        us_price = us_price[['Month', 'US_PPP']]
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        x = np.array(us_price['US_PPP'])
        y = np.array(us_price['Month'])
        ax.grid(False)
        plt.plot(y, x, marker='o')
        plt.xticks(rotation=90)
        plt.rcParams['figure.dpi'] = 360
        plt.figsize=(2,1)
        plt.yticks(np.arange(min(x-120), max(x)+100, 150))
        plt.xticks([1,2,3,4,5,6,7,8], ['January','February','March','April','May','June','July','August'], rotation=75)
        plt.yticks([100,250,400,550,700,850], ['$100','$250','$400','$550','$700','$850'])
        plt.title('Price Per Patient in 2020')
        sns.despine();
        st.pyplot(fig)
    if st.checkbox("Avg. Medication Cost Per Patient, by Month"):
        month = Cura_22[['Month_num','US_PPP']]
        month = month.groupby(['Month_num']).mean()
        month['Month'] = month.index
        month = month.reset_index(drop=True)
        plt.rcParams['figure.dpi'] = 360
        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        chart = sns.barplot(data=month, x='Month', y="US_PPP", ax=ax)
        for p in ax.patches:
            ax.annotate('$'+str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.006))
        plt.xticks([0,1,2,3,4,5,6,7], ['January','February','March','April','May','June','July','August'], rotation=75)
        plt.yticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0], ['$0','$0.50','$1.00','$1.50','$2.00','$2.50','$3.00'])
        plt.xlabel('')
        plt.ylabel('')
        plt.title(('Average Price per Patient, by Month in 2020'), fontsize=16)
        sns.despine();
        st.pyplot(fig)
        st. write("""
                  May was the most expensive month for avg. price per patient, despite it being the second to last month in terms of patients seen. 
                  """)
    st. write("""
                  # CONCLUSION
    * Due to COVID-19, the Medical Center saw a third of the amount of patients compared to 2019 (from April to August).
    * Cura Medical Center has seen approximately 1,191 patients and provided over 60,000 (quantity supplied) of medication.
    * The average medication cost per patient in 2020 was $2.55.
    * The total medication cost for 2020 was $3,034.58 (from January to August).
    * The average cost of medication was $1.67.
    * The most expensive medication cost was $84.15.
    
                  """)


if dataset_name == "Cura Medical Center 2019":
    st. write("""
                  ## Most Common Patient Diagnoses
                  """)
    if st.checkbox("Diagnoses Wordcloud"):
        most = text_0
        wordcloud = WordCloud(background_color="white", width=600, mode = 'RGBA', max_words = 150, height=350).generate(most)
        fig_dims = (30, 15)
        fig, ax = plt.subplots(figsize=fig_dims)
        plt.imshow(wordcloud,interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
        st. write("""
                  ## What is a WordCloud?
                  """)
        st. write("""
                  ### A WordCloud is a technique used for visualizating text within a dataset. The bigger or more frequent the word appears, the more important it is.
        * Malaria - In 2018, there were 405,000 deaths from Malaria
        * Bronchitis 
        * Candidiasis - Yeast infection
        * RTI - Respiratory tract infection
        * URTI - Upper respiratory tract infection
        * Coryza - also known as Rhinitis
                  """)
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
        
                  
                  
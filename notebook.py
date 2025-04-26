#!/usr/bin/env python
# coding: utf-8

# # Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan di Jaya Jaya Institut

# - Nama:Adindra Vickar Ega
# - Email: adindravickar@gmail.com
# - Id Dicoding: mahega_0107

# ## Persiapan

# ### Menyiapkan library yang dibutuhkan

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
from sqlalchemy import create_engine

file_path = "data.csv"

# Load the dataset with delimiter
df = pd.read_csv(file_path, delimiter=';')
df


# ### Menyiapkan data yang akan diguankan

# ## Data Understanding

# In[2]:


# Show basic info
df.info()


# In[3]:


df.describe()


# In[4]:


df.columns


# In[5]:


df.rename(columns={"Nacionality": "Nationality"}, inplace=True)

# Verify the change
print(df.columns)


# In[6]:


#Mengecek jumlah missing value
df.isna().sum()


# ## EDA

# In[7]:


print(df.columns.tolist())


# In[8]:


# Enhanced course cleaning function
def clean_course(x):
    """
    Categorizes courses into broader academic domains.
    
    Parameters:
    x (int): The original course code
    
    Returns:
    str: The course category
    """
    # Using sets for faster membership testing
    science_tech = {33, 9003, 9119, 9130}
    social_science = {171, 9070, 9773, 9853, 9238, 8014}
    business_mgmt = {9147, 9670, 9991, 9254}
    
    if x in science_tech:
        return 'Science & Technology'
    elif x in social_science:
        return 'Social Science'
    elif x in business_mgmt:
        return 'Business & Management'
    else:
        return 'Health Science'

# Enhanced country cleaning function
def clean_country(x):
    """
    Groups countries into geographical regions.
    
    Parameters:
    x (int): The original country code
    
    Returns:
    str: The geographical region
    """
    # Note: 26 appears twice in your original Africa list (typo?)
    latin_america = {101, 109, 108, 41}
    east_europe = {105, 103, 100, 62}
    africa = {26, 25, 24, 22, 21}  # Removed duplicate 26
    nw_europe = {14, 17, 2, 13}
    
    if x in latin_america:
        return 'Latin America'
    elif x in east_europe:
        return 'East Europe'
    elif x in africa:
        return 'Africa'
    elif x in nw_europe:
        return 'North & West Europe'
    else:
        return 'South Europe'  # Fixed typo from "europe" to "Europe"

# Enhanced occupation cleaning function
def clean_occupation(x):
    """
    Categorizes occupations into broader groups.
    
    Parameters:
    x (int): The original occupation code
    
    Returns:
    str: The occupation category
    """
    management = {1, 112, 114}
    professional = {2, 121, 122, 123, 124}
    technician = {3, 131, 132, 134, 135}
    administrative = {4, 141, 143, 144}
    service_sales = {5, 151, 152, 153, 154}
    labour = {6, 7, 8, 9, 161, 163, 171, 172, 174, 175, 
              181, 182, 183, 192, 193, 194, 195}
    armed_forces = {10, 101, 102, 103}
    
    if x in management:
        return 'Management'
    elif x in professional:
        return 'Professional'
    elif x in technician:
        return 'Technician'
    elif x in administrative:
        return 'Administrative'
    elif x in service_sales:
        return 'Service & Sales'
    elif x in labour:
        return 'Labour'
    elif x in armed_forces:
        return 'Armed Forces'
    else:
        return 'Unemployed'

# Enhanced education cleaning function
def clean_education(x):
    """
    Categorizes education levels into broader groups.
    
    Parameters:
    x (int): The original education code
    
    Returns:
    str: The education category
    """
    masters = {4, 43}
    doctorate = {5, 44}
    bachelor = {2, 3, 18, 39, 40, 41, 42}
    high_school = {1, 9, 10, 12, 14, 19, 27, 29}
    middle_school = {11, 26, 30, 38}
    primary = {36, 37}
    
    if x in masters:
        return 'Master'
    elif x in doctorate:
        return 'Doctorate'
    elif x in bachelor:
        return 'Bachelor & Specialized Education'
    elif x in high_school:
        return 'High School'
    elif x in middle_school:
        return 'Middle School'
    elif x in primary:
        return 'Primary'
    else:
        return 'No Education'


# In[9]:


# Distribusi variabel target
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Status', palette='viridis')
plt.title('Distribusi Status Mahasiswa')
plt.xlabel('Status')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

#membuat pie chart untuk status mahasiswa
df['Status'].value_counts(normalize=True).plot(kind = 'pie', autopct='%1.1f%%',title = 'Distribusi Status Mahasiswa',ylabel= '')


# **Catatan / Insight :**
# 
# Persentase dropout (DO) sebesar 32.1%, sementara persentase kelulusan 49.9% dan 17.9% mahasiswa lainnya masih menjalani kuliah.

# In[10]:


# Distribusi umur saat masuk
plt.figure(figsize=(6, 4))
sns.histplot(df['Age_at_enrollment'], bins=20, kde=True, color='steelblue')
plt.title('Distribusi Usia Saat Masuk')
plt.xlabel('Usia')
plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# Mayoritas mahasiswa mendaftar masuk universitas pada usia 20 tahun

# In[11]:


# Set style
sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
plt.figure(figsize=(18, 6))  # Wider figure for side-by-side charts

# First chart - Boxplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first position
sns.boxplot(data=df, x='Status', y='Age_at_enrollment',
            order=['Dropout', 'Enrolled', 'Graduate'],
            showfliers=False, width=0.5)
plt.title('Perbandingan Usia Rata-rata per Status')
plt.xlabel('Status')
plt.ylabel('Usia')
plt.xticks(rotation=45)

# Add average annotations
avg_age = df.groupby('Status')['Age_at_enrollment'].mean()
for i, status in enumerate(['Dropout', 'Enrolled', 'Graduate']):
    plt.text(i, avg_age[status]+0.5, f"Avg: {avg_age[status]:.1f}", 
             ha='center', fontweight='bold')

# Second chart - Stacked bar

# Create age groups
df['Age_Group'] = pd.cut(df['Age_at_enrollment'], 
                        bins=[17, 20, 23, 26, 30, 60],
                        labels=['17-20', '21-23', '24-26', '27-30', '>30'])

age_status = df.groupby(['Age_Group', 'Status']).size().unstack()
age_status = age_status.div(age_status.sum(axis=1), axis=0) * 100
age_status[['Dropout', 'Enrolled', 'Graduate']].plot(
    kind='bar', stacked=True, color=['#ff6b6b', '#48dbfb', '#1dd1a1'])
plt.title('Persentase Status per Kelompok Usia')
plt.xlabel('Kelompok Usia')
plt.ylabel('Persentase (%)')
plt.legend(title='Status', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# 1. Mahasiswa yang Dropout (DO) cenderung lebih tua, atau memiliki usia lebih dari 25 tahun
# 2. Mahasiswa dengan usia muda cenderung lebih sukses

# In[12]:


# --- Distribusi Gender dan Scholarship ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Gender distribution
gender_counts = df['Gender'].value_counts()
axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#ffcccb', '#add8e6'])
axes[0].set_title('Distribusi Gender (0 – Perempuan, 1 – Laki-laki)')

# Scholarship distribution
scholarship_counts = df['Scholarship_holder'].value_counts()
axes[1].pie(scholarship_counts, labels=scholarship_counts.index, autopct='%1.1f%%', colors=['#d3d3d3', '#90ee90'])
axes[1].set_title('Penerima Beasiswa (0 – Tidak, 1 – Ya)')

plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# 1. Mayoritas mahasiswa didominasi oleh perempuan sebesar 64.8%, sedangkan mahasiswa laki-laki sebesar 35.2%
# 2. Hanya 24.8% dari total mahasiswa yang mendapatkan beasiswa, sedangkan mayoritas 75.2% mahasiswa membayar biaya kuliah secara mandiri / pribadi 

# In[13]:


plt.figure(figsize=(10,6))
# Use the exact column name from your DataFrame
sns.countplot(x='Marital_status', data=df)  # or whatever the exact column name is
plt.title('Distribution of Marital Status')
plt.xticks([0,1,2,3,4,5], ['Single','Married','Widower','Divorced','Facto Union','Legally Separated'])
plt.show()


# **Catatan / Insight :**
# 
# Mayoritas mahasiswa masih lajang (single) atau belum menikah

# In[14]:


df


# In[15]:


# Apply cleaning functions
df['course_category'] = df['Course'].apply(clean_course)
df['country_region'] = df['Nationality'].apply(clean_country)
df['occupation_group'] = df['Mothers_occupation'].apply(clean_occupation)
df['education_level'] = df['Mothers_qualification'].apply(clean_education)
df['father_occupation_group'] = df['Fathers_occupation'].apply(clean_occupation)
df['father_education_level'] = df['Fathers_qualification'].apply(clean_education)


# In[16]:


df


# In[17]:


# Set style for better looking plots
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

# 1. Course Categories Distribution
plt.subplot(2, 2, 1)
course_order = ['Science & Technology', 'Social Science', 'Business & Management', 'Health Science']
sns.countplot(data=df, x='course_category', order=course_order, palette='viridis')
plt.title('Distribution of Course Categories')
plt.xticks(rotation=45)
plt.xlabel('Course Category')

# 2. Country Regions Distribution
plt.subplot(2, 2, 2)
region_order = ['Latin America', 'East Europe', 'Africa', 'North & West Europe', 'South Europe']
sns.countplot(data=df, x='country_region', order=region_order, palette='mako')
plt.title('Distribution by Country Region')
plt.xticks(rotation=45)
plt.xlabel('Region')

# 3. Occupation Groups Comparison (Mother vs Father)
plt.subplot(2, 2, 3)
occupation_order = ['Management', 'Professional', 'Technician', 'Administrative', 
                   'Service & Sales', 'Labour', 'Armed Forces', 'Unemployed']

# Create a combined dataframe for occupation comparison
occupation_df = pd.DataFrame({
    'Occupation': pd.concat([df["Mothers_occupation"], df["Fathers_occupation"]]),
    'Parent': ['Mother']*len(df) + ['Father']*len(df),
    'Occupation Group': pd.concat([df['occupation_group'], df['father_occupation_group']])
})

sns.countplot(data=occupation_df, x='Occupation Group', hue='Parent', 
              order=occupation_order, palette='rocket')
plt.title("Parents' Occupation Groups Comparison")
plt.xticks(rotation=45)
plt.xlabel('Occupation Group')
plt.legend(title='Parent')

# 4. Education Levels Comparison (Mother vs Father)
plt.subplot(2, 2, 4)
education_order = ['No Education', 'Primary', 'Middle School', 'High School',
                  'Bachelor & Specialized Education', 'Master', 'Doctorate']

# Create a combined dataframe for education comparison
education_df = pd.DataFrame({
    'Education': pd.concat([df["Mothers_qualification"], df["Fathers_qualification"]]),
    'Parent': ['Mother']*len(df) + ['Father']*len(df),
    'Education Level': pd.concat([df['education_level'], df['father_education_level']])
})

sns.countplot(data=education_df, x='Education Level', hue='Parent', 
              order=education_order, palette='flare')
plt.title("Parents' Education Levels Comparison")
plt.xticks(rotation=45)
plt.xlabel('Education Level')
plt.legend(title='Parent')

plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# 1. Jurusan yang paling banyak diambil yaitu Social Science. Jurusan untuk kluster Science & Technology paling sedikit diminati.
# 2. Mayoritas mahasiswa berasal dari Eropa Selatan
# 3. Mayoritas orang tua mahasiswa bekerja sebagai buruh (labour)
# 4. Mayoritas orang tua mahasiswa memiliki pendidikan terakhir tamatan SMA / high school

# In[18]:


# --- Perbandingan Status berdasarkan Gender ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', hue='Status', palette='cool')
plt.title('Gender vs Status Mahasiswa')
plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# 1. Jumlah mahasiswa perempuan yang Dropout (DO) sedikit lebih tinggi dibandingkan mahasiswa laki-laki.
# 2. Namun demikian, jumlah mahasiswa laki-laki yang DO lebih banyak dibandingkan dengan jumlah mahasiswa laki-laki yang lulus.

# In[19]:


# --- Status vs Scholarship Holder ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Scholarship_holder', hue='Status', palette='coolwarm')
plt.title('Beasiswa vs Status Mahasiswa')
plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# Jumlah mahasiswa penerima beasiswa yang DO jauh lebih sedikit dibandingkan dengan jumlah mahasiswa DO yang tidak menerima beasiswa.

# In[20]:


plt.figure(figsize=(16, 7))
sns.countplot(data=df, x='course_category', hue='Status', palette='Set2')
plt.title('Distribusi Status Mahasiswa per Course')
plt.xlabel('Nama Course')
plt.ylabel('Jumlah Mahasiswa')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Status')
plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# Dilihat dari rasio DO dan kelulusan, mahasiswa yang DO mayoritas didominasi pada jurusan Business & Management serta Science & Technology

# In[21]:


# Set style
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
plt.figure(figsize=(18, 20))

# 1. Grade Distribution by Status
plt.subplot(4, 2, 1)
sns.boxplot(data=df, x='Status', y='Curricular_units_2nd_sem_grade', 
            order=['Dropout', 'Enrolled', 'Graduate'], showfliers=False)
plt.title('2nd Semester Grade Distribution by Status')
plt.xlabel('')
plt.ylabel('Grade')

plt.subplot(4, 2, 2)
sns.boxplot(data=df, x='Status', y='Curricular_units_1st_sem_grade',
            order=['Dropout', 'Enrolled', 'Graduate'], showfliers=False)
plt.title('1st Semester Grade Distribution by Status')
plt.xlabel('')
plt.ylabel('Grade')

# 2. Approved Courses by Status
plt.subplot(4, 2, 3)
sns.boxplot(data=df, x='Status', y='Curricular_units_2nd_sem_approved',
            order=['Dropout', 'Enrolled', 'Graduate'], showfliers=False)
plt.title('2nd Semester Approved Courses by Status')
plt.xlabel('')
plt.ylabel('Number of Approved Courses')

plt.subplot(4, 2, 4)
sns.boxplot(data=df, x='Status', y='Curricular_units_1st_sem_approved',
            order=['Dropout', 'Enrolled', 'Graduate'], showfliers=False)
plt.title('1st Semester Approved Courses by Status')
plt.xlabel('')
plt.ylabel('Number of Approved Courses')

# 3. Evaluation Participation
plt.subplot(4, 2, 5)
sns.boxplot(data=df, x='Status', y='Curricular_units_2nd_sem_evaluations',
            order=['Dropout', 'Enrolled', 'Graduate'], showfliers=False)
plt.title('2nd Semester Evaluations by Status')
plt.xlabel('')
plt.ylabel('Number of Evaluations')

plt.subplot(4, 2, 6)
sns.boxplot(data=df, x='Status', y='Curricular_units_1st_sem_evaluations',
            order=['Dropout', 'Enrolled', 'Graduate'], showfliers=False)
plt.title('1st Semester Evaluations by Status')
plt.xlabel('')
plt.ylabel('Number of Evaluations')

plt.tight_layout()
plt.show()

# Additional Visualizations
plt.figure(figsize=(18, 6))

plt.tight_layout()
plt.show()


# **Catatan / Insight :**
# 
# 1. Mahasiswa yang memiliki nilai rendah pada Semester 1 dan Semester 2 lebih banyak yang Dropout (DO)
# 2. Mahasiswa yang Dropout (DO) mengambil jumlah mata kuliah (curricular unit) lebih sedikit dibandingkan mahasiswa yang tidak DO (enrolled, graduate) -> apakah karena IPK lebih rendah sehingga tidak dapat mengambil mata kuliah lebih banyak, atau karena faktor biaya sehingga mahasiswa DO mengambil mata kuliah lebih sedikit ?
# 

# In[22]:


# Tuition Fees Status
fee_status = df.groupby(['Status', 'Tuition_fees_up_to_date']).size().unstack()
fee_status = fee_status.div(fee_status.sum(axis=1), axis=0)
fee_status.loc[['Dropout', 'Enrolled', 'Graduate']].plot(
    kind='bar', stacked=True, color=['#fc8d62', '#66c2a5'])
plt.title('Tuition Fee Status by Student Status')
plt.xlabel('')
plt.ylabel('Proportion')
plt.legend(title='Fees Up-to-date', labels=['No', 'Yes'], bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)


# **Catatan / Insight :**
# 
# 1. Mahasiswa yang DO cenderung memiliki masalah terkait pembiayaan kuliah (tuition fees), terlebih hanya sedikit mahasiswa yang mendapatkan beasiswa dilihat dari visualisasi data sebelumnya

# # Kesimpulan / Conclusion

# Berdasarkan Exploratory Data Analysis (EDA) yang telah dilakukan, berikut adalah temuan utama:
# 
# 1. Persentase mahasiswa Dropout (DO) di Jaya Jaya Institut adalah 32.1%, yang dimana angka tersebut relatif tinggi dibandingkan dengan persentase kelulusan 49.9%.
# 2. Mahasiswa yang Dropout memiliki:
#    - Nilai lebih rendah (Curricular_units_1st/2nd_sem_grade) dibandingkan yang Graduate/Enrolled.
#    - Jumlah mata kuliah yang disetujui (approved) lebih sedikit.
#    - Partisipasi evaluasi (evaluations) lebih rendah.
# 3. Faktor Non-Akademik yang Berpengaruh :
#    - Tuition_fees_up_to_date:
#      Mahasiswa yang tidak membayar tepat waktu cenderung memiliki risiko Dropout lebih tinggi.
#    - Scholarship_holder:
#      Penerima beasiswa memiliki tingkat kelulusan lebih tinggi.
#    - Demografi:
#      - Usia (Age_at_enrollment): Mahasiswa yang lebih muda cenderung lebih sukses. Mahasiswa yang lebih tua lebih beresiko Dropout (DO)
#      - Gender: Tidak ada perbedaan signifikan dalam dropout rate.
# 4. Perbedaan antar Program Studi (course_category)
#    Beberapa program studi memiliki tingkat Dropout lebih tinggi (misalnya, Sosial & Teknologi), sementara lainnya (misalnya, Bisnis & Kesehatan) memiliki tingkat kelulusan lebih baik.

# **Rekomendasi untuk Mengurangi Tingkat Dropout**
# 
# Berdasarkan temuan EDA, berikut rekomendasi untuk institusi pendidikan:
# 
# 1. Intervensi Akademik
#    - Program Bimbingan Akademik: Fokus pada mahasiswa dengan nilai rendah di semester 1 karena mereka berisiko tinggi Dropout.
#    - Berikan mentoring tambahan untuk mata kuliah dengan tingkat kegagalan tinggi.
# 
# 2. Sistem Peringatan Dini (Early Warning System):
#    - Gunakan machine learning untuk memprediksi mahasiswa berisiko Dropout berdasarkan kinerja semester 1.
#    - Berikan notifikasi kepada dosen/wali jika mahasiswa memiliki nilai di bawah ambang batas.
# 
# 3. Dukungan Finansial
#    - Beasiswa & Bantuan Biaya Kuliah:
#    - Prioritaskan mahasiswa dari keluarga kurang mampu (Debtor = 1).
#    - Berikan penyelesaian biaya kuliah fleksibel untuk mengurangi tekanan finansial.
# 
# 4. Program Kerja Sambil Kuliah:
#    Kolaborasi dengan industri untuk memberikan part-time job bagi mahasiswa yang membutuhkan.
# 
# 5. Peningkatan Keterlibatan Mahasiswa
#    - Tingkatkan Partisipasi Evaluasi: Mahasiswa yang tidak mengikuti evaluasi cenderung Dropout.
#    - Berikan insentif (poin tambahan, sertifikat) untuk meningkatkan kehadiran ujian.
# 
# 6. Kegiatan Non-Akademik:
#    - Program peer mentoring dan komunitas belajar untuk meningkatkan motivasi.
# 
# 7. Segmentasi Mahasiswa:
#    - Kelompokkan mahasiswa berdasarkan risiko Dropout (rendah, sedang, tinggi) dan berikan pendekatan berbeda.
# 
# Dengan strategi ini, Jaya Jaya Institut dapat mengurangi tingkat Dropout dan meningkatkan keberhasilan mahasiswa.

# # Data Preparation / Preprocessing

# In[23]:


df


# In[24]:


print(df.columns.tolist())


# In[25]:


df.info()


# In[26]:


# Buat salinan data
df_corr = df.copy()

# Encode kolom 'Status' ke numerik
# Misalnya: 'Dropout' = 0, 'Graduate' = 1, 'Enrolled' = 2
status_mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2, }
df_corr['Status_encoded'] = df_corr['Status'].map(status_mapping)

# Gabungkan kolom numerik saja + Status_encoded
numeric_df = df_corr.select_dtypes(include=['int64', 'float64'])
numeric_df['Status_encoded'] = df_corr['Status_encoded']

# Hitung korelasi
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix[['Status_encoded']].sort_values(by='Status_encoded', ascending=False),
            annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap terhadap Status Mahasiswa (Encoded)')
plt.show()


# In[27]:


relevant_features = correlation_matrix['Status_encoded'][abs(correlation_matrix['Status_encoded']) > 0.1].sort_values(ascending=False)
print(relevant_features)


# In[28]:


# Feature selection based on correlation and domain knowledge
selected_features = [
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_1st_sem_evaluations',
    'Scholarship_holder',
    'Application_mode',
    'Gender',
    'Debtor',
    'Age_at_enrollment',
    'course_category',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_2nd_sem_enrolled'
]

# Create a binary target variable
df['Dropout'] = df['Status'].apply(lambda x: 0 if x in ['Graduate', 'Enrolled'] else 1)

# Split the data into features and target
X = df[selected_features]
y = df['Dropout']

# Perform stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define numeric and categorical features
numeric_features = [
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_1st_sem_evaluations',
    'Age_at_enrollment',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_2nd_sem_enrolled'
]

categorical_features = [
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Application_mode',
    'Gender',
    'Debtor',
    'course_category'
]

# Define transformations for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformations into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# ## Modeling

# In[29]:


from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd

# Model list
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Results storage
results = []

# Training and evaluation
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'ROC AUC': roc
    })

# Show comparison
results_df = pd.DataFrame(results)
print(results_df)


# ## Evaluation

# In[30]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Prediksi test set
y_pred_gb = pipeline.predict(X_test)

# Hitung confusion matrix
cm = confusion_matrix(y_test, y_pred_gb)

# Tampilkan confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Dropout', 'Dropout'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Gradient Boosting')
plt.show()

# Hitung akurasi manual dari confusion matrix
tn, fp, fn, tp = cm.ravel()  # unpack confusion matrix
manual_accuracy = (tp + tn) / (tn + fp + fn + tp)

print(f"Akurasi dari Confusion Matrix: {manual_accuracy:.4f}")


# In[31]:


best_model = GradientBoostingClassifier(random_state=42)

# Kita training lagi (kalau belum) dengan preprocessor
pipeline_best = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

pipeline_best.fit(X_train, y_train)

# Lalu simpan model pipeline ini
joblib.dump(pipeline_best, "model/gradient_boosting_dropout_model.pkl")

print("Model Gradient Boosting berhasil disimpan!")


# In[32]:


# Load model Gradient Boosting yang sudah disimpan
loaded_pipeline = joblib.load("model/gradient_boosting_dropout_model.pkl")

# Coba prediksi ulang menggunakan loaded model
y_pred_loaded = loaded_pipeline.predict(X_test)

# Cek hasil akurasi
from sklearn.metrics import accuracy_score

loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f"Akurasi model Gradient Boosting yang diload: {loaded_accuracy:.4f}")


# In[33]:


df


# In[34]:


df.info()


# # Mengupload cleaned_df ke Supabase

# In[35]:


# Save the cleaned dataframe to a CSV file
df.to_csv('data/cleaned_student_data.csv', index=False)
print("CSV file 'cleaned_student_data.csv' created successfully!")


# In[36]:


# Read the CSV file
uploaded_df = pd.read_csv('data/cleaned_student_data.csv')

# Supabase connection details
URL = "postgresql://postgres.nttbzzncjgfvqkmucidt:root123@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
engine = create_engine(URL)

# Upload the dataframe to Supabase
uploaded_df.to_sql('cleaned_student_data', engine, if_exists='replace', index=False)
print("Data successfully uploaded to Supabase!")


# # Membuat File requirements.txt

# In[37]:


get_ipython().system('pip install pipreqs')


# In[ ]:


get_ipython().system('pipreqs . --force')


# In[ ]:


ls


# In[ ]:


get_ipython().system('type requirements.txt')


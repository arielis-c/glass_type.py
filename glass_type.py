import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "Building window float processed".upper()
    elif glass_type == 2:
        return "Building windows non-float processed".upper()
    elif glass_type == 3:
        return "Vehicle windows float processed".upper()
    elif glass_type == 4:
        return "Vehicle windows non-float processed".upper()
    elif glass_type == 5:
        return "Containers".upper()
    else:
        return "Headlamps".upper()
st.title("Glass type predicted")
st.sidebar.title("Exporative data analysis")

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Full dataset")
    st.dataframe(glass_df)

st.sidebar.subheader("Scatter plot")

st.set_option("deprecation.showPyplotGlobalUse", False)

features_list = st.sidebar.multiselect("Select the x-axis values", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))


for feature in features_list:
    st.subheader("Scatter Plot")
    plt.figure(figsize = (12, 7))
    sns.scatterplot(x = feature, y = "GlassType", data = glass_df)
    st.pyplot()

st.sidebar.subheader("Visualization Selector")
plot_types = st.sidebar.multiselect("Select the charts or plots", ("Histogram", "Boxplot", "Count plot", "Pie chart", "Heat map", "Pair plot"))
if "Histogram" in plot_types:
    st.subheader("Histogram")
    columns = st.sidebar.selectbox("Select the column to create it's histogram", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    plt.figure(figsize = (15, 7))
    plt.title(f"Histogram for {columns}")
    plt.hist(glass_df[columns], bins = "sturges", edgecolor = "black")
    st.pyplot()

if "Boxplot" in plot_types:
    st.subheader("Boxplot")
    columns = st.sidebar.selectbox("Select the column to create it's boxplot", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    plt.figure(figsize = (15, 7))
    plt.title(f"Boxplot for {columns}")
    sns.boxplot(glass_df[columns])
    st.pyplot()

if "Count plot" in plot_types:
    st.subheader("Count plot")
    plt.figure(figsize = (15, 7))
    sns.countplot(x = "GlassType", data = glass_df)
    st.pyplot()

if "Pie chart" in plot_types:
    st.subheader("Pie chart")
    pie_data = glass_df["GlassType"].value_counts()
    plt.figure(figsize = (15, 7))
    plt.pie(pie_data, labels = pie_data.index, autopct = "%1.2f%%", startangle = 30, explode = np.linspace(.06, .16, 6))
    st.pyplot()

if "Heat map" in plot_types:
    st.subheader("Heat map")
    plt.figure(figsize = (14, 7))
    ax = sns.heatmap(glass_df.corr(), annot = True)
    bottom, top = ax.get_ylim()
    st.pyplot()

if "Pair plot" in plot_types:
    st.subheader("Pair plot")
    plt.figure(figsize = (14, 5))
    sns.pairplot(glass_df)
    st.pyplot()

st.sidebar.subheader("Select your values")
ri = st.sidebar.slider("Input RI", float(glass_df["RI"].min()), float(glass_df["RI"].max()))
na = st.sidebar.slider("Input Na", float(glass_df["Na"].min()), float(glass_df["Na"].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df["Mg"].min()), float(glass_df["Mg"].max()))
al = st.sidebar.slider("Input Al", float(glass_df["Al"].min()), float(glass_df["Al"].max()))
si = st.sidebar.slider("Input Si", float(glass_df["Si"].min()), float(glass_df["Si"].max()))
k = st.sidebar.slider("Input K", float(glass_df["K"].min()), float(glass_df["K"].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df["Ca"].min()), float(glass_df["Ca"].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df["Ba"].min()), float(glass_df["Ba"].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df["Fe"].min()), float(glass_df["Fe"].max()))

st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine", "Random Forest Classifier", "Logistic Regression"))

if classifier == "Support Vector Machine":
    st.sidebar.subheader("Model Hyper Parameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st.sidebar.number_input("Gama", 1, 100, step = 1)

    if st.sidebar.button("Classify"):
        st.subheader("Support Vector Machine")
        svc_model = SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train, y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The type of glass predicted:", glass_type)
        st.write("Accuracy:", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()


if classifier == "Random Forest Classifier":
    st.sidebar.subheader("Model Hyper Parameters")

    n_estimators_input = st.sidebar.number_input("Number of Trees in the Forest", 100, 5000, step = 10)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    max_depth_input = st.sidebar.number_input("Maximum Depth of the Tree", 1, 100, step = 1)

    if st.sidebar.button("Classify"):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train, y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The type of glass predicted:", glass_type)
        st.write("Accuracy:", accuracy.round(2))
        plot_confusion_matrix(rf_clf, X_test, y_test)
        st.pyplot()


if classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyper Parameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    max_iteration_input = st.sidebar.number_input("Maximum Iteration", 1, 100, step = 1)

    if st.sidebar.button("Classify"):
        st.subheader("Logistic Regression")
        lr = LogisticRegression(C = c_value, max_iter = max_iteration_input)
        lr.fit(X_train, y_train)
        accuracy = lr.score(X_test, y_test)
        glass_type = prediction(lr, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The type of glass predicted:", glass_type)
        st.write("Accuracy:", accuracy.round(2))
        plot_confusion_matrix(lr, X_test, y_test)
        st.pyplot()









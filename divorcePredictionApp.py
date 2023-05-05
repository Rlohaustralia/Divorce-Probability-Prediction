from model_prediction import *
from PIL import Image
import pandas

st.write("")
st.write("")
st.write("")
st.write("")
st.title("Divorce Prediction App 💔")
st.write("")
image = Image.open('divorceFacts_image.png')
st.image(image)


# Ask user the questions
st.write("")
st.write("")
st.write("👉 Please answer for all the questions")
st.write("")
df = pandas.read_csv('Marriage_Divorce_DB.csv')


# Ask user
answers = []
for col in range(29):
    column_name = df.columns[col]
    min_value = float(df[column_name].min())
    max_value = float(df[column_name].max())
    default_value = float(df[column_name].median())
    selected_value = st.slider(f"Q,{col+1} 🗣️ Select {column_name}",min_value, max_value)
    st.write("You selected:", selected_value)
    answers.append(selected_value)


# Load the best model
model_filename = 'best_regr_model.h5'
loaded_model = pickle.load(open(model_filename, 'rb'))


# Predict the probability of divorce based on the user' choices
if st.button("Predict"):
    # set the result value and get rid of the square bracket from pandas df
    result = loaded_model.predict([answers])[0]
    average_value = float(df['Divorce Probability'].mean())
    if result > average_value:
        st.write("Based on your answers, the probability of divorce is:", str(result))
        st.write("You have a higher probability of divorce than average 😥💦")
    elif result == average_value:
        st.write("Based on your answers, the probability of divorce is:", str(result))
        st.write("You have an average probability of divorce 🤔💭")
    elif result < average_value:
        st.write("Based on your answers, the probability of divorce is:", str(result))
        st.write("You have a lower probability of divorce than average 👩‍❤️‍👨")
    else:
        st.write("Unexpected error has occurred. Please try again...")

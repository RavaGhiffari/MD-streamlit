import streamlit as st
import joblib as jb

def main():
  st.title('Dermatology Machine Learning')
  st.info('This app uses machine learning')
  #input data by user
  erythema = st.radio('Erythema', min_value=0, max_value=3, value=2)

if __name__ == "__main__":
  main()

import streamlit as st
import joblib as jb

def main():
  st.title('Dermatology Machine Learning')
  st.info('This app uses machine learning')
  # Input Data by User

  erythema = st.slider('Erythema', min_value = 0, max_value = 3, value = 2)

  scaling = st.slider('Scaling', min_value = 0, max_value = 3, value = 2)

  definite_borders = st.slider('Definite Borders', min_value = 0, max_value = 3, value = 2)

  itching = st.slider('Itching', min_value = 0, max_value = 3, value = 0)

  koebner_phenomenon = st.slider('koebner_phenomenon', min_value = 0, max_value = 3, value = 0)

  polygonal_papules = st.slider('Polygonal Papules', min_value = 0, max_value = 3, value = 0)

  follicular_papules = st.slider('Follicular Papules', min_value = 0, max_value = 3, value = 0)

  oral_mucosal_involvement = st.slider('Oral Mucosal Involvement', min_value = 0, max_value = 3, value = 0)

  knee_and_elbow_involvement = st.slider('Knee and Elbow Involvement', min_value = 0, max_value = 3, value = 0)

  scalp_involvement = st.slider('Scalp Involvement', min_value = 0, max_value = 3, value = 0)

  family_history = st.slider('Family History', min_value = 0, max_value = 1, value = 0)

  melanin_incontinence = st.slider('Melanin Incontinence', min_value = 0, max_value = 3, value = 0)

  eosinophils_infiltrate = st.slider('Eosinophils Infiltrate', min_value = 0, max_value = 2, value = 0)

  PNL_infiltrate = st.slider('PNL Infiltrate', min_value = 0, max_value = 3, value = 0)

  fibrosis_papillary_dermis = st.slider('Fibrosis Papillary Dermis', min_value = 0, max_value = 3, value = 0)

  exocytosis = st.slider('Exocytosis', min_value = 0, max_value = 3, value = 2)

  acanthosis = st.slider('Acanthosis', min_value = 0, max_value = 3, value = 2)

  hyperkeratosis = st.slider('Hyperkeratosis', min_value = 0, max_value = 3, value = 0)

  parakeratosis = st.slider('Parakeratosis', min_value = 0, max_value = 3, value = 2)

  clubbing_rete_ridges = st.slider('Clubbing Rete Ridges', min_value = 0, max_value = 3, value = 0)

  elongation_rete_ridges = st.slider('Elongation Rete Ridges', min_value = 0, max_value = 3, value = 0)

  thinning_suprapapillary_epidermis = st.slider('Thinning Suprapapillary Epidermis', min_value = 0, max_value = 3, value = 0)

  spongiform_pustule = st.slider('Spongiform Pustule', min_value = 0, max_value = 3, value = 0)

  munro_microabcess = st.slider('Munro Microabcess', min_value = 0, max_value = 3, value = 0)

  focal_hypergranulosis = st.slider('Focal Hypergranulosis', min_value = 0, max_value = 3, value = 0)

  disappearance_granular_layer = st.slider('disappearance_granular_layer', min_value = 0, max_value = 3, value = 0)

  vacuolisation_damage_basal_layer = st.slider('vacuolisation damage basal layer', min_value = 0, max_value = 3, value = 0)

  spongiosis = st.slider('Spongiosis', min_value = 0, max_value = 3, value = 0)

  saw_tooth_appearance_retes = st.slider('saw tooth appearance retes', min_value = 0, max_value = 3, value = 0)

  follicular_horn_plug = st.slider('follicular horn plug', min_value = 0, max_value = 3, value = 0)

  perifollicular_parakeratosis = st.slider('perifollicular parakeratosis', min_value = 0, max_value = 3, value = 0)

  inflammatory_mononuclear_infiltrate = st.slider('inflammatory mononuclear infiltrate', min_value = 0, max_value = 3, value = 2)

  band_like_infiltrate = st.slider('band like infiltrate', min_value = 0, max_value = 3, value = 0)

  age = st.slider('age', min_value = 0, max_value = 75, value = 40)

if __name__ == "__main__":
  main()

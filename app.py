"""Streamlit app for Gemini 1.0 Pro."""
import os

import streamlit as st
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import HarmCategory
from vertexai.generative_models import Part


PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
  """Load the generative models for text and multimodal generation.

  Returns:
      Tuple: A tuple containing the text model and multimodal model.
  """
  text_model_pro = GenerativeModel("gemini-1.0-pro")
  multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
  return text_model_pro, multimodal_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
  safety_settings = {
      HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
          HarmBlockThreshold.BLOCK_NONE
      ),
      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
          HarmBlockThreshold.BLOCK_NONE
      ),
  }

  responses = model.generate_content(
      prompt,
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=stream,
  )

  final_response = []
  for response in responses:
    try:
      # st.write(response.text)
      final_response.append(response.text)
    except IndexError:
      # st.write(response)
      final_response.append("")
      continue
  return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
  generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
  responses = model.generate_content(
      prompt_list, generation_config=generation_config, stream=stream
  )
  final_response = []
  for response in responses:
    try:
      final_response.append(response.text)
    except IndexError:
      pass
  return "".join(final_response)


st.header("Autoscreenwriter", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()


st.write("Using Gemini 1.0 Pro - Text only model")
st.subheader("Generate a story")

# Story premise
character_name = st.text_input(
    "Enter character name: \n\n", key="character_name", value="Mittens"
)
character_type = st.text_input(
    "What type of character is it? \n\n", key="character_type", value="Cat"
)
character_persona = st.text_input(
    "What personality does the character have? \n\n",
    key="character_persona",
    value="Mitten is a very friendly cat.",
)
character_location = st.text_input(
    "Where does the character live? \n\n",
    key="character_location",
    value="Andromeda Galaxy",
)
story_premise = st.multiselect(
    "What is the story premise? (can select multiple) \n\n",
    [
        "Love",
        "Adventure",
        "Mystery",
        "Horror",
        "Comedy",
        "Sci-Fi",
        "Fantasy",
        "Thriller",
    ],
    key="story_premise",
    default=["Love", "Adventure"],
)
creative_control = st.radio(
    "Select the creativity level: \n\n",
    ["Low", "High"],
    key="creative_control",
    horizontal=True,
)
length_of_story = st.radio(
    "Select the length of the story: \n\n",
    ["Short", "Long"],
    key="length_of_story",
    horizontal=True,
)

if creative_control == "Low":
    temperature = 0.30
else:
    temperature = 0.95

max_output_tokens = 2048

prompt = f"""Write a {length_of_story} story based on the following premise: \n
character_name: {character_name} \n
character_type: {character_type} \n
character_persona: {character_persona} \n
character_location: {character_location} \n
story_premise: {",".join(story_premise)} \n
If the story is "short", then make sure to have 5 chapters or else if it is "long" then 10 chapters.
Important point is that each chapters should be generated based on the premise given above.
First start by giving the book introduction, chapter introductions and then each chapter. It should also have a proper ending.
The book should have prologue and epilogue.
"""
config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
}

generate_t2t = st.button("Generate my story", key="generate_t2t")
if generate_t2t and prompt:
# st.write(prompt)
  with st.spinner("Generating your story using Gemini 1.0 Pro ..."):
    first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
  with first_tab1:
    response = get_gemini_pro_text_response(
        text_model_pro,
        prompt,
        generation_config=config,
    )
    if response:
      st.write("Your story:")
      st.write(response)
  with first_tab2:
    st.text(prompt)

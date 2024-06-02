import streamlit as st
import numpy as np
import PIL
from main import SDXL, Kandinsky, Generator, LLM

image_kandinsky = np.zeros((10, 10))

@st.cache_resource
def init_model():
    return [LLM(), 0, Kandinsky(), Generator()]

llm, sdxl, kandinsky, generator = init_model()

with st.sidebar: 
    use_llm = st.toggle('use LLM?')

    prompt = st.text_input('Type your prompt here')

    if st.button('Generate'):
        if use_llm:
            prompt = llm.get_prompt(prompt)
        

        # image_sdxl = sdxl.get_image(prompt)
        image_kandinsky = kandinsky.get_image(prompt+' in style realism.', 1024, 1024)

        generator.get_broadcast_visuals(image_kandinsky)
    else:
        pass

col1, col2 = st.columns(2)

col1.write('SDXL')
# col1.image(image_sdxl)

col2.write('Kandinsky')
col2.image(image_kandinsky)

if use_llm:
    st.write('by promt:')
    st.write(prompt)
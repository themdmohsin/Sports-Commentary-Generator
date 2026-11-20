from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.generator import Generator


@st.cache_resource
def load_model():
    return Generator()


st.set_page_config(page_title="Sports Commentary Generator")
st.title("Sports Commentary Generator")
event = st.text_area("Match event", "Tara hits a three-pointer to level it at 81-81.", max_chars=240)
if st.button("Generate commentary", type="primary", disabled=not event.strip()):
    try:
        with st.spinner("Calling the play..."):
            line = load_model().generate(event)
        if line:
            st.subheader("Commentary")
            st.write(line)
        else:
            st.info("Try a more specific play, including the player's name.")
    except (OSError, ValueError) as error:
        st.error(f"Could not generate commentary. {error}")


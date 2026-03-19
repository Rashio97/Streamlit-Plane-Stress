import streamlit as st

st.title("Exempel i Streamlit")

if "count" not in st.session_state:
    st.session_state.count = 0

col1, col2, col3 = st.columns([1,2,1])

with col1:
    if st.button("Klicka här"):
        st.session_state.count += 1

with col3: st.write(f"Antal klick: {st.session_state.count}")
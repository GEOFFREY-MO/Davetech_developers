import streamlit as st
from jinja2 import Template
import subprocess
import os

# LaTeX Templates
TEMPLATES = {
    "Template 1": r"""
    \documentclass{article}
    \begin{document}
    \title{Curriculum Vitae}
    \author{{{ name }}} 
    \date{\today}
    \maketitle
    \section*{Contact Information}
    \begin{tabular}{rl}
    Email: & {{ email }} \\
    Phone: & {{ phone }} \\
    \end{tabular}
    \section*{Education}
    {% for edu in education %}
    \textbf{{ edu.degree }} in {{ edu.field }} \\
    {{ edu.institution }}, {{ edu.year }} \\
    {% endfor %}
    \section*{Experience}
    {% for exp in experience %}
    \textbf{{ exp.position }} \\
    {{ exp.company }}, {{ exp.year }} \\
    {% endfor %}
    \end{document}
    """,
    # Add more templates as needed
}

def create_latex_file(template, context):
    latex_code = Template(template).render(context)
    with open("cv.tex", "w") as f:
        f.write(latex_code)
    subprocess.run(["pdflatex", "cv.tex"], stdout=subprocess.PIPE)

def main():
    st.title("CV Builder")
    
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    
    st.subheader("Education")
    education = []
    for i in range(3):  # You can add more or make it dynamic
        degree = st.text_input(f"Degree {i+1}")
        field = st.text_input(f"Field of Study {i+1}")
        institution = st.text_input(f"Institution {i+1}")
        year = st.text_input(f"Year {i+1}")
        education.append({"degree": degree, "field": field, "institution": institution, "year": year})
    
    st.subheader("Experience")
    experience = []
    for i in range(3):  # You can add more or make it dynamic
        position = st.text_input(f"Position {i+1}")
        company = st.text_input(f"Company {i+1}")
        year = st.text_input(f"Year {i+1}")
        experience.append({"position": position, "company": company, "year": year})
    
    template_choice = st.selectbox("Select Template", list(TEMPLATES.keys()))
    
    if st.button("Generate CV"):
        context = {
            "name": name,
            "email": email,
            "phone": phone,
            "education": education,
            "experience": experience
        }
        create_latex_file(TEMPLATES[template_choice], context)
        st.success("CV Generated successfully!")
        with open("cv.pdf", "rb") as pdf_file:
            st.download_button(
                label="Download CV",
                data=pdf_file,
                file_name="cv.pdf",
                mime="application/octet-stream",
            )

if __name__ == "__main__":
    main()

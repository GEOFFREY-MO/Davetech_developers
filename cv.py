import streamlit as st
import subprocess
from datetime import date

def main():
  st.title("CV Builder")

  # Personal Information
  st.header("Personal Information")
  name = st.text_input("Full Name", key="name")
  email = st.text_input("Email", key="email")
  phone = st.text_input("Phone Number", key="phone")
  address = st.text_area("Address", key="address")

  # Education
  st.header("Education")
  education = []
  num_education = st.number_input("Number of Education Entries", min_value=1, max_value=10, step=1, key="num_education")
  for i in range(num_education):
      st.subheader(f"Education {i+1}")
      school = st.text_input(f"School/University {i+1}", key=f"school_{i}")
      degree = st.text_input(f"Degree {i+1}", key=f"degree_{i}")
      field_of_study = st.text_input(f"Field of Study {i+1}", key=f"field_of_study_{i}")
      start_date = st.date_input(f"Start Date {i+1}", date(2000, 1, 1), key=f"edu_start_date_{i}")
      end_date = st.date_input(f"End Date {i+1}", date(2004, 1, 1), key=f"edu_end_date_{i}")
      education.append({
          "school": school,
          "degree": degree,
          "field_of_study": field_of_study,
          "start_date": start_date,
          "end_date": end_date
      })

  # Work Experience
  st.header("Work Experience")
  work_experience = []
  num_experience = st.number_input("Number of Work Experience Entries", min_value=1, max_value=10, step=1, key="num_experience")
  for i in range(num_experience):
      st.subheader(f"Work Experience {i+1}")
      company = st.text_input(f"Company {i+1}", key=f"company_{i}")
      position = st.text_input(f"Position {i+1}", key=f"position_{i}")
      start_date = st.date_input(f"Start Date {i+1}", date(2000, 1, 1), key=f"work_start_date_{i}")
      end_date = st.date_input(f"End Date {i+1}", date(2004, 1, 1), key=f"work_end_date_{i}")
      description = st.text_area(f"Description {i+1}", key=f"description_{i}")
      work_experience.append({
          "company": company,
          "position": position,
          "start_date": start_date,
          "end_date": end_date,
          "description": description
      })

  # Skills
  st.header("Skills")
  skills = st.text_area("List your skills (separated by commas)", key="skills")

  # Generate CV
  if st.button("Generate CV", key="generate_cv"):
      generate_cv(name, email, phone, address, education, work_experience, skills)

def generate_cv(name, email, phone, address, education, work_experience, skills):
  latex_content = r"""
  \documentclass[lighthipster]{simplehipstercv}
  \usepackage[utf8]{inputenc}
  \usepackage[default]{raleway}
  \usepackage[margin=1cm, a4paper]{geometry}
  \title{New Simple CV}
  \author{LaTeX Ninja}
  \date{July 2024}
  \pagestyle{empty}
  \begin{document}
  \thispagestyle{empty}
  \simpleheader{headercolour}{""" + name.split()[0] + r"""}{""" + name.split()[1] + r"""}{Position}{white}
  \section*{Personal Information}
  \begin{itemize}
      \item \textbf{Name:} """ + name + r"""
      \item \textbf{Email:} """ + email + r"""
      \item \textbf{Phone:} """ + phone + r"""
      \item \textbf{Address:} """ + address + r"""
  \end{itemize}
  \section*{Education}
  """
  for edu in education:
      latex_content += r"""
      \subsection*{""" + edu['degree'] + r""" in """ + edu['field_of_study'] + r"""}
      \begin{itemize}
          \item \textbf{School/University:} """ + edu['school'] + r"""
          \item \textbf{Start Date:} """ + str(edu['start_date']) + r"""
          \item \textbf{End Date:} """ + str(edu['end_date']) + r"""
      \end{itemize}
      """

  latex_content += r"""
  \section*{Work Experience}
  """
  for exp in work_experience:
      latex_content += r"""
      \subsection*{""" + exp['position'] + r""" at """ + exp['company'] + r"""}
      \begin{itemize}
          \item \textbf{Start Date:} """ + str(exp['start_date']) + r"""
          \item \textbf{End Date:} """ + str(exp['end_date']) + r"""
          \item \textbf{Description:} """ + exp['description'] + r"""
      \end{itemize}
      """

  latex_content += r"""
  \section*{Skills}
  """ + skills + r"""
  \end{document}
  """

  with open("cv.tex", "w") as f:
      f.write(latex_content)

  subprocess.run(["pdflatex", "cv.tex"])
  st.success("CV generated successfully! Check the file 'cv.pdf'.")

if __name__ == "__main__":
  main()

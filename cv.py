pip install pylatex
import streamlit as st
from pylatex import Document, Section, Subsection, Command, Package
from pylatex.utils import NoEscape
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
  doc = Document(documentclass='simplehipstercv', document_options='lighthipster')
  doc.packages.append(Package('inputenc', options='utf8'))
  doc.packages.append(Package('geometry', options='margin=1cm, a4paper'))
  doc.packages.append(Package('raleway', options='default'))

  doc.preamble.append(Command('title', 'New Simple CV'))
  doc.preamble.append(Command('author', 'LaTeX Ninja'))
  doc.preamble.append(Command('date', 'July 2024'))
  doc.append(NoEscape(r'\pagestyle{empty}'))
  doc.append(NoEscape(r'\begin{document}'))
  doc.append(NoEscape(r'\thispagestyle{empty}'))

  # Header
  doc.append(NoEscape(r'\simpleheader{headercolour}{' + name.split()[0] + r'}{' + name.split()[1] + r'}{Position}{white}'))

  # Personal Information
  with doc.create(Section('Personal Information', numbering=False)):
      doc.append(f"**Name:** {name}\n")
      doc.append(f"**Email:** {email}\n")
      doc.append(f"**Phone:** {phone}\n")
      doc.append(f"**Address:** {address}\n")

  # Education
  with doc.create(Section('Education', numbering=False)):
      for edu in education:
          with doc.create(Subsection(f"{edu['degree']} in {edu['field_of_study']}", numbering=False)):
              doc.append(f"**School/University:** {edu['school']}\n")
              doc.append(f"**Start Date:** {edu['start_date']}\n")
              doc.append(f"**End Date:** {edu['end_date']}\n")

  # Work Experience
  with doc.create(Section('Work Experience', numbering=False)):
      for exp in work_experience:
          with doc.create(Subsection(f"{exp['position']} at {exp['company']}", numbering=False)):
              doc.append(f"**Start Date:** {exp['start_date']}\n")
              doc.append(f"**End Date:** {exp['end_date']}\n")
              doc.append(f"**Description:** {exp['description']}\n")

  # Skills
  with doc.create(Section('Skills', numbering=False)):
      doc.append(skills)

  doc.append(NoEscape(r'\end{document}'))

  # Save the LaTeX document
  doc.generate_pdf('cv', clean_tex=False)
  st.success("CV generated successfully! Check the file 'cv.pdf'.")

if __name__ == "__main__":
  main()

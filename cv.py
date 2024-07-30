import streamlit as st
import pdfkit
from datetime import date
import os

def main():
    st.title("Comprehensive CV Builder")

    # Personal Information
    st.header("Personal Information")
    name = st.text_input("Full Name", key="name")
    email = st.text_input("Email", key="email")
    phone = st.text_input("Phone Number", key="phone")
    address = st.text_area("Address", key="address")
    profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"], key="profile_pic")

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

    # Projects
    st.header("Projects")
    projects = []
    num_projects = st.number_input("Number of Projects", min_value=1, max_value=10, step=1, key="num_projects")
    for i in range(num_projects):
        st.subheader(f"Project {i+1}")
        project_name = st.text_input(f"Project Name {i+1}", key=f"project_name_{i}")
        project_description = st.text_area(f"Project Description {i+1}", key=f"project_description_{i}")
        projects.append({
            "name": project_name,
            "description": project_description
        })

    # Certifications
    st.header("Certifications")
    certifications = []
    num_certifications = st.number_input("Number of Certifications", min_value=1, max_value=10, step=1, key="num_certifications")
    for i in range(num_certifications):
        st.subheader(f"Certification {i+1}")
        cert_name = st.text_input(f"Certification Name {i+1}", key=f"cert_name_{i}")
        cert_issuer = st.text_input(f"Issuer {i+1}", key=f"cert_issuer_{i}")
        cert_date = st.date_input(f"Date {i+1}", date.today(), key=f"cert_date_{i}")
        certifications.append({
            "name": cert_name,
            "issuer": cert_issuer,
            "date": cert_date
        })

    # Languages
    st.header("Languages")
    languages = []
    num_languages = st.number_input("Number of Languages", min_value=1, max_value=10, step=1, key="num_languages")
    for i in range(num_languages):
        st.subheader(f"Language {i+1}")
        language = st.text_input(f"Language {i+1}", key=f"language_{i}")
        proficiency = st.selectbox(f"Proficiency {i+1}", ["Beginner", "Intermediate", "Advanced", "Native"], key=f"proficiency_{i}")
        languages.append({
            "language": language,
            "proficiency": proficiency
        })

    # Skills
    st.header("Skills")
    skills = st.text_area("List your skills (separated by commas)", key="skills")

    # Generate CV
    if st.button("Generate CV", key="generate_cv"):
        generate_cv(name, email, phone, address, profile_pic, education, work_experience, projects, certifications, languages, skills)

def generate_cv(name, email, phone, address, profile_pic, education, work_experience, projects, certifications, languages, skills):
    html_content = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .section-title {{ font-size: 24px; font-weight: bold; margin-top: 20px; }}
        .subsection-title {{ font-size: 20px; font-weight: bold; margin-top: 10px; }}
        .item {{ margin-top: 10px; }}
        .item-title {{ font-weight: bold; }}
        .item-details {{ margin-left: 20px; }}
    </style>
    </head>
    <body>
    <h1>{name}</h1>
    <p><strong>Email:</strong> {email}</p>
    <p><strong>Phone:</strong> {phone}</p>
    <p><strong>Address:</strong> {address}</p>
    """

    html_content += "<div class='section-title'>Education</div>"
    for edu in education:
        html_content += f"""
        <div class='item'>
            <div class='item-title'>{edu['degree']} in {edu['field_of_study']}</div>
            <div class='item-details'>
                <p><strong>School/University:</strong> {edu['school']}</p>
                <p><strong>Start Date:</strong> {edu['start_date']}</p>
                <p><strong>End Date:</strong> {edu['end_date']}</p>
            </div>
        </div>
        """

    html_content += "<div class='section-title'>Work Experience</div>"
    for exp in work_experience:
        html_content += f"""
        <div class='item'>
            <div class='item-title'>{exp['position']} at {exp['company']}</div>
            <div class='item-details'>
                <p><strong>Start Date:</strong> {exp['start_date']}</p>
                <p><strong>End Date:</strong> {exp['end_date']}</p>
                <p><strong>Description:</strong> {exp['description']}</p>
            </div>
        </div>
        """

    html_content += "<div class='section-title'>Projects</div>"
    for proj in projects:
        html_content += f"""
        <div class='item'>
            <div class='item-title'>{proj['name']}</div>
            <div class='item-details'>
                {proj['description']}
            </div>
        </div>
        """

    html_content += "<div class='section-title'>Certifications</div>"
    for cert in certifications:
        html_content += f"""
        <div class='item'>
            <div class='item-title'>{cert['name']}</div>
            <div class='item-details'>
                <p><strong>Issuer:</strong> {cert['issuer']}</p>
                <p><strong>Date:</strong> {cert['date']}</p>
            </div>
        </div>
        """

    html_content += "<div class='section-title'>Languages</div>"
    for lang in languages:
        html_content += f"""
        <div class='item'>
            <div class='item-title'>{lang['language']} ({lang['proficiency']})</div>
        </div>
        """

    html_content += f"<div class='section-title'>Skills</div><div>{skills}</div>"

    html_content += "</body></html>"

    # Save HTML content to a file
    with open("cv.html", "w") as file:
        file.write(html_content)

    # Convert HTML to PDF
    pdfkit.from_file("cv.html", "cv.pdf")

    # Display success message and provide download link
    st.success("CV generated successfully!")
    st.download_button("Download CV", data=open("cv.pdf", "rb"), file_name="cv.pdf")

if __name__ == "__main__":
    main()

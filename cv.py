def main():
  st.title("Davetech CV Builder")

  # Personal Information
  st.header("Personal Information")
  name = st.text_input("Full Name")
  email = st.text_input("Email")
  phone = st.text_input("Phone Number")
  address = st.text_area("Address")

  # Education
  st.header("Education")
  education = []
  num_education = st.number_input("Number of Education Entries", min_value=1, max_value=10, step=1)
  for i in range(num_education):
      st.subheader(f"Education {i+1}")
      school = st.text_input(f"School/University {i+1}")
      degree = st.text_input(f"Degree {i+1}")
      field_of_study = st.text_input(f"Field of Study {i+1}")
      start_date = st.date_input(f"Start Date {i+1}", date(2000, 1, 1))
      end_date = st.date_input(f"End Date {i+1}", date(2004, 1, 1))
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
  num_experience = st.number_input("Number of Work Experience Entries", min_value=1, max_value=10, step=1)
  for i in range(num_experience):
      st.subheader(f"Work Experience {i+1}")
      company = st.text_input(f"Company {i+1}")
      position = st.text_input(f"Position {i+1}")
      start_date = st.date_input(f"Start Date {i+1}", date(2000, 1, 1))
      end_date = st.date_input(f"End Date {i+1}", date(2004, 1, 1))
      description = st.text_area(f"Description {i+1}")
      work_experience.append({
          "company": company,
          "position": position,
          "start_date": start_date,
          "end_date": end_date,
          "description": description
      })

  # Skills
  st.header("Skills")
  skills = st.text_area("List your skills (separated by commas)")

  # Generate CV
  if st.button("Generate CV"):
      generate_cv(name, email, phone, address, education, work_experience, skills)

def generate_cv(name, email, phone, address, education, work_experience, skills):
  st.header("Generated CV")
  st.subheader("Personal Information")
  st.write(f"**Name:** {name}")
  st.write(f"**Email:** {email}")
  st.write(f"**Phone:** {phone}")
  st.write(f"**Address:** {address}")

  st.subheader("Education")
  for edu in education:
      st.write(f"**School/University:** {edu['school']}")
      st.write(f"**Degree:** {edu['degree']}")
      st.write(f"**Field of Study:** {edu['field_of_study']}")
      st.write(f"**Start Date:** {edu['start_date']}")
      st.write(f"**End Date:** {edu['end_date']}")
      st.write("---")

  st.subheader("Work Experience")
  for exp in work_experience:
      st.write(f"**Company:** {exp['company']}")
      st.write(f"**Position:** {exp['position']}")
      st.write(f"**Start Date:** {exp['start_date']}")
      st.write(f"**End Date:** {exp['end_date']}")
      st.write(f"**Description:** {exp['description']}")
      st.write("---")

  st.subheader("Skills")
  st.write(skills)

if __name__ == "__main__":
  main()

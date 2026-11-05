# Hospital Appointment Scheduling using AI
# Made with Python, Streamlit and scikit-learn

import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

FILE = "appointments.csv"
COLUMNS = ["id", "name", "phone", "age", "department", "doctor",
           "date", "time", "symptoms", "risk", "status"]

# list of departments and their doctors
doctors = {
    "General Medicine": ["Dr. Arora", "Dr. Bansal"],
    "Cardiology": ["Dr. Sharma", "Dr. Gill"],
    "Dermatology": ["Dr. Kaur"],
    "Orthopedics": ["Dr. Singh", "Dr. Verma"],
    "Neurology": ["Dr. Mehta"],
    "ENT": ["Dr. Khanna"],
    "Pediatrics": ["Dr. Malhotra"],
}

# time slots of the day
times = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
         "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]

# small dataset for symptoms (symptom text, department)
symptom_data = [
    ("chest pain", "Cardiology"),
    ("heart beating fast", "Cardiology"),
    ("high blood pressure", "Cardiology"),
    ("breathing problem", "Cardiology"),
    ("pain in chest", "Cardiology"),
    ("chest tightness", "Cardiology"),
    ("left side chest pain", "Cardiology"),
    ("heart pain", "Cardiology"),
    ("skin rash", "Dermatology"),
    ("itching on skin", "Dermatology"),
    ("pimples and acne", "Dermatology"),
    ("hair fall", "Dermatology"),
    ("knee pain", "Orthopedics"),
    ("back pain", "Orthopedics"),
    ("bone fracture", "Orthopedics"),
    ("joint pain", "Orthopedics"),
    ("severe headache", "Neurology"),
    ("migraine", "Neurology"),
    ("numbness in hands", "Neurology"),
    ("memory loss", "Neurology"),
    ("fever", "General Medicine"),
    ("cold and cough", "General Medicine"),
    ("body weakness", "General Medicine"),
    ("stomach pain", "General Medicine"),
    ("pain in stomach", "General Medicine"),
    ("stomach ache and vomiting", "General Medicine"),
    ("loose motions", "General Medicine"),
    ("acidity and gas", "General Medicine"),
    ("ear pain", "ENT"),
    ("blocked nose", "ENT"),
    ("throat pain", "ENT"),
    ("hearing problem", "ENT"),
    ("pain in ear", "ENT"),
    ("ear infection", "ENT"),
    ("ear blocked", "ENT"),
    ("sore throat", "ENT"),
    ("child has fever", "Pediatrics"),
    ("baby not eating", "Pediatrics"),
    ("child cough", "Pediatrics"),
    ("baby vaccination", "Pediatrics"),
]


@st.cache_resource
def train_models():
    # Model 1: symptoms -> department (Naive Bayes)
    texts = [row[0] for row in symptom_data]
    labels = [row[1] for row in symptom_data]
    vectorizer = CountVectorizer()
    symptom_model = MultinomialNB()
    symptom_model.fit(vectorizer.fit_transform(texts), labels)

    # Model 2: no-show prediction (Random Forest)
    # real data was not available so I made random data
    np.random.seed(1)
    n = 3000
    age = np.random.randint(1, 90, n)
    days_before = np.random.randint(0, 60, n)
    sms = np.random.randint(0, 2, n)
    previous = np.random.poisson(0.4, n)
    # more days before, more previous no-shows -> more chance of no-show
    score = -2 + 0.04 * days_before + 0.7 * previous - 0.8 * sms
    chance = 1 / (1 + np.exp(-score))
    no_show = (np.random.rand(n) < chance).astype(int)

    X = np.column_stack([age, days_before, sms, previous])
    noshow_model = RandomForestClassifier(n_estimators=50, random_state=1)
    noshow_model.fit(X, no_show)
    return vectorizer, symptom_model, noshow_model


def load_data():
    if os.path.exists(FILE):
        return pd.read_csv(FILE, dtype={"phone": str})
    return pd.DataFrame(columns=COLUMNS)


def save_data(df):
    df.to_csv(FILE, index=False)


vectorizer, symptom_model, noshow_model = train_models()

# ---------------- Chatbot (rule based + AI department model) ----------------
def chatbot_reply(msg, df):
    text = msg.lower()
    words = text.split()

    # if user types a 10 digit phone number, show their appointments
    digits = "".join(ch for ch in msg if ch.isdigit())
    if len(digits) == 10:
        mine = df[df["phone"] == digits]
        if len(mine) == 0:
            return "I could not find any appointment for this phone number."
        reply = "Your appointments:\n"
        for i, row in mine.iterrows():
            reply += ("- #" + str(row["id"]) + ": " + row["doctor"] + " (" + row["department"] +
                      ") on " + str(row["date"]) + " at " + row["time"] + " - " + row["status"] + "\n")
        return reply

    # greetings
    if len(words) > 0 and words[0] in ["hi", "hello", "hey"]:
        return "Hello! I am the hospital assistant. You can ask me about doctors, timings, free slots, or tell me your symptoms."

    if "thank" in text:
        return "You are welcome! Take care."

    # free slots of a doctor (today)
    for dept in doctors:
        for doc in doctors[dept]:
            last_name = doc.lower().replace("dr. ", "")
            if last_name in text:
                booked = df[(df["doctor"] == doc) & (df["date"] == str(date.today())) &
                            (df["status"] == "Booked")]["time"].tolist()
                free = [t for t in times if t not in booked]
                if len(free) == 0:
                    return doc + " (" + dept + ") has no free slots today."
                return doc + " (" + dept + ") is free today at: " + ", ".join(free)

    if "doctor" in text or "department" in text:
        if "symptom" not in text:
            reply = "Our departments and doctors:\n"
            for dept in doctors:
                reply += "- " + dept + ": " + ", ".join(doctors[dept]) + "\n"
            return reply

    if "time" in text or "timing" in text or "open" in text or "hours" in text:
        return "Appointments are available from 9:00 AM to 12:30 PM and 2:00 PM to 4:30 PM."

    if "cancel" in text or "reschedule" in text:
        return "To cancel an appointment, open 'Update Appointment' from the menu on the left."

    if "book" in text or "appointment" in text:
        return "To book an appointment, open 'Book Appointment' from the menu on the left."

    # use the AI model to suggest a department from the symptoms
    probs = symptom_model.predict_proba(vectorizer.transform([msg]))[0]
    if probs.max() > 0.35:
        dept = symptom_model.classes_[probs.argmax()]
        return ("Based on your symptoms, I suggest the **" + str(dept) + "** department. " +
                "This is only a suggestion, not a diagnosis. You can book from the 'Book Appointment' page.")

    return ("Sorry, I did not understand. You can ask about doctors, timings, free slots of a doctor "
            "(example: 'slots for Dr. Sharma'), or tell me your symptoms. "
            "Type your phone number to see your appointments.")


st.title("Hospital Appointment Scheduling")
st.write("Book appointments easily. AI suggests the department and predicts no-show risk.")

page = st.sidebar.radio("Menu", ["Book Appointment", "View Appointments",
                                 "Update Appointment", "Statistics", "Chatbot"])
df = load_data()

# ---------------- Page 1: Book ----------------
if page == "Book Appointment":
    st.header("Book an Appointment")

    name = st.text_input("Patient name")
    phone = st.text_input("Phone number")
    age = st.number_input("Age", 1, 100, 25)
    sms = st.checkbox("Send SMS reminder", value=True)
    symptoms = st.text_area("Describe your symptoms")

    # AI part 1: suggest department
    suggested = "General Medicine"
    if symptoms.strip() != "":
        suggested = str(symptom_model.predict(vectorizer.transform([symptoms]))[0])
        st.info("AI suggested department: " + suggested)
        st.caption("This is only a suggestion, not a medical diagnosis.")

    dept_list = list(doctors.keys())
    department = st.selectbox("Department", dept_list, index=dept_list.index(suggested))
    doctor = st.selectbox("Doctor", doctors[department])
    day = st.date_input("Date", min_value=date.today())

    # find free slots (remove slots that are already booked)
    booked = df[(df["doctor"] == doctor) & (df["date"] == str(day)) &
                (df["status"] == "Booked")]["time"].tolist()
    free_slots = [t for t in times if t not in booked]

    if len(free_slots) == 0:
        st.warning("No slots free on this day. Please choose another date.")
    else:
        time = st.selectbox("Time slot", free_slots)

        # previous no-shows of this patient
        previous = len(df[(df["phone"] == phone) & (df["status"] == "No-show")])
        days_before = (day - date.today()).days

        # AI part 2: predict no-show
        features = [[age, days_before, int(sms), previous]]
        risk = noshow_model.predict_proba(features)[0][1]
        st.write("Predicted no-show risk: **" + str(round(risk * 100)) + "%**")
        if risk > 0.5:
            st.warning("High risk. Hospital may call this patient to confirm.")

        if st.button("Book Appointment"):
            if name == "" or phone == "":
                st.error("Please enter name and phone number.")
            else:
                new_row = {
                    "id": len(df) + 1, "name": name, "phone": phone, "age": age,
                    "department": department, "doctor": doctor, "date": str(day),
                    "time": time, "symptoms": symptoms, "risk": round(risk, 2),
                    "status": "Booked",
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(df)
                st.success("Appointment booked with " + doctor + " on " +
                           str(day) + " at " + time)

# ---------------- Page 2: View ----------------
elif page == "View Appointments":
    st.header("All Appointments")
    if len(df) == 0:
        st.write("No appointments yet.")
    else:
        dept_filter = st.selectbox("Filter by department", ["All"] + list(doctors.keys()))
        if dept_filter != "All":
            st.dataframe(df[df["department"] == dept_filter])
        else:
            st.dataframe(df)

# ---------------- Page 3: Update ----------------
elif page == "Update Appointment":
    st.header("Update Appointment")
    if len(df) == 0:
        st.write("No appointments yet.")
    else:
        appt_id = st.selectbox("Appointment ID", df["id"].tolist())
        new_status = st.selectbox("New status", ["Cancelled", "Completed", "No-show"])
        if st.button("Update"):
            df.loc[df["id"] == appt_id, "status"] = new_status
            save_data(df)
            st.success("Appointment " + str(appt_id) + " updated to " + new_status)

# ---------------- Page 4: Statistics ----------------
elif page == "Statistics":
    st.header("Statistics")
    if len(df) == 0:
        st.write("No data yet.")
    else:
        st.write("Total appointments:", len(df))
        st.write("Average no-show risk:", round(df["risk"].mean() * 100), "%")
        st.subheader("Appointments per department")
        st.bar_chart(df["department"].value_counts())
        st.subheader("Appointments by status")
        st.bar_chart(df["status"].value_counts())

# ---------------- Page 5: Chatbot ----------------
elif page == "Chatbot":
    st.header("Hospital Chatbot")
    st.write("Ask about doctors, timings, free slots, or tell me your symptoms.")

    # keep chat history in session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # show old messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Type your message here")
    if user_msg:
        reply = chatbot_reply(user_msg, df)
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.messages.append({"role": "assistant", "content": reply})

🏠 Airbnb Occupancy Rate Predictor:
An interactive web app to model and predict Airbnb occupancy rates using real NYC Airbnb data.
Built with Streamlit, it lets you explore model performance and make custom predictions.

🚀 Features:
*Choose between Linear Regression and Decision Tree

*Visualize actual vs predicted occupancy rates

*Predict occupancy for custom listing inputs

*NYC-inspired clean design

🛠 Technologies:
.Python 3
.Streamlit
.scikit-learn
.pandas
.matplotlib
.NumPy

💻 How to Run:
1️⃣ Clone the repository:
git clone https://github.com/gayathri945/airbnb-occupancy-predictor.git
cd airbnb-occupancy-predictor

2️⃣ (Optional) Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate  # For Windows
#or
source venv/bin/activate  # For Mac/Linux

3️⃣ Install dependencies:
pip install -r requirements.txt

4️⃣ Run the app:
streamlit run app.py

📌 Notes:
Uses NYC Airbnb data (listings.csv) with engineered occupancy rate:
occupancy_rate = (number_of_reviews / availability_365) * 100
Occupancy rate is capped at 100%.

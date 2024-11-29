import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv('Maternal Health Risk Data Set (1).csv')

# Mapping the risk levels
risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df["RiskLevel"] = df["RiskLevel"].map(risk_mapping)

# Apply log transformation to the numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.difference(['RiskLevel'])
df[numeric_columns] = df[numeric_columns].apply(lambda x: np.log1p(x))

X = df.drop(columns=["RiskLevel"])
y = df["RiskLevel"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Initialize and train models
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()

# Fit the models
clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        systolic_bp = float(request.form['systolicBP'])
        diastolic_bp = float(request.form['diastolicBP'])
        bs = float(request.form['bs'])
        body_temp = float(request.form['bodyTemp'])
        heart_rate = float(request.form['heartRate'])

        # Transform user input (log transformation)
        input_data = np.log1p([age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]).reshape(1, -1)

        # Predict the risk level using the best model (Random Forest or XGBoost)
        risk_level = rf.predict(input_data)[0]

        # Map the risk level back to string labels
        risk_levels = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}
        risk = risk_levels[risk_level]

        # Provide a detailed recommendation based on the risk level
        if risk == 'low risk':
            prescription = """
            <h4>Dietary Advice:</h4>
            <ul>
                <li>Continue eating a variety of vegetables, lean proteins, and whole grains. Aim for a well-balanced meal plan that includes fruits, veggies, and healthy fats.</li>
            </ul>
            <h4>Physical Activity Recommendations:</h4>
            <ul>
                <li>Maintain your fitness level with regular activity. Aim for 30 minutes of moderate exercise, such as walking, cycling, or swimming, at least 5 days a week.</li>
            </ul>
            <h4>Stress Management:</h4>
            <ul>
                <li>Practice mindfulness or meditation to maintain emotional balance. Ensure adequate sleep and focus on positive lifestyle habits that reduce stress.</li>
            </ul>
            <h4>Medical Follow-Up:</h4>
            <ul>
                <li>Keep your scheduled check-ups and screenings, such as blood pressure, cholesterol, and glucose tests, to ensure early detection of any potential risks.</li>
            </ul>
            """
        elif risk == 'mid risk':
            prescription = """
            <h4>Dietary Advice:</h4>
            <ul>
                <li>Try to cut down on processed foods, limit sugar intake, and aim for a Mediterranean diet rich in antioxidants. Consult a nutritionist for more personalized advice.</li>
            </ul>
            <h4>Physical Activity Recommendations:</h4>
            <ul>
                <li>Incorporate moderate to high-intensity exercises to improve cardiovascular health. Yoga, strength training, or high-intensity interval training (HIIT) could be beneficial.</li>
            </ul>
            <h4>Stress Management:</h4>
            <ul>
                <li>Try yoga, breathing exercises, or regular mental relaxation techniques to reduce anxiety or stress. Psychological counseling might also help if you feel overwhelmed.</li>
            </ul>
            <h4>Medical Follow-Up:</h4>
            <ul>
                <li>You may need more frequent check-ups. Monitor your health closely with regular testing to track your progress and avoid any further complications.</li>
            </ul>
            """
        else:
            prescription = """
            <h4>Dietary Advice:</h4>
            <ul>
                <li>Consult with a dietitian for a strict meal plan that helps control risk factors. Consider reducing salt, sugar, and fat intake. A low-carb or DASH diet might be recommended.</li>
            </ul>
            <h4>Physical Activity Recommendations:</h4>
            <ul>
                <li>If your health condition limits physical activity, discuss alternatives like physiotherapy or guided physical therapy exercises with your doctor.</li>
            </ul>
            <h4>Stress Management:</h4>
            <ul>
                <li>Consider professional help, such as therapy or counseling, especially if stress is affecting your overall health. Managing stress can improve your health outcomes significantly.</li>
            </ul>
            <h4>Medical Follow-Up:</h4>
            <ul>
                <li>Frequent medical assessments are essential. Consult your healthcare provider immediately for tailored advice, and make sure to follow through with recommended tests or procedures.</li>
            </ul>
            """
        
        return render_template('index.html', risk=risk, prescription=prescription)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

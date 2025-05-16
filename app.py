import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify
from config import GEMINI_API_KEY
from sklearn.linear_model import LinearRegression
import numpy as np
import random

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # Auto-reload templates

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Example historical data (for demo purposes)
historical_campaign_data = {
    'budget': [100, 200, 300, 400, 500],
    'clicks': [20, 40, 55, 65, 80],
    'conversions': [2, 8, 10, 13, 16],
    'impressions': [2000, 4000, 6000, 8000, 10000]
}

# Machine Learning Model for Budget Optimization (linear regression as an example)
def budget_optimization_model():
    model = LinearRegression()
    X = np.array(historical_campaign_data['budget']).reshape(-1, 1)
    y = np.array(historical_campaign_data['conversions'])
    model.fit(X, y)
    return model

# Campaign Optimization: Provide budget optimization prediction
def optimize_budget(budget):
    model = budget_optimization_model()
    predicted_conversions = model.predict(np.array([[budget]]))
    return predicted_conversions[0]

# AI Recommendations for Campaign Enhancement
def generate_campaign_recommendations(budget, clicks, conversions):
    # AI recommendation based on campaign performance
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Given the current campaign data:
        - Budget: ₹{budget}
        - Clicks: {clicks}
        - Conversions: {conversions}

        Suggest recommendations to optimize the budget allocation, increase conversions, and improve overall campaign performance. 
        The recommendations should include:
        - Suggestions for budget allocation
        - Recommendations for optimizing ad creatives or targeting
        - Insights on improving CTR (Click-Through Rate)
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating recommendations: {e}"

# Generate Google Ads with AI
def generate_google_ads(product_info, input_name):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Generate high-converting Google Ads copy for the following product:
        Product Name: {input_name}
        Product Details: {product_info}

        Create variations focusing on:
        - Urgency (Limited-time offers, Act now, Hurry up!)
        - Special Offers (Discounts, Free shipping, Buy one get one free, etc.)
        - CTR Improvement (Engaging, compelling, and action-driven language)

        Provide exactly:
        Headlines:
        1. "First engaging headline with urgency"
        2. "Second headline emphasizing an offer"
        3. "Third headline optimized for CTR"
        
        Descriptions:
        1. "First compelling description with urgency"
        2. "Second persuasive description highlighting an offer"
        """
        
        response = model.generate_content(prompt)
        
        text = response.text.split("\n")
        
        headlines = []
        descriptions = []
        is_headline_section = False
        is_description_section = False

        for line in text:
            line = line.strip()
            if "Headlines:" in line:
                is_headline_section = True
                is_description_section = False
                continue
            elif "Descriptions:" in line:
                is_description_section = True
                is_headline_section = False
                continue

            if is_headline_section and line.startswith(("1.", "2.", "3.")):
                headlines.append(line[3:].strip().strip('"'))
            elif is_description_section and line.startswith(("1.", "2.")):
                descriptions.append(line[3:].strip().strip('"'))

        return {"headlines": headlines, "descriptions": descriptions}
    
    except Exception as e:
        return {"error": str(e)}

# Generate Google Ads A/B Test Results
def generate_ab_test(product_info, input_name):
    ad_copy_a = generate_google_ads(product_info, input_name)
    ad_copy_b = generate_google_ads(product_info, input_name)

    return {"ad_copy_a": ad_copy_a, "ad_copy_b": ad_copy_b}

# Generate Campaign Metrics Visualization (including additional data)
def generate_metrics_plot():
    ctr = [random.uniform(0, 0.2) for _ in range(5)]  # CTR values
    conversions = [random.randint(1, 15) for _ in range(5)]  # Conversion values
    budgets = [100, 200, 300, 400, 500]
    
    plt.figure(figsize=(8, 6))
    plt.plot(budgets, ctr, label="CTR", marker='o', linestyle='--', color='b')
    plt.plot(budgets, conversions, label="Conversions", marker='s', linestyle='-', color='g')
    plt.xlabel('Budget')
    plt.ylabel('Value')
    plt.title('Campaign Metrics Analysis (CTR & Conversions)')
    plt.legend()
    
    # Save to a byte stream
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return plot_url

@app.route("/", methods=["GET", "POST"])
def home():
    ad_suggestions = None
    ab_test_results = None
    campaign_recommendations = None
    metrics_plot_url = generate_metrics_plot()  # Generate the metrics plot
    
    if request.method == "POST":
        product_info = request.form["product_info"]
        input_name = request.form.get("input_name", "Product")
        ad_suggestions = generate_google_ads(product_info, input_name)
        ab_test_results = generate_ab_test(product_info, input_name)
        
        # AI Recommendations based on the campaign data
        budget = float(request.form["budget"])
        clicks = int(request.form["clicks"])
        conversions = int(request.form["conversions"])
        campaign_recommendations = generate_campaign_recommendations(budget, clicks, conversions)
    
    return render_template("index.html", 
                           product_info=None, 
                           ad_suggestions=ad_suggestions, 
                           metrics_plot_url=metrics_plot_url, 
                           ab_test_results=ab_test_results,
                           campaign_recommendations=campaign_recommendations)

@app.route("/optimize_budget", methods=["POST"])
def optimize_ad_budget():
    try:
        budget = float(request.form["budget"])
        predicted_conversions = optimize_budget(budget)
        return jsonify({"predicted_conversions": predicted_conversions})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

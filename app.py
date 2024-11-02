from flask import Flask, render_template, request, url_for, session, redirect
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# Function to generate and save plots
def generate_plots(N, mu, sigma2, S):
    X = np.random.rand(N)
    Y = mu + np.sqrt(sigma2) * np.random.randn(N)

    X = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted Line')
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plot1_path = f"static/plot1_{len(session['results'])}.png"
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = mu + np.sqrt(sigma2) * np.random.randn(N)
        X_sim = X_sim.reshape(-1, 1)
        sim_model = LinearRegression()
        sim_model.fit(X_sim, Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = f"static/plot2_{len(session['results'])}.png"
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme, slope, intercept

@app.route("/", methods=["GET", "POST"])
def index():
    if 'results' not in session:
        session['results'] = []

    if request.method == "POST":
        if request.form.get("clear"):
            session.pop('results', None)
            return redirect(url_for('index'))

        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme, slope, intercept = generate_plots(N, mu, sigma2, S)

        # Save results in session
        session['results'].append({
            'N': N, 'mu': mu, 'sigma2': sigma2, 'S': S,
            'plot1': plot1, 'plot2': plot2,
            'slope': slope, 'intercept': intercept,
            'slope_extreme': slope_extreme, 'intercept_extreme': intercept_extreme
        })
        session.modified = True

        return redirect(url_for('index'))

    return render_template("index.html", results=session.get('results', []))

if __name__ == "__main__":
    app.run(debug=True)

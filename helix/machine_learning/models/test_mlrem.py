# MLREM Testing and evaluation
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100) * 0.5  # y = 4 + 3x + noise
    
    # Train model
    model = EMLinearRegression(alpha=0.1, beta=1.0)
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Compute metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Plot results
    plt.scatter(y, y_pred, alpha=0.7, label="Predicted vs Measured")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    plt.title(f"EM Linear Regression\nR²: {r2:.3f}, MSE: {mse:.3f}")
    plt.legend()
    plt.show()

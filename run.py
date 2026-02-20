from mnist.trainer import trainer
from mnist.visualization import plot_curves, plot_confusionmatrix

load_model = True
model, history, test_data = trainer(load_model)


# Plots 
plot_curves(history)
plot_confusionmatrix(test_data["labels"], test_data["preds"])

# PokemonAI

### Project Overview

This project aims to classify Pokémon as either Mega Evolution or Regular Pokémon using machine learning. The classification is performed based on numerical attributes such as HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, and Total Stats, without relying on the Pokémon name.

### Dataset

The dataset is obtained from Kaggle and contains information about various Pokémon, including their stats, types, and legendary status.

### Objective

- Use machine learning to classify Pokémon as Mega Evolution or Regular.

- Do not use the Pokémon name as a feature.

- Train models using numerical attributes only.

- Evaluate the models using:

   - Confusion Matrix

   - ROC Curve

   - Precision-Recall Curve

- Save the final predictions in a CSV file with the following columns:

- Pokemon: Name of the Pokémon (only for output identification).

- Mega_Evolution: "Yes" if the Pokémon is a Mega Evolution, "No" otherwise.

### Results and Conclusion

- The model successfully classifies Pokémon based on numerical stats.

- The best-performing model is selected based on the highest AUC score.

- The classification accuracy is evaluated using Confusion Matrix, ROC Curve, and Precision-Recall Curve.



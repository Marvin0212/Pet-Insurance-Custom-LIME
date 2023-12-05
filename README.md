# Pet-Insurance-Custom-LIME
Custom LIME for Pet Insurance: Tailored AI interpretability framework addressing multicollinearity and feature interdependencies in insurance data, enhancing model transparency and decision-making insights.
Certainly! Below is a concise yet comprehensive project description for your GitHub repository, based on the detailed overview of your work:

## Custom LIME: Enhancing Model Interpretability for Insurance Data

**Project Description:**

This repository hosts the Custom LIME architecture, a novel adaptation of the Local Interpretable Model-agnostic Explanations (LIME) framework, specifically tailored for enhancing interpretability in insurance datasets, with a focus on pet insurance data. This work stems from the recognition that conventional LIME's assumptions about feature independence and data structure are not always aligned with the complexities of real-world datasets, particularly those in the insurance domain.

### Key Modifications and Features:

1. **Authentic Data Generation Strategy**: To address LIME’s limitation regarding feature independence, we introduce a modified data perturbation method. This method respects statistical dependencies among features, ensuring that synthetic data samples are representative of the real-world structure of the pet insurance dataset.

2. **Customized Neighborhood Definition and Distance Metrics**: Recognizing the diversity of the dataset, we have developed unique distance metrics for different data types. These metrics accurately capture the intricate relationships within the data, enabling the formation of a more representative data neighborhood for generating meaningful model explanations.

3. **Feature Selection Aligned with the Model**: We implement a feature selection strategy that aligns closely with the underlying LightGBM model used in our case study. This approach ensures that the selected features are not only significant for the model’s operations but also contribute to clear and concise explanations.

4. **Integration of Feature Interactions**: Our custom LIME framework considers the interdependence among dataset features, particularly evident in canine comorbidity research. By incorporating feature interactions, our model reflects these dependencies, offering more nuanced and accurate explanations.

### Project Overview:

This repository hosts the Custom LIME architecture – essentially, the classic LIME library enhanced with custom modifications tailored for the complexities of insurance data. The main changes are concentrated in the following files: `lime_base.py`, `lime_tabular.py`, `pet_feature_selection.py`, and `pet_generation.py`. These files embody the core of our adjustments, making the framework suitable for the intricate nature of pet insurance data.

This project serves as a vital case study in adapting explainable AI (XAI) methods to complex datasets. The enhancements to the classical LIME framework address challenges like multicollinearity and inter-feature dependencies that are prevalent in insurance data.

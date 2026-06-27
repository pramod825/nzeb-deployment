#!/bin/bash
echo "Downloading RF model..."
pip install gdown
python -c "
import gdown, os
os.makedirs('models', exist_ok=True)
gdown.download('https://drive.google.com/uc?id=1kfbtbv2VVou-XjOq7bVq8-GJRVS_BmEF', 'models/rf_model.pkl', quiet=False)
print('RF model downloaded!')
"
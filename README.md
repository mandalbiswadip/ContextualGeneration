# ContextualGeneration

To train the paragraph generation / contextual generation model, use the following command:

```python LED_paragraph_gen.py --repfile allenai/led-base-16384 --train_dataset <training data path> --distant_dataset <distant data path> --dev_dataset <dev data path>--checkpoint <model_save_path>```

The model can be run in a standard Tesla V100s-PCIE-32GB GPU. 

The evaluation pipeline can be found at:

```notebooks/evaluate_LED-para_generation.ipynb```

Analysis for comparing the baseline and the infilling approach can be found here. 

```notebooks/Analyze-paragraph-generation.ipynb```
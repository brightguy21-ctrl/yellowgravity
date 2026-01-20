# predict.py
from cog import BasePredictor, Input, Path, Concatenate
from typing import Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from PIL import Image
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# YOUR HF TOKEN HERE (required for gated MedGemma)
HF_TOKEN = "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"  # ← PASTE YOUR READ TOKEN HERE
login(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)
# -------------------------------

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load MedGemma once at startup."""
        self.model_id = "google/medgemma-1.5-4b-it"
        
        # Use 4-bit quantization to save memory on Replicate GPU
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("MedGemma loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Upload a clear photo of the baby's eyes or face"),
        age_hours: int = Input(description="Baby's age in hours", default=24),
        gestation_weeks: int = Input(description="Gestational age in weeks", default=39),
        birth_weight_kg: float = Input(description="Birth weight in kg", default=3.2),
        clinical_note: str = Input(description="Any clinical notes", default="Feeding well, alert")
    ) -> str:
        """Main prediction function — called by Replicate."""
        # Open the uploaded image
        pil_image = Image.open(image).convert("RGB")

        # Build multimodal prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": f"Analyze this newborn's photo for neonatal jaundice. Age: {age_hours} hours, Gestation: {gestation_weeks} weeks, Weight: {birth_weight_kg} kg. Note: '{clinical_note}'. Focus on sclera (whites of eyes) and skin yellowing. Estimate severity: none/low, mild, moderate, severe. Compare to similar babies. Give advice and strong disclaimer."}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=300)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Clean output
        if "assistant" in response:
            response = response.split("assistant", 1)[1].strip()

        # Simple risk zone (based on response keywords)
        zone = "Unknown"
        if "severe" in response.lower() or "high" in response.lower():
            zone = "Danger Zone"
        elif "moderate" in response.lower() or "medium" in response.lower():
            zone = "Mild/Monitor Zone"
        else:
            zone = "Safe/Low Risk Zone"

        full_report = f"**Yellow Gravity Report**\n\n**Zone**: {zone}\n\n{response}\n\n**Disclaimer**: This is AI screening only — not a medical diagnosis. Always consult a doctor for proper bilirubin testing."

        return full_report

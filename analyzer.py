from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"

print("Loading HealthSignal AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("HealthSignal AI model loaded successfully!")


def get_value(case_data, *keys, default=""):
    """
    Gets a value from case_data using multiple possible key names.
    """

    for key in keys:
        if key in case_data:
            return case_data[key]

    return default


def detect_indicators(case_data):
    """
    Detects awareness-oriented lifestyle indicators.

    Important:
    These rules are illustrative and are not medical diagnostic criteria.
    """

    indicators = []

    sleep_value = get_value(
        case_data,
        "sleep",
        "Sleep hours per day",
        default="0"
    )

    exercise_value = get_value(
        case_data,
        "exercise",
        "Exercise minutes per week",
        default="0"
    )

    water_value = get_value(
        case_data,
        "water",
        "Water intake per day",
        default="0"
    )

    stress = get_value(
        case_data,
        "stress",
        "Stress level"
    )

    smoking = get_value(
        case_data,
        "smoking",
        "Smoking status"
    )

    diet = get_value(
        case_data,
        "diet",
        "Fruit and vegetable intake"
    )

    fatigue = get_value(
        case_data,
        "fatigue",
        "Persistent fatigue"
    )

    try:
        sleep = float(sleep_value or 0)
    except (ValueError, TypeError):
        sleep = 0

    try:
        exercise = float(exercise_value or 0)
    except (ValueError, TypeError):
        exercise = 0

    try:
        water = float(water_value or 0)
    except (ValueError, TypeError):
        water = 0

    # Signal 1: Sleep
    if sleep > 0 and sleep < 8:
        indicators.append({
            "name": "Low sleep duration",
            "reason": (
                "The reported sleep duration is below a commonly used "
                "general reference for many adults. Sleep habits may "
                "deserve further attention."
            ),
            "level": "Attention"
        })

    # Signal 2: Physical activity
    if exercise < 150:
        indicators.append({
            "name": "Low physical activity",
            "reason": (
                "The reported weekly activity is below a commonly used "
                "general activity reference. Gradual activity improvement "
                "may be worth discussing."
            ),
            "level": "Attention"
        })

    # Signal 3: Stress
    if str(stress).lower() == "high":
        indicators.append({
            "name": "High stress level",
            "reason": (
                "The reported stress level may affect daily well-being, "
                "sleep, concentration, and coping. Further discussion "
                "may be helpful."
            ),
            "level": "Attention"
        })

    # Signal 4: Water Intake
    if water >= 5:
        indicators.append({
            "name": "High reported water intake",
            "reason": (
                "The reported water intake is relatively high for many people. "
                "Water needs vary depending on weather, physical activity, diet, "
                "and health conditions. If this intake is associated with unusual "
                "thirst or frequent urination, professional discussion may be helpful."
            ),
            "level": "Attention"
        })

    # Signal 5: Smoking
    if str(smoking).lower() == "yes":
        indicators.append({
            "name": "Current smoking",
            "reason": (
                "Current smoking is a lifestyle factor that may deserve "
                "professional health discussion and support."
            ),
            "level": "Attention"
        })

    # Signal 6: Diet
    if str(diet).lower() == "low":
        indicators.append({
            "name": "Low fruit and vegetable intake",
            "reason": (
                "The reported intake may benefit from reviewing overall "
                "food variety and nutritional balance."
            ),
            "level": "Attention"
        })

    # Signal 7: Fatigue
    if str(fatigue).lower() == "yes":
        indicators.append({
            "name": "Persistent fatigue",
            "reason": (
                "Persistent fatigue can have many possible causes. If it "
                "continues or affects daily activities, professional "
                "discussion may be appropriate."
            ),
            "level": "Attention"
        })

    return indicators


def build_prompt(case_data, indicators):
    """
    Creates a controlled prompt for the local language model.
    """

    case_text = "\n".join(
        f"{key}: {value}"
        for key, value in case_data.items()
    )

    if indicators:
        indicator_text = "\n".join(
            f"- {item['name']}: {item['reason']}"
            for item in indicators
        )
    else:
        indicator_text = (
            "- No attention-worthy signals were identified "
            "by the prototype's illustrative rules."
        )

    prompt = f"""
You are HealthSignal AI.

Create a responsible health and lifestyle awareness summary
for a fictional case.

Fictional case:
{case_text}

Detected awareness signals:
{indicator_text}

Write a clear report using these exact headings:

Overall summary:
Attention-worthy indicators:
Why these indicators may matter:
Uncertainty and limitations:
Responsible next steps:
Safety disclaimer:

Important instructions:
- Do not diagnose diseases.
- Do not predict disease probability.
- Do not say the person has a medical condition.
- Do not invent symptoms or test results.
- Explain that these are awareness signals only.
- Mention that the rules are illustrative and not medically validated.
- Use calm and simple language.
- Recommend professional discussion for persistent or concerning issues.
- Do not repeat these instructions in your answer.
- Do not write the word "Hypothesis".
- Do not refer to yourself as a prompt.
"""

    return prompt


def generate_explanation(prompt):
    """
    Generates the local AI explanation.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result.strip()


def create_fallback_report(indicators):
    """
    Safe fallback report if the local model produces an incomplete answer.
    """

    if indicators:
        indicator_names = "\n".join(
            f"- {item['name']}"
            for item in indicators
        )

        reasons = "\n".join(
            f"- {item['name']}: {item['reason']}"
            for item in indicators
        )
    else:
        indicator_names = (
            "No attention-worthy signals were identified "
            "by the prototype's illustrative rules."
        )

        reasons = (
            "No specific signals were highlighted from the provided "
            "fictional inputs."
        )

    return f"""
Overall summary:

This fictional case contains some lifestyle-related factors that may
deserve awareness and further discussion. The result is not a diagnosis.

Attention-worthy indicators:

{indicator_names}

Why these indicators may matter:

{reasons}

Uncertainty and limitations:

This is a fictional demonstration. The rules used by this prototype
are illustrative and are not medically validated diagnostic criteria.
The result does not confirm or rule out any disease. Missing information
may change the interpretation.

Responsible next steps:

Review lifestyle habits gradually, track changes over time, and discuss
persistent or concerning issues with a qualified healthcare professional.

Safety disclaimer:

HealthSignal AI provides awareness-oriented information only. It does
not diagnose diseases, calculate medical risk scores, or replace
professional medical advice.
""".strip()


def analyze_case(case_data):
    """
    Main analysis function used by Flask.
    """

    indicators = detect_indicators(case_data)

    prompt = build_prompt(
        case_data,
        indicators
    )

    try:
        ai_explanation = generate_explanation(prompt)

        # Use fallback if model output is too short or suspicious.
        suspicious_words = [
            "hypothesis:",
            "you are healthsignal",
            "repeat these instructions",
            "medical condition."
        ]

        is_suspicious = (
            len(ai_explanation) < 100
            or any(
                word in ai_explanation.lower()
                for word in suspicious_words
            )
        )

        if is_suspicious:
            ai_explanation = create_fallback_report(indicators)

    except Exception as error:
        print("AI generation error:", error)
        ai_explanation = create_fallback_report(indicators)

    return {
        "case_data": case_data,
        "indicators": indicators,
        "explanation": ai_explanation
    }
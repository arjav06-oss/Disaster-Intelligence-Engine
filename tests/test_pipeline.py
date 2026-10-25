"""
test_pipeline.py - End-to-End Pipeline Test
=============================================
Creates synthetic disaster text data that mimics CrisisMMD format,
then runs the FULL pipeline: load → preprocess → train → evaluate → predict.

This validates that ALL code works before the real dataset is downloaded.

WHY SYNTHETIC DATA?
-------------------
The real CrisisMMD dataset must be downloaded manually from QCRI's website.
This test creates realistic fake data so you can verify the entire pipeline
works correctly on your machine without needing the actual dataset.

The synthetic data contains:
  - Informative examples: actual disaster reports (damage, casualties, rescues)
  - Not informative examples: opinions, prayers, casual comments

Usage:
    python tests/test_pipeline.py
"""

import os
import sys
import csv
import random

# Add project root to path so Python can find the 'src' package
# __file__ is this script's path → go up one level (..) to reach project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_synthetic_dataset():
    """
    Create a synthetic TSV file mimicking CrisisMMD format.
    Uses realistic disaster-related text examples.
    
    The file is saved to data/raw/test_synthetic_data.tsv with columns:
      - event_name: simulated disaster event (e.g., 'hurricane_harvey')
      - tweet_id: unique identifier for each text
      - tweet_text: the actual text content
      - label: 'informative' or 'not_informative'
    """
    print("📝 Creating synthetic test dataset...")
    
    # ── INFORMATIVE EXAMPLES ──
    # These texts contain ACTIONABLE disaster information:
    # specific damage reports, casualty numbers, locations, rescue operations,
    # evacuation orders, relief efforts — the kind of information emergency
    # responders actually need to act on.
    informative_texts = [
        "Multiple buildings collapsed in downtown area after earthquake magnitude 7.2",
        "Flood waters rising rapidly. Over 200 families evacuated from riverside neighborhood",
        "Emergency shelters set up at Lincoln High School. Red Cross providing food and water",
        "Highway 101 completely blocked by mudslide. Avoid the area. Use alternate routes",
        "Fire department reports 3 casualties and 15 injured in building collapse",
        "Water supply contaminated after flooding. Boil water advisory issued for zones 3 through 7",
        "Search and rescue teams deployed to earthquake affected region. 50 people still missing",
        "Hurricane Category 4 making landfall. Wind speeds exceeding 150 mph along coast",
        "Power outages affecting 50000 homes across the county after the storm",
        "Bridges damaged by floodwaters. Do not attempt to cross any waterways",
        "FEMA opens disaster relief centers at three locations in the affected county",
        "Hospital overwhelmed with injured victims. Requesting additional medical teams urgently",
        "National Guard deployed to assist with evacuation efforts in coastal communities",
        "Gas leak reported near elementary school after earthquake. Area being evacuated",
        "Tornado warning issued for the entire county. Seek shelter immediately",
        "Road closures on Interstate 95 due to severe flooding from hurricane",
        "Rescue helicopter deployed to save stranded residents from rooftops in flood zone",
        "Emergency declaration issued by governor for five counties affected by wildfire",
        "Structural damage assessment teams being sent to affected neighborhoods tomorrow morning",
        "Drinking water distribution centers open at city hall and community center from 8am",
        "Over 100 aftershocks recorded since the main earthquake struck last night",
        "Tsunami advisory issued for coastal regions following the undersea earthquake",
        "Fire crews battling massive wildfire that has burned over 5000 acres since yesterday",
        "Volunteers needed at the food bank to help distribute supplies to displaced families",
        "Medical camp set up near disaster zone providing free treatment to affected people",
        "Electricity restored to 30 percent of affected areas after repair crews worked overnight",
        "School closures announced for the entire district due to structural safety concerns",
        "Emergency blood drive organized at community hospital for earthquake victims",
        "Damage to sewage system causing health hazard in flooded neighborhoods",
        "Coast guard rescued 45 people from capsized boats during the tropical storm",
        "International aid organizations arriving with supplies for the disaster affected region",
        "Evacuation orders issued for low-lying areas as dam shows signs of structural weakness",
        "Firefighters contain wildfire to 60 percent. Full containment expected by Friday",
        "Temporary housing being arranged for 3000 displaced families by the relief committee",
        "Airport closed until further notice due to runway damage from the earthquake",
        "Mobile clinics deployed to remote villages cut off by landslides after heavy rainfall",
        "Death toll from the flooding has risen to 47 with dozens still unaccounted for",
        "Emergency fund of 5 million dollars announced by the government for disaster relief",
        "Water rescue teams saving stranded motorists from submerged vehicles on highway",
        "Building inspector reports 200 structures condemned as unsafe after the earthquake",
        "Relief supplies including blankets food and medicine being airlifted to affected areas",
        "Communication towers damaged by the storm causing widespread cell phone outages",
        "Temporary shelters reaching full capacity need additional locations for evacuees",
        "Hazardous materials spill reported near the industrial area following the earthquake",
        "Disaster medical assistance teams from neighboring states arriving to help",
        "Major dam breach warning issued for communities downstream along the river valley",
        "Search dogs deployed to locate survivors trapped under collapsed building rubble",
        "Makeshift hospital treating over 500 patients injured in the earthquake disaster",
        "Electricity and water service disrupted to entire eastern district since the cyclone",
        "Emergency hotline established for reporting missing family members after the flood",
    ]
    
    # ── NOT INFORMATIVE EXAMPLES ──
    # These texts are disaster-related but do NOT contain actionable information:
    # personal opinions, prayers, emotional reactions, retweet requests, general
    # complaints about climate change, etc. A human responder cannot take any
    # specific rescue or relief action based on these posts.
    not_informative_texts = [
        "This storm is crazy omg 😭",
        "Praying for everyone affected by the earthquake 🙏",
        "Stay safe everyone! Sending love and prayers ❤️",
        "I think the government should do more about disaster preparedness",
        "Just saw the news about the flooding. So heartbreaking",
        "Mother nature is angry these days. Climate change is real",
        "Does anyone else feel like natural disasters are getting worse every year",
        "Watching the hurricane coverage on TV. This is terrifying",
        "My thoughts and prayers go out to all the victims",
        "We need to take climate change more seriously after this disaster",
        "OMG the earthquake was so scary! I literally screamed",
        "Can not believe another hurricane is coming. When will this end",
        "Sending positive vibes to everyone in the disaster zone",
        "Just heard about the earthquake. Hope everyone is alright",
        "Another day another natural disaster. World is ending",
        "RT if you stand with the flood victims. Share to raise awareness",
        "This is why we need better infrastructure. Buildings keep collapsing",
        "Wishing all the affected families strength during this difficult time",
        "The photos of the destruction are absolutely devastating to see",
        "I remember the last hurricane we had. It was nothing like this one",
        "People need to evacuate when told to. Why do they stay behind",
        "God bless the first responders working tirelessly to save lives",
        "Natural disasters remind us how small we are compared to nature",
        "Can we please talk about how unprepared our city is for disasters",
        "Just donated to the relief fund. Everyone should contribute what they can",
        "This earthquake reminded me of the one we had back in 2015",
        "Is it just me or are there more earthquakes happening this year",
        "The weather has been absolutely insane this whole season honestly",
        "Hope the rain stops soon. My backyard is completely flooded already",
        "Thinking about all the families who lost everything in this disaster",
        "We really need better early warning systems for these natural disasters",
        "Social media is full of disaster news today. So depressing honestly",
        "Climate scientists have been warning us about this for decades now",
        "The news coverage of this disaster is really making me anxious",
        "Heard the thunder last night and thought it was an earthquake",
        "Why do people build houses in flood zones then complain about floods",
        "My cousin lives near the affected area. Thank god she is safe",
        "Another hurricane season and we are still not prepared for it",
        "The pictures from the earthquake zone are breaking my heart right now",
        "Government response to this disaster has been incredibly slow as usual",
        "Having flashbacks to when our town was hit by the tornado years ago",
        "When will people learn to take evacuation orders seriously",
        "Cannot stop watching the disaster coverage. It is so devastating",
        "Feeling so helpless watching all this destruction on the news today",
        "Teachers and students must be terrified after the earthquake",
        "Why does it always seem like poor communities get hit the hardest",
        "Wow just saw the video of the flood waters. That is insane",
        "My heart goes out to all the animals affected by the wildfire",
        "This hurricane season has been the worst one in recent memory",
        "I cannot even imagine what those people are going through right now",
    ]
    
    # Create TSV file matching CrisisMMD format
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    test_file = os.path.join(raw_dir, "test_synthetic_data.tsv")
    
    events = [
        "hurricane_harvey", "hurricane_irma", "hurricane_maria",
        "california_wildfires", "iraq_iran_earthquake",
        "mexico_earthquake", "srilanka_floods"
    ]
    
    rows = []
    
    # Create informative samples (repeat to get ~200+ rows for robust training)
    for i, text in enumerate(informative_texts):
        rows.append({
            'event_name': random.choice(events),
            'tweet_id': f'info_{i+1:04d}',
            'tweet_text': text,
            'label': 'informative'
        })
    # Duplicate with slight variations for more training data
    for i, text in enumerate(informative_texts):
        rows.append({
            'event_name': random.choice(events),
            'tweet_id': f'info_{i+100:04d}',
            'tweet_text': text + " Please help.",
            'label': 'informative'
        })
    for i, text in enumerate(informative_texts[:30]):
        rows.append({
            'event_name': random.choice(events),
            'tweet_id': f'info_{i+200:04d}',
            'tweet_text': "URGENT: " + text,
            'label': 'informative'
        })
    
    # Create not_informative samples
    for i, text in enumerate(not_informative_texts):
        rows.append({
            'event_name': random.choice(events),
            'tweet_id': f'notinfo_{i+1:04d}',
            'tweet_text': text,
            'label': 'not_informative'
        })
    for i, text in enumerate(not_informative_texts):
        rows.append({
            'event_name': random.choice(events),
            'tweet_id': f'notinfo_{i+100:04d}',
            'tweet_text': text + " 🙏",
            'label': 'not_informative'
        })
    for i, text in enumerate(not_informative_texts[:30]):
        rows.append({
            'event_name': random.choice(events),
            'tweet_id': f'notinfo_{i+200:04d}',
            'tweet_text': "Honestly, " + text.lower(),
            'label': 'not_informative'
        })
    
    # Shuffle
    random.seed(42)
    random.shuffle(rows)
    
    # Write TSV
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['event_name', 'tweet_id', 'tweet_text', 'label'], 
                                delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Created synthetic dataset: {test_file}")
    print(f"   Total rows: {len(rows)}")
    print(f"   Informative: {sum(1 for r in rows if r['label'] == 'informative')}")
    print(f"   Not Informative: {sum(1 for r in rows if r['label'] == 'not_informative')}")
    
    return test_file


def run_full_pipeline():
    """
    Run the complete training pipeline using synthetic data.
    
    This function orchestrates all steps of the ML pipeline in sequence:
    1. Create synthetic test data (simulating CrisisMMD)
    2. Run train.py which handles: load → preprocess → TF-IDF → train → evaluate → save
    3. Test predictions on sample texts
    4. Verify all expected output files were created
    
    If this function completes without errors, the entire codebase is working.
    """
    
    print("\n" + "=" * 60)
    print("🧪 FULL PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Create synthetic data
    create_synthetic_dataset()
    
    # Step 2: Run the training pipeline
    print("\n" + "=" * 60)
    print("🚀 Running training pipeline...")
    print("=" * 60)
    
    from src.train import main as train_main
    train_main()
    
    # Step 3: Test predictions
    print("\n" + "=" * 60)
    print("🔮 Testing predictions...")
    print("=" * 60)
    
    from src.predict import load_model, predict, predict_batch
    
    model, vectorizer = load_model()
    
    test_inputs = [
        "People are trapped after the flood near the bridge and need immediate rescue.",
        "This storm is crazy omg 😭",
        "Red Cross setting up emergency shelters at Lincoln High School for flood evacuees.",
        "Praying for everyone affected by the earthquake 🙏",
        "URGENT: 3 buildings collapsed in downtown after 7.2 magnitude earthquake. Multiple casualties reported.",
        "Just saw the news about the hurricane. Stay safe everyone!",
        "Water supply contaminated after flood. Boil water advisory issued for zones 3-7.",
    ]
    
    print("\n📋 Prediction Results:")
    for text in test_inputs:
        label, confidence = predict(text, model, vectorizer)
        icon = "🟢" if label == "Informative" else "🔴"
        print(f"\n  Input: \"{text[:70]}{'...' if len(text)>70 else ''}\"")
        print(f"  {icon} {label} | Confidence: {confidence:.1%}")
    
    # Step 4: Verify all output files exist
    print("\n" + "=" * 60)
    print("📁 Verifying output files...")
    print("=" * 60)
    
    expected_files = [
        os.path.join(PROJECT_ROOT, "models", "disaster_model.pkl"),
        os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl"),
        os.path.join(PROJECT_ROOT, "models", "model_results.json"),
        os.path.join(PROJECT_ROOT, "reports", "figures", "model_comparison.png"),
        os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.csv"),
        os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_data.csv"),
    ]
    
    all_ok = True
    for filepath in expected_files:
        exists = os.path.exists(filepath)
        icon = "✅" if exists else "❌"
        name = os.path.relpath(filepath, PROJECT_ROOT)
        size = f"({os.path.getsize(filepath):,} bytes)" if exists else "(MISSING)"
        print(f"  {icon} {name} {size}")
        if not exists:
            all_ok = False
    
    # Check confusion matrix files
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    cm_files = [f for f in os.listdir(figures_dir) if f.startswith("confusion_matrix_")]
    print(f"  {'✅' if len(cm_files) == 4 else '⚠️'} Confusion matrices: {len(cm_files)} found")
    for f in cm_files:
        print(f"      📊 {f}")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL TESTS PASSED! Pipeline is working correctly.")
    else:
        print("⚠️ Some files are missing. Check the errors above.")
    print("=" * 60)
    
    # Clean up: remove synthetic data file (keep the trained model)
    # Commenting out so the app can be tested:
    # os.remove(os.path.join(PROJECT_ROOT, "data", "raw", "test_synthetic_data.tsv"))
    
    return all_ok


if __name__ == "__main__":
    run_full_pipeline()

"""
Comprehensive disease database for all 39 plant disease classes.
Provides scientifically accurate information for each disease class
detected by the CNN model.
"""

DISEASE_DATABASE = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "display_name": "Apple — Apple Scab",
        "description": "Apple scab is caused by the fungus Venturia inaequalis. It is one of the most serious diseases of apple trees worldwide, causing dark, scabby lesions on leaves and fruit that reduce marketability and yield.",
        "symptoms": ["Dark olive-green to brown velvety spots on leaves", "Scabby, corky lesions on fruit surface", "Premature leaf drop and defoliation", "Distorted or cracked fruit in severe cases"],
        "treatment": ["Apply fungicides (captan or myclobutanil) during early spring", "Remove and destroy fallen infected leaves", "Plant scab-resistant apple cultivars"],
        "severity": "high",
        "category": "fungal"
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "display_name": "Apple — Black Rot",
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. It affects leaves, fruit, and bark of apple trees, producing characteristic 'frog-eye' leaf spots and rotting fruit.",
        "symptoms": ["Circular brown leaf spots with purple borders (frog-eye)", "Firm black rot spreading from calyx end of fruit", "Cankers on branches and trunk", "Mummified fruit remaining on the tree"],
        "treatment": ["Prune out dead wood and cankers during dormancy", "Remove mummified fruit from trees and ground", "Apply fungicides during bloom and post-bloom periods"],
        "severity": "high",
        "category": "fungal"
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "display_name": "Apple — Cedar Apple Rust",
        "description": "Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae. It requires both cedar/juniper and apple trees to complete its life cycle, producing bright orange spots on apple leaves.",
        "symptoms": ["Bright yellow-orange spots on upper leaf surfaces", "Tube-like projections on lower leaf surface", "Reduced fruit size and quality", "Premature leaf drop in severe infections"],
        "treatment": ["Remove nearby cedar and juniper trees within 2 miles if possible", "Apply fungicides (myclobutanil) from pink bud through petal fall", "Plant rust-resistant apple varieties"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "display_name": "Apple — Healthy",
        "description": "This apple leaf shows no signs of disease. The leaf displays normal green coloration, intact margins, and healthy venation patterns indicating proper nutrient uptake and growth.",
        "symptoms": ["No disease symptoms present", "Uniform green coloration", "Intact leaf margins", "Normal growth patterns"],
        "treatment": ["Continue regular maintenance and monitoring", "Maintain proper watering and fertilization schedule", "Practice preventive fungicide applications during wet seasons"],
        "severity": "none",
        "category": "healthy"
    },
    "Background_without_leaves": {
        "plant": "Background",
        "disease": "No Leaves Detected",
        "display_name": "Background — Without Leaves",
        "description": "The image does not contain identifiable plant leaf tissue. This may be a background image, soil, or non-plant material. Please upload a clear image of a plant leaf for diagnosis.",
        "symptoms": ["No plant tissue detected", "Background material identified", "Unable to perform disease analysis"],
        "treatment": ["Upload a clear image of a plant leaf", "Ensure the leaf fills most of the frame", "Use good lighting and sharp focus"],
        "severity": "none",
        "category": "other"
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "display_name": "Blueberry — Healthy",
        "description": "This blueberry leaf appears healthy with no signs of disease. Normal coloration and leaf structure indicate good plant health and proper growing conditions.",
        "symptoms": ["No disease symptoms present", "Normal green coloration", "Healthy leaf structure", "No discoloration or spotting"],
        "treatment": ["Maintain acidic soil pH (4.5-5.5)", "Ensure adequate mulching and irrigation", "Monitor for pests and diseases regularly"],
        "severity": "none",
        "category": "healthy"
    },
    "Cherry___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "display_name": "Cherry — Powdery Mildew",
        "description": "Cherry powdery mildew is caused by the fungus Podosphaera clandestina. It produces a white powdery coating on leaves and shoots, affecting photosynthesis and reducing fruit quality.",
        "symptoms": ["White powdery fungal growth on leaf surfaces", "Leaf curling and distortion", "Stunted new shoot growth", "Reduced fruit set and quality"],
        "treatment": ["Apply sulfur-based or potassium bicarbonate fungicides", "Improve air circulation by pruning", "Avoid overhead irrigation to reduce humidity"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Cherry___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "display_name": "Cherry — Healthy",
        "description": "This cherry leaf is healthy with no visible signs of infection. The leaf shows normal color, shape, and venation characteristic of a vigorous cherry tree.",
        "symptoms": ["No disease symptoms present", "Normal green coloration", "Healthy leaf margins", "Normal venation patterns"],
        "treatment": ["Continue regular orchard maintenance", "Apply preventive fungicide sprays during wet seasons", "Monitor for early signs of powdery mildew"],
        "severity": "none",
        "category": "healthy"
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "display_name": "Corn — Gray Leaf Spot",
        "description": "Gray leaf spot is caused by Cercospora zeae-maydis. It is one of the most yield-limiting diseases of corn, producing rectangular gray to tan lesions that follow leaf veins.",
        "symptoms": ["Rectangular gray to tan lesions between leaf veins", "Lesions may coalesce causing large necrotic areas", "Lower leaves affected first, progressing upward", "Severe blighting of foliage"],
        "treatment": ["Plant resistant corn hybrids", "Rotate crops to reduce inoculum", "Apply foliar fungicides (strobilurin-based) at tassel stage"],
        "severity": "high",
        "category": "fungal"
    },
    "Corn___Common_rust": {
        "plant": "Corn",
        "disease": "Common Rust",
        "display_name": "Corn — Common Rust",
        "description": "Common rust of corn is caused by the fungus Puccinia sorghi. It produces small, circular to elongated, cinnamon-brown pustules on both leaf surfaces that release powdery spores.",
        "symptoms": ["Small circular to elongated brown-red pustules on leaves", "Pustules on both upper and lower leaf surfaces", "Powdery rust-colored spores released from pustules", "Chlorosis and necrosis around pustules in severe cases"],
        "treatment": ["Plant rust-resistant corn hybrids", "Apply foliar fungicides if detected early in season", "Scout fields regularly during humid conditions"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "display_name": "Corn — Northern Leaf Blight",
        "description": "Northern corn leaf blight is caused by the fungus Exserohilum turcicum. It produces large, cigar-shaped gray-green lesions on corn leaves that can significantly reduce yield.",
        "symptoms": ["Large cigar-shaped gray-green to tan lesions (1-6 inches)", "Lesions may coalesce blighting entire leaves", "Lower leaves affected first", "Dark fungal sporulation visible in humid conditions"],
        "treatment": ["Plant resistant hybrids with Ht genes", "Apply fungicides at early disease onset", "Rotate corn with non-host crops for at least one year"],
        "severity": "high",
        "category": "fungal"
    },
    "Corn___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "display_name": "Corn — Healthy",
        "description": "This corn leaf shows no symptoms of disease. The leaf displays normal green coloration and healthy tissue, indicating a vigorous, well-nourished corn plant.",
        "symptoms": ["No disease symptoms present", "Uniform green coloration", "Healthy leaf tissue", "No lesions or discoloration"],
        "treatment": ["Maintain balanced fertility program", "Monitor soil moisture levels", "Scout regularly for early disease detection"],
        "severity": "none",
        "category": "healthy"
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "display_name": "Grape — Black Rot",
        "description": "Grape black rot is caused by the fungus Guignardia bidwellii. It can destroy an entire grape crop, causing characteristic brown circular leaf spots and shriveled black mummified fruit.",
        "symptoms": ["Circular brown lesions with dark borders on leaves", "Small black pycnidia visible in lesion centers", "Fruit turns brown then shrivels into hard black mummies", "Shoot lesions may cause cankers"],
        "treatment": ["Apply fungicides (mancozeb, myclobutanil) from bud break through veraison", "Remove mummified fruit and infected debris", "Ensure good canopy management for air circulation"],
        "severity": "high",
        "category": "fungal"
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "display_name": "Grape — Esca (Black Measles)",
        "description": "Esca is a complex trunk disease caused by multiple fungi including Phaeomoniella chlamydospora and Phaeoacremonium spp. It causes tiger-stripe patterns on leaves and internal wood decay.",
        "symptoms": ["Interveinal chlorosis creating tiger-stripe pattern on leaves", "Dark spots or streaks on berries (measles)", "Sudden vine collapse (apoplexy) in hot weather", "Brown to black wood discoloration in cross-sections"],
        "treatment": ["No curative treatment available; manage symptomatically", "Prune during dry weather to minimize infection", "Remedial surgery to remove infected wood and apply wound sealants"],
        "severity": "high",
        "category": "fungal"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "display_name": "Grape — Isariopsis Leaf Spot",
        "description": "Isariopsis leaf spot is caused by Pseudocercospora vitis. It produces angular dark brown spots on grape leaves, primarily affecting older leaves and reducing photosynthetic capacity.",
        "symptoms": ["Dark brown angular spots on leaves", "Spots may have yellowish halos", "Premature defoliation of lower leaves", "Reduced vine vigor in severe cases"],
        "treatment": ["Apply copper-based fungicides preventively", "Maintain proper canopy management", "Remove and destroy infected leaf debris"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "display_name": "Grape — Healthy",
        "description": "This grape leaf shows no signs of disease. Normal coloration, intact margins, and healthy venation indicate a vigorous grapevine with proper canopy management.",
        "symptoms": ["No disease symptoms present", "Uniform green coloration", "Normal leaf shape and margins", "Healthy venation patterns"],
        "treatment": ["Maintain balanced fertilization and irrigation", "Continue preventive spray programs", "Monitor for early disease symptoms regularly"],
        "severity": "none",
        "category": "healthy"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "display_name": "Orange — Citrus Greening (HLB)",
        "description": "Huanglongbing (HLB) is caused by the bacterium Candidatus Liberibacter asiaticus, transmitted by the Asian citrus psyllid. It is the most devastating citrus disease worldwide with no cure available.",
        "symptoms": ["Asymmetric blotchy yellow mottling on leaves", "Small, lopsided, green fruit that fails to ripen", "Bitter, misshapen fruit with aborted seeds", "Twig dieback and overall tree decline"],
        "treatment": ["Control Asian citrus psyllid vector with insecticides", "Remove and destroy infected trees to reduce spread", "Plant certified disease-free nursery stock"],
        "severity": "high",
        "category": "bacterial"
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "display_name": "Peach — Bacterial Spot",
        "description": "Bacterial spot of peach is caused by Xanthomonas arboricola pv. pruni. It affects leaves, fruit, and twigs, causing angular leaf spots, fruit lesions, and defoliation.",
        "symptoms": ["Small angular purple-brown spots on leaves", "Shot-hole effect as lesion centers fall out", "Shallow, dark, cracked lesions on fruit", "Twig cankers and defoliation in severe cases"],
        "treatment": ["Apply copper-based bactericides during early season", "Plant resistant peach varieties when available", "Avoid overhead irrigation to reduce leaf wetness"],
        "severity": "moderate",
        "category": "bacterial"
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "display_name": "Peach — Healthy",
        "description": "This peach leaf appears healthy with no bacterial or fungal infections. Normal elongated leaf shape and green coloration indicate a well-maintained peach tree.",
        "symptoms": ["No disease symptoms present", "Normal green coloration", "Intact leaf tissue", "No spots or lesions"],
        "treatment": ["Continue regular orchard care", "Apply preventive copper sprays during dormant season", "Monitor for bacterial spot during wet weather"],
        "severity": "none",
        "category": "healthy"
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pepper (Bell)",
        "disease": "Bacterial Spot",
        "display_name": "Bell Pepper — Bacterial Spot",
        "description": "Bacterial spot of pepper is caused by Xanthomonas campestris pv. vesicatoria. It is a major disease in warm, humid climates, causing leaf spots, fruit blemishes, and defoliation.",
        "symptoms": ["Small dark water-soaked spots on leaves", "Spots turn brown with yellow halos", "Raised scab-like lesions on fruit", "Severe defoliation leading to sunscald on fruit"],
        "treatment": ["Use disease-free seed and transplants", "Apply copper-based sprays with mancozeb", "Rotate crops and avoid working in wet fields"],
        "severity": "moderate",
        "category": "bacterial"
    },
    "Pepper,_bell___healthy": {
        "plant": "Pepper (Bell)",
        "disease": "Healthy",
        "display_name": "Bell Pepper — Healthy",
        "description": "This bell pepper leaf is healthy with no signs of bacterial or other infections. The smooth, dark green leaf indicates optimal growing conditions.",
        "symptoms": ["No disease symptoms present", "Uniform dark green color", "Smooth leaf surface", "Normal leaf shape"],
        "treatment": ["Maintain consistent watering schedule", "Apply balanced fertilization", "Monitor for pests and disease regularly"],
        "severity": "none",
        "category": "healthy"
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "display_name": "Potato — Early Blight",
        "description": "Potato early blight is caused by the fungus Alternaria solani. It produces characteristic concentric-ring target spots on older leaves and can reduce tuber yield significantly.",
        "symptoms": ["Dark brown target-shaped spots with concentric rings", "Spots appear on older, lower leaves first", "Yellowing around leaf lesions", "Premature defoliation reducing tuber size"],
        "treatment": ["Apply chlorothalonil or mancozeb fungicides preventively", "Maintain adequate plant nutrition (especially nitrogen)", "Practice crop rotation with non-solanaceous crops"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "display_name": "Potato — Late Blight",
        "description": "Late blight is caused by the oomycete Phytophthora infestans — the same pathogen responsible for the Irish Potato Famine. It can destroy entire fields within days under favorable conditions.",
        "symptoms": ["Large water-soaked gray-green lesions on leaves", "White fuzzy mold on undersides of leaves in humid conditions", "Rapid browning and death of entire plant", "Firm brown rot in tubers"],
        "treatment": ["Apply preventive fungicides (chlorothalonil, metalaxyl) before symptoms appear", "Destroy volunteer potato plants and cull piles", "Harvest tubers during dry conditions and cure properly"],
        "severity": "high",
        "category": "fungal"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "display_name": "Potato — Healthy",
        "description": "This potato leaf shows no signs of early blight, late blight, or other diseases. Healthy green foliage indicates proper growing conditions and disease management.",
        "symptoms": ["No disease symptoms present", "Normal green coloration", "No lesions or water-soaked areas", "Vigorous growth pattern"],
        "treatment": ["Continue preventive fungicide program during wet weather", "Maintain proper hill spacing and drainage", "Monitor for late blight alerts in your region"],
        "severity": "none",
        "category": "healthy"
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "display_name": "Raspberry — Healthy",
        "description": "This raspberry leaf appears healthy with no visible disease symptoms. The trifoliate leaf structure and green coloration indicate normal plant development.",
        "symptoms": ["No disease symptoms present", "Normal trifoliate leaf structure", "Healthy green color", "No spotting or discoloration"],
        "treatment": ["Maintain proper cane management and thinning", "Ensure adequate air circulation in the row", "Apply preventive fungicides during bloom"],
        "severity": "none",
        "category": "healthy"
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "display_name": "Soybean — Healthy",
        "description": "This soybean leaf is healthy with no signs of disease. The trifoliate leaves show normal green coloration indicating proper nitrogen fixation and plant health.",
        "symptoms": ["No disease symptoms present", "Normal trifoliate leaf shape", "Uniform green coloration", "No lesions or chlorosis"],
        "treatment": ["Maintain appropriate planting density", "Scout regularly for soybean rust and other diseases", "Ensure proper soil fertility and pH"],
        "severity": "none",
        "category": "healthy"
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "display_name": "Squash — Powdery Mildew",
        "description": "Squash powdery mildew is primarily caused by Podosphaera xanthii. It creates a white powdery coating on leaves that reduces photosynthesis and can significantly impact fruit production.",
        "symptoms": ["White powdery spots on upper leaf surfaces", "Spots enlarge and coalesce covering entire leaves", "Yellowing and browning of severely infected leaves", "Premature leaf senescence reducing fruit quality"],
        "treatment": ["Apply potassium bicarbonate or sulfur-based fungicides", "Plant powdery mildew-resistant squash varieties", "Increase plant spacing for better air circulation"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "display_name": "Strawberry — Leaf Scorch",
        "description": "Strawberry leaf scorch is caused by the fungus Diplocarpon earlianum. It produces small irregular dark purple spots that enlarge and merge, creating a scorched appearance on leaves.",
        "symptoms": ["Small irregular dark purple spots on upper leaf surface", "Spots enlarge and coalesce creating scorched appearance", "Leaf margins and areas between spots turn brown", "Severely affected leaves dry up and curl"],
        "treatment": ["Remove and destroy infected leaves and plant debris", "Apply fungicides (captan, thiophanate-methyl) during renovation", "Renovate strawberry beds to remove old infected foliage"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "display_name": "Strawberry — Healthy",
        "description": "This strawberry leaf is healthy with no visible signs of leaf scorch or other diseases. The trifoliate leaves show bright green coloration and normal serrated margins.",
        "symptoms": ["No disease symptoms present", "Bright green trifoliate leaves", "Normal serrated leaf margins", "No spotting or browning"],
        "treatment": ["Maintain proper plant spacing and weed control", "Avoid overhead irrigation when possible", "Apply preventive fungicides during wet periods"],
        "severity": "none",
        "category": "healthy"
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "display_name": "Tomato — Bacterial Spot",
        "description": "Tomato bacterial spot is caused by several Xanthomonas species. It causes leaf spots, fruit lesions, and defoliation, particularly severe in warm, rainy conditions.",
        "symptoms": ["Small dark water-soaked spots on leaves", "Spots may have yellow halos", "Raised, scabby lesions on fruit", "Severe defoliation in wet weather"],
        "treatment": ["Use pathogen-free seed and transplants", "Apply copper-based bactericides preventively", "Avoid overhead irrigation and handling wet plants"],
        "severity": "moderate",
        "category": "bacterial"
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "display_name": "Tomato — Early Blight",
        "description": "Tomato early blight is caused by Alternaria solani. It produces characteristic concentric-ring target spots starting on lower leaves and progressing upward, reducing yield and fruit quality.",
        "symptoms": ["Dark brown concentric-ring target spots on older leaves", "Yellow halos around lesions", "Progressive defoliation from bottom upward", "Dark leathery spots on stem and fruit"],
        "treatment": ["Apply chlorothalonil or copper fungicides at first symptoms", "Stake plants and mulch to prevent soil splash", "Practice 3-year crop rotation with non-solanaceous crops"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "display_name": "Tomato — Late Blight",
        "description": "Tomato late blight is caused by Phytophthora infestans. Under cool, wet conditions it can destroy tomato plants rapidly, producing water-soaked lesions that turn brown and papery.",
        "symptoms": ["Large water-soaked gray-green blotches on leaves", "White fuzzy sporulation on leaf undersides in humid weather", "Rapid browning and collapse of foliage", "Firm brown lesions on green and ripe fruit"],
        "treatment": ["Apply preventive fungicides before symptoms appear", "Remove and destroy infected plants immediately", "Improve air circulation and avoid overhead watering"],
        "severity": "high",
        "category": "fungal"
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "display_name": "Tomato — Leaf Mold",
        "description": "Tomato leaf mold is caused by the fungus Passalora fulva (formerly Cladosporium fulvum). It thrives in greenhouse conditions with high humidity, producing olive-green to brown velvety mold on leaves.",
        "symptoms": ["Pale green to yellow spots on upper leaf surface", "Olive-green to brown velvety mold on lower leaf surface", "Affected leaves curl, wither, and drop", "Can spread to fruit in severe cases"],
        "treatment": ["Improve greenhouse ventilation and reduce humidity below 85%", "Apply chlorothalonil or copper-based fungicides", "Remove and destroy infected leaves promptly"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "display_name": "Tomato — Septoria Leaf Spot",
        "description": "Septoria leaf spot is caused by Septoria lycopersici. It produces numerous small circular spots with dark borders and tan centers, causing extensive defoliation primarily on lower leaves.",
        "symptoms": ["Numerous small circular spots (1-3mm) with dark borders", "Tan to gray centers with tiny dark pycnidia", "Lower leaves affected first with upward progression", "Extensive defoliation exposing fruit to sunscald"],
        "treatment": ["Apply fungicides (chlorothalonil, mancozeb) at first appearance", "Mulch around plants to prevent soil splashing", "Remove lower affected leaves to slow spread"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-Spotted Spider Mite)",
        "display_name": "Tomato — Two-Spotted Spider Mite",
        "description": "Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that feed on plant cells, causing stippling and yellowing. Infestations explode in hot, dry conditions and can severely weaken plants.",
        "symptoms": ["Fine yellow stippling on upper leaf surfaces", "Fine webbing on leaf undersides and between leaves", "Bronzing and drying of leaves in severe infestations", "Tiny moving dots visible with magnification"],
        "treatment": ["Spray forceful water jets to dislodge mites", "Apply miticides (abamectin) or horticultural oils", "Introduce predatory mites (Phytoseiulus persimilis) for biological control"],
        "severity": "moderate",
        "category": "pest"
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "display_name": "Tomato — Target Spot",
        "description": "Target spot is caused by the fungus Corynespora cassiicola. It produces concentric-ring target-like spots on leaves, stems, and fruit, and can be particularly damaging in warm, humid climates.",
        "symptoms": ["Brown spots with concentric rings (target pattern) on leaves", "Spots may have yellow halos", "Lesions on stems and fruit", "Severe defoliation in humid conditions"],
        "treatment": ["Apply fungicides (chlorothalonil, azoxystrobin) preventively", "Improve air circulation by proper spacing and staking", "Remove and destroy crop debris after harvest"],
        "severity": "moderate",
        "category": "fungal"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "display_name": "Tomato — Yellow Leaf Curl Virus",
        "description": "Tomato yellow leaf curl virus (TYLCV) is transmitted by the whitefly Bemisia tabaci. Infected plants show severe stunting, leaf curling, and yellowing, with dramatic yield losses.",
        "symptoms": ["Severe upward curling and cupping of leaves", "Yellowing of leaf margins and interveinal areas", "Stunted plant growth and shortened internodes", "Dramatically reduced fruit set"],
        "treatment": ["Control whitefly populations with insecticides or reflective mulches", "Remove and destroy infected plants immediately", "Use TYLCV-resistant tomato varieties"],
        "severity": "high",
        "category": "viral"
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Mosaic Virus",
        "display_name": "Tomato — Mosaic Virus",
        "description": "Tomato mosaic virus (ToMV) is a highly stable tobamovirus that can survive on tools, hands, and plant debris. It causes mottled light and dark green mosaic patterns on leaves and reduced fruit quality.",
        "symptoms": ["Mottled light and dark green mosaic pattern on leaves", "Leaf distortion, curling, and reduced leaf size", "Stunted growth and reduced fruit production", "Uneven fruit ripening and internal browning"],
        "treatment": ["No cure exists; remove and destroy infected plants", "Disinfect tools, hands, and stakes with 10% bleach solution", "Plant TMV/ToMV-resistant tomato varieties"],
        "severity": "high",
        "category": "viral"
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "display_name": "Tomato — Healthy",
        "description": "This tomato leaf shows no signs of disease. The compound leaf displays normal green coloration, healthy leaflets, and no lesions, indicating a well-maintained and disease-free plant.",
        "symptoms": ["No disease symptoms present", "Normal green compound leaves", "Healthy leaflet shape and size", "No spots, mottling, or curling"],
        "treatment": ["Continue regular monitoring and scouting", "Maintain proper watering (avoid wetting foliage)", "Practice crop rotation and sanitation"],
        "severity": "none",
        "category": "healthy"
    }
}


def get_disease_info(class_name):
    """Get disease info for a specific class name."""
    return DISEASE_DATABASE.get(class_name, {
        "plant": "Unknown",
        "disease": "Unknown",
        "display_name": class_name.replace("___", " — ").replace("_", " "),
        "description": "No information available for this class.",
        "symptoms": ["Unknown"],
        "treatment": ["Consult a local agricultural extension specialist"],
        "severity": "moderate",
        "category": "other"
    })


def get_all_diseases():
    """Get the complete disease database."""
    return DISEASE_DATABASE


def get_diseases_by_plant(plant):
    """Filter diseases by plant type (case-insensitive)."""
    plant_lower = plant.lower()
    return {
        k: v for k, v in DISEASE_DATABASE.items()
        if v["plant"].lower() == plant_lower
    }


def get_unique_plants():
    """Get a sorted list of unique plant names."""
    plants = sorted(set(v["plant"] for v in DISEASE_DATABASE.values()))
    return plants


def get_disease_categories():
    """Get a sorted list of unique disease categories."""
    return sorted(set(v["category"] for v in DISEASE_DATABASE.values()))

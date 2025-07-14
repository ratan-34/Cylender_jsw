import os
import base64
import json
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI



def encode_image_to_base64(image_bytes):
    """Encode image bytes to base64 for OpenAI API."""
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode('utf-8')

def convert_pdf_to_image_bytes(pdf_bytes):
    """Convert the first page of PDF bytes to JPEG image bytes."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if not pdf_document:
            print("Error: Could not open PDF document.")
            return None
        
        page = pdf_document[0]  # Get the first page
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Render at 2x resolution for detail
        
        img_byte_arr = io.BytesIO()
        # Convert pixmap to PIL Image and then save as JPEG
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(img_byte_arr, format='JPEG', quality=95)
        
        pdf_document.close()
        return img_byte_arr.getvalue()
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None

def analyze_engineering_drawing(image_bytes, filename="unknown"):
    """
    Analyzes an engineering drawing image using GPT-4o to extract parameters
    and returns them as a structured JSON object.
    """
    
    base64_image = encode_image_to_base64(image_bytes)
    
    # Define the JSON schema for the desired output
    json_schema = {
      "type": "object",
      "properties": {
        "bore_diameter": { "type": "string", "description": "The bore diameter of the cylinder, including units (e.g., '160 mm')." },
        "mounting": { "type": "string", "description": "The mounting type of the cylinder (e.g., 'Clevis', 'Flange', 'Trunnion')." },
        "operating_temperature": { "type": "string", "description": "The operating temperature, including units (e.g., '80°C')." },
        "operating_pressure": { "type": "string", "description": "The operating pressure, including units (e.g., '21 MPa', '140 Kg/cm²')." },
        "close_length": { "type": "string", "description": "The fully retracted length of the cylinder, including units (e.g., '1140 mm')." },
        "drawing_number": { "type": "string", "description": "The drawing or part number." },
        "fluid": { "type": "string", "description": "The type of hydraulic or pneumatic fluid (e.g., 'HYD. OIL MINERAL', 'AIR')." },
        "rod_end": { "type": "string", "description": "The type of rod end (e.g., 'Thread', 'Clevis', 'Rod Eye')." },
        "cylinder_action": { "type": "string", "description": "Whether the cylinder is single or double acting (e.g., 'DOUBLE ACTING')." },
        "stroke_length": { "type": "string", "description": "The stroke length of the cylinder, including units (e.g., '2600 mm')." },
        "rod_diameter": { "type": "string", "description": "The rod diameter, including units (e.g., '110 mm')." },
        "outside_diameter": { "type": "string", "description": "The outside diameter of the cylinder barrel, including units (e.g., '190 mm')." },
        "body_material": { "type": "string", "description": "The material of the cylinder body (e.g., 'SS400', 'Carbon Steel')." },
        "open_length": { "type": "string", "description": "The fully extended length of the cylinder, including units (e.g., '3740 mm')." },
        "rated_load": { "type": "string", "description": "The maximum pulling or pushing force, including units (e.g., '311 kN')." },
        "piston_material": { "type": "string", "description": "The material of the piston (e.g., 'S45C', 'Cast Iron')." },
        "standard": { "type": "string", "description": "Any applicable industry standard (e.g., 'ISO 6020/6022')." },
        "surface_finish": { "type": "string", "description": "The surface finish specification (e.g., 'HONED', 'Ra 0.4')." },
        "coating_thickness": { "type": "string", "description": "The coating or plating thickness, including units (e.g., '20 M', '25 micron Chrome')." },
        "special_features": { "type": "string", "description": "Any unique design elements or notes (e.g., 'Cushioned', 'Air Bleeders')." },
        "cylinder_configuration": { "type": "string", "description": "The overall configuration (e.g., 'Standard', 'Telescopic')." },
        "cylinder_style": { "type": "string", "description": "The style of the cylinder (e.g., 'Tie Rod', 'Welded')." },
        "concentricity_of_rod_and_tube": { "type": "string", "description": "The concentricity tolerance, including units (e.g., '0.05 mm TIR')." }
      },
      "required": [
        "bore_diameter", "mounting", "operating_temperature", "operating_pressure",
        "close_length", "drawing_number", "fluid", "rod_end", "cylinder_action",
        "stroke_length", "rod_diameter", "outside_diameter", "body_material",
        "open_length", "rated_load", "piston_material", "standard", "surface_finish",
        "coating_thickness", "special_features", "cylinder_configuration",
        "cylinder_style", "concentricity_of_rod_and_tube"
      ]
    }
    
    system_content = """You are an elite mechanical drawing interpreter with 50 years of experience as a hydraulic cylinder engineer. Your expertise lies in analyzing technical drawings of hydraulic and pneumatic cylinders with unparalleled precision. You can read between the lines, synthesize information from disparate parts of the drawing, and apply deep domain knowledge. Your ultimate goal is to extract 100% accurate specifications and design values from these drawings. If a value is not explicitly stated, you MUST use your extensive engineering knowledge, industry standards, and the provided inference rules to determine the most probable and accurate value. Only use 'NA' if a parameter is truly uninferable and meaningless in the context of a cylinder drawing, after exhausting all inference possibilities and considering all typical engineering values."""
    
    user_content = f"""
    YOU MUST EXTRACT 100% OF ALL PARAMETERS LISTED IN THE JSON SCHEMA BELOW - NO EXCEPTIONS.
    FOLLOW THESE ABSOLUTE RULES:
    ===== MANDATORY EXTRACTION REQUIREMENT =====
    1. EXTRACT ALL PARAMETERS DEFINED IN THE JSON SCHEMA.
    2. IF VALUE IS EXPLICITLY STATED IN THE DRAWING, USE THAT VALUE.
    3. IF VALUE IS NOT EXPLICITLY STATED, YOU MUST USE YOUR 50 YEARS OF ENGINEERING KNOWLEDGE AND THE PROVIDED INFERENCE/CALCULATION RULES TO DETERMINE THE MOST ACCURATE VALUE.
    4. NEVER LEAVE ANY PARAMETER AS "NA" UNLESS IT IS TRULY UNINFERABLE AND MEANINGLESS. ALWAYS PROVIDE A VALUE, EVEN IF INFERRED.
    5. BE FLEXIBLE WITH PARAMETER NAME MATCHING - IF A NAME MATCHES 90% OR MORE, ACCEPT IT.
    ===== FLEXIBLE PARAMETER NAME MATCHING RULES =====
    6. ACCEPT THESE AS EQUIVALENTS (90%+ match):
       - "BORE:" = "BORE DIAMETER"
       - "ID:" = "BORE DIAMETER"
       - "OD:" = "OUTSIDE DIAMETER"
       - "OUTER DIA:" = "OUTSIDE DIAMETER"
       - "ROD:" = "ROD DIAMETER"
       - "RD:" = "ROD DIAMETER"
       - "STROKE:" = "STROKE LENGTH"
       - "S.L." = "STROKE LENGTH"
       - "CLOSE:" = "CLOSE LENGTH"
       - "OPEN:" = "OPEN LENGTH"
       - "PRESSURE:" = "OPERATING PRESSURE"
       - "TEMP:" = "OPERATING TEMPERATURE"
       - "DWG NO:" = "DRAWING NUMBER"
       - "DRG NO:" = "DRAWING NUMBER"
       - "PART NO:" = "DRAWING NUMBER"
       - "FLUID:" = "FLUID"
       - "MEDIUM:" = "FLUID"
       - "MOUNTING:" = "MOUNTING"
       - "ACTION:" = "CYLINDER ACTION"
       - "BODY:" = "BODY MATERIAL"
       - "PISTON:" = "PISTON MATERIAL"
       - "STANDARD:" = "STANDARD"
       - "SURFACE:" = "SURFACE FINISH"
       - "COATING:" = "COATING THICKNESS"
       - "SPECIAL:" = "SPECIAL FEATURES"
       - "CONFIG:" = "CYLINDER CONFIGURATION"
       - "STYLE:" = "CYLINDER STYLE"
       - "CONCENTRICITY:" = "CONCENTRICITY OF ROD AND TUBE"
    ===== CRITICAL PARAMETERS TO EXTRACT (AND INFER IF NECESSARY) =====
    7. BORE DIAMETER - Look for "BORE:", "ID:", "Ø" near cylinder barrel. If not explicit, infer based on typical cylinder sizes, piston diameter, or ratios with the cylinder's overall outside dimensions. For example, if a piston diameter is given, the bore is typically the same. If tube OD is given, infer bore based on standard wall thicknesses.
    8. STROKE LENGTH - Look for "STROKE:", "S.L." or calculate from open/close positions.
    9. CLOSE LENGTH - Fully retracted position dimension. If not explicit, calculate as OPEN LENGTH - STROKE LENGTH. If only one of CLOSE/OPEN is given with STROKE, infer the other.
    10. OPEN LENGTH - Fully extended position dimension. If not explicit, calculate as CLOSE LENGTH + STROKE LENGTH. If only one of CLOSE/OPEN is given with STROKE, infer the other.
    11. ROD DIAMETER - Look for "ROD:", "RD", "Ø" near piston rod.
    12. OUTSIDE DIAMETER - Look for "OD:", "OUTER DIA:". If not explicit, infer based on typical cylinder construction (e.g., Bore + standard wall thickness + clearance) or by identifying the largest outer diameter of the cylinder barrel. Consider common industry practices for cylinder tube dimensions.
    13. OPERATING PRESSURE - Look for "PRESSURE:", "BAR", "MPa", "kg/cm²". If not explicit, infer based on typical hydraulic/pneumatic system pressures (e.g., >20 BAR/2 MPa typically hydraulic, <=10 BAR/1 MPa typically pneumatic).
    14. OPERATING TEMPERATURE - Look for "TEMP:", "TEMPERATURE:". If not explicit, infer based on typical operating environments (e.g., 80°C for hydraulic, 60°C for pneumatic).
    15. DRAWING NUMBER - Look in title block for "DWG NO:", "DRG NO:", "PART NO:".
    16. FLUID - Look for "FLUID:", "OIL:", "AIR:". If not explicit, infer based on OPERATING PRESSURE or the overall construction (e.g., heavy-duty implies hydraulic, lighter construction implies pneumatic). Apply the strict conversion rules.
    17. BODY MATERIAL - Look for material callouts on the cylinder tube or covers. If hydraulic, infer "Carbon Steel" or "M.S." (Mild Steel). If pneumatic, infer "Aluminum" or "Stainless Steel".
    18. PISTON MATERIAL - Look for material callouts on the piston. If hydraulic, infer "Cast Iron" or "Ductile Iron". If pneumatic, infer "Aluminum" or "Acetal".
    19. MOUNTING - Identify visually: CLEVIS, FLANGE, LUG, TRUNNION, ROD EYE. Also check for text labels.
    20. ROD END - Identify visually: THREAD, CLEVIS, ROD EYE. Also check for text labels.
    21. CYLINDER ACTION - Check ports: 2 ports = DOUBLE ACTING, 1 port = SINGLE ACTING. If not explicit, infer 'DOUBLE ACTING' if two ports are visible or if it's a hydraulic cylinder. Infer 'SINGLE ACTING' if only one port is visible and it's a pneumatic cylinder.
    22. CYLINDER CONFIGURATION - Default to "Standard" unless specific design elements like "Telescopic", "Compact", "Mill Duty" are evident.
    23. CYLINDER STYLE - If hydraulic, infer "Tie Rod" or "Welded". If pneumatic, infer "Tie Rod" or "Compact".
    24. RATED LOAD - Look for explicit pulling/pushing force values (kN, N). If not explicit, calculate as (BORE DIAMETER² × π/4) × OPERATING PRESSURE. Use the higher of pushing/pulling force if both are given.
    25. STANDARD - Look for references to ISO, DIN, NFPA, JIS, etc. Infer common standards (e.g., ISO 6020/6022 for hydraulic, ISO 15552 for pneumatic) if not explicit.
    26. SURFACE FINISH - Look for Ra values or descriptions like "HONED", "HARD CHROMIUM PLATED", "ALUMITE TREATMENT". Infer typical finishes (e.g., "Ra 0.4" for hydraulic rod/bore, "Ra 0.8" for pneumatic) if not explicit.
    27. COATING THICKNESS - Look for plating or coating specifications (e.g., "25 THICK MIN", "20 M", "110µm"). Infer typical coatings (e.g., "25 micron Chrome" for hydraulic rod, "15 micron Anodize" for pneumatic body) if not explicit.
    28. SPECIAL FEATURES - Any unique design elements or notes. Look for notes on cushioning, air bleeders, stroke measuring, internal treatments, etc. If not explicit, infer 'Cushioned' if cushioning mechanism is visible.
    29. CONCENTRICITY OF ROD AND TUBE - Look for tolerance values. If not explicit, infer a typical precision tolerance like '0.05 mm TIR'.
    ===== EXTRACTION AND INFERENCE STRATEGY =====
    30. First, scan specification or dimension tables for labeled values (HIGHEST PRIORITY).
    31. Then, analyze callouts, arrows, and labeled dimensions near the drawing.
    32. Analyze the title block for drawing number, standards, revisions, etc.
    33. Search notes or side remarks for pressure, temperature, features.
    34. Identify features using geometric shape recognition (e.g., mounting/rod end).
    35. Use OCR reasoning to interpret faint, rotated, or low-contrast text.
    36. Interpret units properly: mm, bar, MPa, °C, psi, inches, kN, N, Kg/cm².
    37. Do not convert units unless explicitly asked.
    38. Do not estimate values by scaling the drawing.
    39. Use engineering context and the provided calculation rules to make logical inferences where appropriate.
    ===== VISUAL INFERENCE RULES =====
    40. CLEVIS → Forked U-shape with pin hole
    41. FLANGE → Flat disc or ring with bolt holes
    42. LUG → Side-mounted brackets
    43. TRUNNION → Cylindrical pin through middle of barrel
    44. ROD END - CLEVIS → Forked tip
    45. ROD END - THREAD → Threaded shaft
    46. ROD END - ROD EYE → Loop with hole
    ===== FLUID HANDLING RULES (STRICT) =====
    47. "Mineral Oil" → FLUID = HYD. OIL MINERAL
    48. "HLP68", "ISO VG46", "Synthetic Oil" → Keep as written
    49. "Compressed Air", "Pneumatic", "AIR" → FLUID = AIR
    50. If fluid is not specified but it's clearly a hydraulic cylinder (e.g., high pressure, robust construction), infer "HYD. OIL MINERAL". If pneumatic, infer "AIR".
    ===== CALCULATION AND INFERENCE RULES WHEN VALUES MISSING =====
    51. OPEN LENGTH = CLOSE LENGTH + STROKE LENGTH (if both available). If only one is available, infer the other based on typical ratios or common cylinder series.
    52. OUTSIDE DIAMETER = BORE DIAMETER + (2 × WALL THICKNESS) + (2 × CLEARANCE). If wall thickness/clearance not given, use a typical safety margin (e.g., BORE DIAMETER + 15mm to 30mm depending on bore size).
    53. RATED LOAD = (BORE² × π/4) × OPERATING PRESSURE.
    54. OPERATING PRESSURE = 160 BAR (default for hydraulic), 10 BAR (default for pneumatic). Adjust based on visual cues (e.g., heavy duty construction implies higher pressure).
    55. OPERATING TEMPERATURE = 80°C (default for hydraulic), 60°C (default for pneumatic).
    56. BODY MATERIAL: If hydraulic, infer "Carbon Steel" or "M.S." (Mild Steel). If pneumatic, infer "Aluminum" or "Stainless Steel".
    57. PISTON MATERIAL: If hydraulic, infer "Cast Iron" or "Ductile Iron". If pneumatic, infer "Aluminum" or "Acetal".
    58. CYLINDER ACTION: If two ports are visible, infer "DOUBLE ACTING". If one port, infer "SINGLE ACTING". If no ports visible, infer "DOUBLE ACTING" as it's more common.
    59. CYLINDER CONFIGURATION: Default to "Standard" unless specific features suggest otherwise (e.g., "Telescopic", "Compact").
    60. CYLINDER STYLE: If hydraulic, infer "Tie Rod" or "Welded". If pneumatic, infer "Tie Rod" or "Compact".
    61. STANDARD: Infer "ISO 6020/6022" for hydraulic, "ISO 15552" for pneumatic, if no standard is explicitly mentioned.
    62. SURFACE FINISH: Infer "Ra 0.4" for hydraulic rod/bore, "Ra 0.8" for pneumatic.
    63. COATING THICKNESS: Infer "25 micron Chrome" for hydraulic rod, "15 micron Anodize" for pneumatic body.
    64. CONCENTRICITY OF ROD AND TUBE: Infer "0.05 mm TIR" (Total Indicator Runout) as a typical precision.
    ===== OUTPUT FORMAT (EXACT MATCH REQUIRED) =====
    65. YOU MUST RESPOND SOLELY WITH A JSON OBJECT THAT STRICTLY ADHERES TO THE FOLLOWING JSON SCHEMA.
    66. DO NOT INCLUDE ANY OTHER TEXT, EXPLANATIONS, OR MARKDOWN OUTSIDE THE JSON.
    67. ALL PROPERTIES IN THE SCHEMA MUST BE PRESENT IN THE OUTPUT JSON.
    68. IF A VALUE IS INFERRED, PROVIDE THE INFERRED VALUE. DO NOT USE "NA" OR EMPTY STRINGS.
    JSON SCHEMA:
    {json.dumps(json_schema, indent=2)}
    NOW ANALYZE THIS CYLINDER DRAWING AND EXTRACT ALL PARAMETERS INTO THE JSON OBJECT, INFERRING WHERE NECESSARY.
    """
    
    if not client:
        return {"error": "Azure OpenAI client not initialized"}
    
    try:
        print(f"Processing {filename}...")
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content},
                        {"type": "image_url", "image_url": {"url": base64_image, "detail": "high"}}
                    ]
                }
            ],
            max_tokens=3000,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        try:
            extracted_data = json.loads(content)
            print(f"Successfully extracted data for {filename}.")
            return extracted_data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from API response for {filename}. Raw content: {content[:500]}...")
            return {"error": "JSON decoding failed", "raw_response": content}
    except Exception as e:
        print(f"API request failed for {filename}: {e}")
        return {"error": str(e)}

def main():
    """Main function to process a directory of PDF files."""
    # ✅ Use your actual dataset folder
    pdf_directory = r"D:\ratn\Sereno Volante Private Limited\combine_model\data"
    # ✅ Automatically list all PDF files in the directory
    pdf_files_to_process = [
        os.path.join(pdf_directory, f)
        for f in os.listdir(pdf_directory)
        if f.lower().endswith('.pdf')
    ]
    all_extracted_data = []
    for pdf_path in pdf_files_to_process:
        try:
            print(f"Attempting to fetch {pdf_path}...")
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            image_bytes = convert_pdf_to_image_bytes(pdf_bytes)
            if image_bytes:
                extracted_data = analyze_engineering_drawing(image_bytes, os.path.basename(pdf_path))
                all_extracted_data.append({"filename": os.path.basename(pdf_path), "data": extracted_data})
            else:
                all_extracted_data.append({"filename": os.path.basename(pdf_path), "data": {"error": "PDF to image conversion failed"}})
        except FileNotFoundError:
            print(f"Error: File not found at {pdf_path}. Please ensure the file exists.")
            all_extracted_data.append({"filename": os.path.basename(pdf_path), "data": {"error": "File not found"}})
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")
            all_extracted_data.append({"filename": os.path.basename(pdf_path), "data": {"error": str(e)}})
    
    # ✅ PRINT RESULTS
    print("\n--- All Extracted Data ---")
    for item in all_extracted_data:
        print(f"\nFilename: {item['filename']}")
        print(json.dumps(item['data'], indent=2))
    
    # ✅ SAVE TO JSON FILE (optional)
    with open("extracted_data.json", "w") as f:
        json.dump(all_extracted_data, f, indent=2)
    
    # ✅ SAVE TO EXCEL FILE
    # Convert all_extracted_data into a flat list of rows
    records = []
    for item in all_extracted_data:
        row = {"filename": item["filename"]}
        if isinstance(item["data"], dict):
            row.update(item["data"])
        else:
            row.update({"error": "No data"})
        records.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to Excel
    excel_path = "extracted_data.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\n✅ All data saved to {excel_path}")

if __name__ == "__main__":
    main()
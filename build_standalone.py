import re

files_to_bundle = [
    "config.py",
    "utils.py",
    "preprocessing.py",
    "alignment.py",
    "difference.py",
    "segmentation.py",
    "validation.py",
    "detector.py",
]

all_imports = set()
all_code = []

# standard and third-party modules we might see
local_modules = ["config", "utils", "preprocessing", "alignment", "difference", "segmentation", "validation", "detector", "api_models", "routers"]

import_pattern = re.compile(r'^(?:from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.*)|import\s+(.*))$')

for file_name in files_to_bundle:
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    code_lines = []
    in_local_import = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip FastAPI imports since standalone won't use it
        if "fastapi" in stripped or "api_models" in stripped or "routers" in stripped:
            continue
            
        m = import_pattern.match(stripped)
        if m:
            from_mod, from_names, direct_mod = m.groups()
            
            # Check if it's a local module
            is_local = False
            if from_mod:
                base_mod = from_mod.split('.')[0]
                if base_mod in local_modules:
                    is_local = True
            elif direct_mod:
                base_mod = direct_mod.split('.')[0]
                if base_mod in local_modules:
                    is_local = True
                    
            if not is_local:
                all_imports.add(stripped)
            continue
            
        # Also remove multi-line imports if any (basic heuristic)
        if stripped.startswith("from ") and "(" in stripped and not ")" in stripped:
            base_mod = stripped.split(" ")[1].split(".")[0]
            if base_mod in local_modules:
                in_local_import = True
            continue
            
        if in_local_import:
            if ")" in stripped:
                in_local_import = False
            continue
            
        # Skip logger lines to avoid duplicate loggers, or just let them be (they will overwrite)
        if "logger = get_logger(" in line and file_name != "utils.py":
            line = f'logger = logging.getLogger("Standalone")\n'
            
        code_lines.append(line)
        
    all_code.append(f"\n\n# {'='*60}\n# Extracted from {file_name}\n# {'='*60}\n")
    all_code.append("".join(code_lines))

# Now add the wrapper code
wrapper_code = """

# ============================================================
# Standalone Inference Wrapper
# ============================================================
import base64

def decode_base64_image(b64_string: str) -> np.ndarray:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

def encode_image_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def detect_anomalies(ref_b64: str, curr_b64: str) -> dict:
    ref_img = decode_base64_image(ref_b64)
    curr_img = decode_base64_image(curr_b64)

    if ref_img is None or curr_img is None:
        return {"error": "Failed to decode input images."}

    detector = FODDetector(DEFAULT_CONFIG)
    
    ts = int(time.time() * 1000)
    prefix = f"standalone_{ts}"

    result = detector.detect(
        ref_images=[ref_img],
        curr_image=curr_img,
        ignore_mask=None,
        output_prefix=prefix,
    )

    out_path = OUTPUT_DIR / Path(result["output_image"]).name
    if out_path.exists():
        out_img = cv2.imread(str(out_path))
        if out_img is not None:
            result["output_image_base64"] = encode_image_base64(out_img)
    else:
        result["output_image_base64"] = None

    return result
"""

final_script = []
for imp in sorted(list(all_imports)):
    final_script.append(imp + "\n")

final_script.append("\n")
for code_part in all_code:
    final_script.append(code_part)
    
final_script.append(wrapper_code)

with open("standalone_full_inference.py", "w", encoding="utf-8") as f:
    f.writelines(final_script)

print("Generated standalone_full_inference.py")

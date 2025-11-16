#git clone https://github.com/jkoppel/QuixBugs.git

import os
from typing import Dict, List

# --- 설정 (로컬 클론 경로에 맞게 수정하세요) ---
# QuixBugs 저장소를 클론한 경로를 지정해야 합니다.
QUIXBUGS_ROOT_DIR = "c:/Users/user/QuixBugs" # git clone한 QuixBug 저장소의 Path

# 추출 대상 서브 디렉토리
TARGET_DIRS = ["python_testcases", "python_programs"]
# -------------------------------------------------------------


def extract_python_code_from_dirs(root_dir: str, target_dirs: List[str]) -> Dict[str, Dict[str, str]]:
    """
    주어진 루트 디렉토리 내의 대상 디렉토리에서 모든 .py 파일의 코드를 추출합니다.
    (저장 로직은 문자열만 저장하도록 유지되어야 합니다.)
    """
    extracted_data: Dict[str, Dict[str, str]] = {}

    if not os.path.isdir(root_dir):
        print(f"❌ Error: Root directory not found at '{root_dir}'. Please clone the QuixBugs repo first.")
        return extracted_data

    for target_dir_name in target_dirs:
        target_path = os.path.join(root_dir, target_dir_name)
        
        if not os.path.isdir(target_path):
            print(f"⚠️ Warning: Target directory '{target_path}' not found. Skipping.")
            continue

        file_contents: Dict[str, str] = {}
        
        print(f"✅ Extracting code from: {target_dir_name}")
        
        for dirpath, _, filenames in os.walk(target_path):
            for filename in filenames:
                if filename.endswith(".py"):
                    full_path = os.path.join(dirpath, filename)
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            code_content = f.read()
                            # 파일 내용은 반드시 문자열로 저장합니다.
                            file_contents[filename] = code_content
                            
                    except Exception as e:
                        print(f"❌ Error reading file {full_path}: {e}")

        extracted_data[target_dir_name] = file_contents
        print(f"   -> Successfully extracted {len(file_contents)} files.")
        
    return extracted_data

def safe_extract_and_print_code(data_dict: Dict[str, str], dict_name: str):
    """
    딕셔너리에서 첫 번째 항목을 안전하게 추출하여 출력합니다.
    튜플 객체가 들어와도 첫 번째 요소를 시도하여 오류를 방지합니다.
    """
    if not data_dict:
        print(f"### 🧪 {dict_name} 데이터 없음")
        return

    # 첫 번째 파일 이름 추출
    first_file = next(iter(data_dict.keys()), None)
    if not first_file:
        return

    raw_content = data_dict[first_file]
    code_content = None

    try:
        # 1. 문자열인 경우: 그대로 사용
        if isinstance(raw_content, str):
            code_content = raw_content
        # 2. 튜플인 경우: 첫 번째 요소를 사용 (오류의 원인이었음)
        elif isinstance(raw_content, tuple) and raw_content and isinstance(raw_content[0], str):
            code_content = raw_content[0]
        # 3. 그 외의 경우: 출력할 수 없음을 알림
        else:
            print(f"⚠️ Warning: {first_file}의 내용이 예상치 못한 형식입니다: {type(raw_content)}")
            return
            
        print(f"\n### 🧪 {dict_name} (Example: {first_file})")
        # 300자까지 자르고 .strip()을 적용하여 출력
        print(code_content[:300].strip() + "\n[... 후략 ...]")

    except Exception as e:
        print(f"❌ Error processing content for {first_file}: {e}")


# --- 코드 추출 실행 ---
extracted_code = extract_python_code_from_dirs(QUIXBUGS_ROOT_DIR, TARGET_DIRS)

## 추출된 데이터 확인

print("\n--- Extracted Data Summary ---")

if extracted_code:
    # python_testcases 데이터 안전하게 출력
    testcases = extracted_code.get("python_testcases", {})
    safe_extract_and_print_code(testcases, "Python Testcases")

    # python_programs 데이터 안전하게 출력
    programs = extracted_code.get("python_programs", {})
    safe_extract_and_print_code(programs, "Python Programs")
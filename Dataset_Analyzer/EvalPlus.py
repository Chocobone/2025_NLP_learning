# pip install evalplus

import json
from typing import Dict, Any
from evalplus.data import get_human_eval_plus

def extract_all_human_eval_data() -> Dict[str, Dict[str, Any]]:
    """
    EvalPlus 패키지를 사용하여 HumanEval 원본 데이터셋의 모든 문제를 추출합니다.
    """
    try:
        # HumanEval 원본 데이터셋 로드
        # 결과는 {'HumanEval/0': {...}, 'HumanEval/1': {...}, ...} 형태입니다.
        problems: Dict[str, Dict[str, Any]] = get_human_eval_plus()
    except Exception as e:
        print(f"❌ Error: HumanEval 데이터 로드 실패. 'evalplus' 패키지가 설치되어 있는지 확인하십시오. ({e})")
        return {}

    num_problems = len(problems)
    print(f"✅ 총 {num_problems}개의 HumanEval 원본 문제 데이터 추출 완료.")
    print("---")

    # 예시로 첫 번째 문제의 데이터를 출력하여 구조를 확인합니다.
    example_task_id = "HumanEval/0"
    if example_task_id in problems:
        problem_data = problems[example_task_id]
        
        print(f"### 🧩 Task ID: {example_task_id} 데이터 구조")
        print(f"1. Prompt (문제 및 시그니처):\n{problem_data['prompt'][:100].strip()}...\n")
        print(f"2. Canonical Solution (정답 코드):\n{problem_data['canonical_solution'][:100].strip()}...\n")
        print(f"3. Entry Point (함수 이름): {problem_data['entry_point']}\n")
        print(f"4. Test Code (원본 기본 테스트):\n{problem_data['test'][:100].strip()}...\n")
        
    print("---")
    print("➡️ 모든 HumanEval 문제가 딕셔너리 형태로 반환됩니다.")
    
    return problems

def extract_all_evalplus_test_codes() -> Dict[str, str]:
    """
    EvalPlus HumanEval+ 데이터셋에서 모든 문제의 Task ID와 Full Test Code를 추출합니다.
    """
    try:
        # HumanEval+ 데이터셋 로드
        # 결과는 {'HumanEval/0': {...}, 'HumanEval/1': {...}, ...} 형태입니다.
        problems: Dict[str, Dict[str, Any]] = get_human_eval_plus()
    except Exception as e:
        print(f"❌ Error: EvalPlus 데이터 로드 실패. 'evalplus' 패키지가 설치되어 있는지 확인하십시오. ({e})")
        return {}

    all_test_codes: Dict[str, str] = {}

    print(f"✅ 총 {len(problems)}개의 HumanEval+ 문제 테스트 코드 추출 시작.")
    print("---")

    for task_id, problem_data in problems.items():
        # 1. 'test' 필드에 포함된 전체 테스트 코드를 추출합니다.
        # 이 문자열이 HumanEval의 기본 테스트와 EvalPlus의 추가(LLM/뮤테이션 생성) 테스트를 모두 포함합니다.
        test_code_full = problem_data.get('test', 'N/A')
        
        # 2. 결과 딕셔너리에 저장
        all_test_codes[task_id] = test_code_full
        
        # 3. 예시 출력 (첫 3개만 간결하게 출력)
        if len(all_test_codes) <= 3:
            print(f"### 🧪 Task ID: {task_id}")
            # 테스트 코드가 길므로 앞부분만 출력합니다.
            print(f"Test Code Snippet:\n{test_code_full[:200].strip()}...\n")
            
    print("---")
    print(f"➡️ 모든 {len(all_test_codes)}개 문제의 테스트 코드 추출 완료.")
    
    return all_test_codes

# HumanEval/0 문제에 대한 데이터 추출 실행
extracted_human_codes = extract_all_human_eval_data()
# 함수 실행
extracted_test_codes = extract_all_evalplus_test_codes()

# 모든 테스트 코드가 포함된 딕셔너리를 파일로 저장하고 싶다면 (선택 사항)
# with open('evalplus_test_codes.json', 'w', encoding='utf-8') as f:
#     json.dump(extracted_test_codes, f, indent=4)
# print("\n데이터가 'evalplus_test_codes.json' 파일에 저장되었습니다.")
# ----------------------------
# 표준 라이브러리
# ----------------------------
import os
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from itertools import combinations
from typing import List, Dict

# ----------------------------
# 외부 라이브러리
# ----------------------------
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------
# 1. 모델과 토크나이저를 한 번만 로드하도록 전역 변수로 설정
# ----------------------------------------------------
print("Loading model and tokenizer...")
# MODEL_ID를 Qwen Coder AWQ 모델로 변경
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ" 
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",        # AWQ 모델도 auto로 설정하면 됩니다.
    trust_remote_code=True,  # Qwen 모델 실행 시 필요합니다.
    device_map="auto"        # GPU VRAM에 맞게 모델을 자동 배치합니다.
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
print("Model and tokenizer loaded successfully! 🚀")

# ----------------------------------------------------
# 2. LLM 호출을 위한 재사용 가능한 함수 정의
# ----------------------------------------------------

def extract_code(text: str) -> str:
    """
    모델의 전체 응답 텍스트에서 마지막 C++ 코드 블록만 추출합니다.
    """
    if "<think>" in text:
        print("[Debug] '<think>' tag found. Skipping markdown code block search.")
        start_match = re.search(r"#include", text)
        if not start_match:
            print("[Warning] No '#include' found.")
            return text

        start_index = start_match.start()
        think_match = re.search(r"<think>", text)
        search_area_end_index = think_match.start()

        search_area = text[start_index:search_area_end_index]
        last_brace_index_in_area = search_area.rfind("}")

        if last_brace_index_in_area != -1:
            print("[Debug] Extracted code from '#include' to last '}' before '<think>'.")
            return search_area[:last_brace_index_in_area + 1].strip()
        
        print("[Warning] Could not find closing '}' before <think>.")
        return search_area.strip()

    pattern = r"```(?:cpp)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        print(f"[Debug] Found {len(matches)} code blocks. Extracting the last one.")
        return matches[-1].strip()

    print("[Warning] No code block found at all.")
    return text


def generate_code_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 3500
) -> str:
    """
    코드 생성 전용: 주어진 메시지를 기반으로 LLM의 응답을 생성하고 '코드'만 추출합니다.
    do_sample=False로 설정하여 결정론적 생성 (temperature 무시됨)
    """
    with torch.no_grad():
        inputs = TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(MODEL.device)

        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # ✅ 결정론적 생성 - temperature 무시됨
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id
        )

        response_ids = outputs[:, inputs["input_ids"].shape[-1]:]
        full_response_text = TOKENIZER.batch_decode(response_ids, skip_special_tokens=True)[0]
    
    torch.cuda.empty_cache()
    extracted_code = extract_code(full_response_text)
    
    return extracted_code


def generate_text_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512
) -> str:
    """
    일반 텍스트 생성 전용: 테스트 입력 등을 생성할 때 사용.
    코드 추출 없이 원본 응답을 그대로 반환합니다.
    do_sample=False로 설정하여 결정론적 생성
    """
    with torch.no_grad():
        inputs = TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(MODEL.device)

        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # ✅ 결정론적 생성
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id
        )

        response_ids = outputs[:, inputs["input_ids"].shape[-1]:]
        full_response_text = TOKENIZER.batch_decode(response_ids, skip_special_tokens=True)[0]
    
    torch.cuda.empty_cache()
    
    return full_response_text


def _read_text_if_exists(path: Path | None) -> str:
    """파일이 존재하면 텍스트를 읽고, 없으면 빈 문자열을 반환합니다."""
    if path and path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[Warning] Could not read file {path}: {e}")
            return ""
    return ""


def load_trickybugs_data_revised(base_path: str, lang: str = None) -> Dict:
    """
    TrickyBugs 파일 구조에 맞춰 데이터를 로드합니다.
    """
    print(f"Loading TrickyBugs data...")

    base = Path(base_path)
    problems_dir = base / "problems"

    if not problems_dir.exists():
        raise FileNotFoundError(f"Problems directory not found at: {problems_dir}")

    loaded_problems = {}
    pid_dirs = sorted([p for p in problems_dir.iterdir() if p.is_dir()])

    for pid_dir in pid_dirs:
        pid = pid_dir.name
        buggy_base = pid_dir / "buggy_programs"
        if not buggy_base.exists():
            print(f"[Warning] No buggy_programs for {pid}")
            continue

        if lang:
            selected_lang_dir = buggy_base / lang
            if not selected_lang_dir.exists():
                print(f"[Warning] No buggy_programs/{lang} for {pid}")
                continue
        else:
            lang_dirs = [d for d in buggy_base.iterdir() if d.is_dir()]
            if not lang_dirs:
                print(f"[Warning] No language directories in buggy_programs for {pid}")
                continue
            selected_lang_dir = lang_dirs[0]
            lang = selected_lang_dir.name

        source_files = list(selected_lang_dir.glob(f"*.{lang}"))
        if not source_files:
            print(f"[Warning] No source files found in: {selected_lang_dir}")
            continue

        put_path = source_files[0]
        put_code = _read_text_if_exists(put_path)
        if not put_code:
            continue

        spec_path = pid_dir / "problem_description.txt"
        spec_text = _read_text_if_exists(spec_path)

        meta_path = pid_dir / "metainfo.json"
        meta_data = {}
        if meta_path.exists():
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[Warning] Failed to read {meta_path}: {e}")

        loaded_problems[pid] = {
            "language": lang,
            "spec": spec_text,
            "put_code": put_code,
            "meta": {
                "pid_path": str(pid_dir),
                "put_path": str(put_path),
                "metainfo": meta_data
            }
        }

    print(f"✅ TrickyBugs 로딩 완료: {len(loaded_problems)}개 문제")
    return loaded_problems


def create_variants(problems, output_base_path, k: int = 1):
    """
    TrickyBugs 문제 데이터에서 각 코드의 수정 버전을 생성합니다.
    """
    genprog_prompt = """You are a professional coding competition participant, skilled at identifying bugs and logic flaws in code.
You will receive a description of a coding problem, and a piece of code attempting to solve the problem.
Your task is to repair the code.

Your response MUST contain only the complete C++ code, formatted inside a single markdown code block.
Start your response IMMEDIATELY with ```cpp and end your response IMMEDIATELY with ```.
Do not provide any text, explanation, or reasoning before or after the code block.

**PROBLEM DESCRIPTION**:
{pro_des}

**CODE**:
{code}
"""

    for pid, problem in problems.items():
        print(f"\n--- Processing PID: {pid} ---")
        variant_dir = Path(output_base_path) / "GenProgs" / pid
        variant_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, k + 1):
            print(f"🤖 Generating repaired program variant #{i}...")

            prog_messages = [
                {
                    "role": "user",
                    "content": genprog_prompt.format(
                        pro_des=problem["spec"],
                        code=problem["put_code"]
                    )
                }
            ]

            # ✅ 수정: generate_code_response 직접 호출, 1280 토큰 사용
            variant_code = generate_code_response(prog_messages, max_new_tokens=1280)

            variant_file = variant_dir / f"variant_{i}.{problem['language']}"
            with open(variant_file, "w", encoding="utf-8") as f:
                f.write(variant_code)

            print(f"   ✅ Saved repaired code to: {variant_file}")

    print(f"\n🎯 모든 문제의 {k}개 변형 코드 생성 완료!")


def generate_buggy_test_inputs(problems, output_base_path, num_inputs: int = 10):
    """
    LLM에게 각 문제 명세와 버그 코드를 전달하여 테스트 입력을 생성합니다.
    """
    output_base = Path(output_base_path) / "chat_generated_inputs"
    output_base.mkdir(parents=True, exist_ok=True)

    for pid, problem in problems.items():
        print(f"\n--- Generating test inputs for PID: {pid} ---")
        
        pid_dir = output_base / pid
        pid_dir.mkdir(parents=True, exist_ok=True)
        
        prompt = f"""**INSTRUCTION**:
You are a professional software testing engineer. You will get a problem description of a coding problem, and a piece of code attempting to solve the problem. 
Please generate {num_inputs} diverse and corner test inputs that could potentially trigger bugs.
Every input must adhere to the constraints and format mentioned in the problem description.
Please reply with ONLY the generated input without any other content, use the following template:
INPUT1:
(content of the 1st generated test input)
INPUT2:
(content of the 2nd generated test input)
...
INPUT{num_inputs}:
(content of the {num_inputs}-th generated test input)

**PROBLEM DESCRIPTION**:
{problem["spec"]}

**CODE**:
{problem["put_code"]}
"""
        messages = [{"role": "user", "content": prompt}]
        
        # ✅ 수정: generate_text_response 직접 호출, 512 토큰 사용
        response = generate_text_response(messages, max_new_tokens=512)

        pattern = r"INPUT\d+:\s*(.*?)\s*(?=INPUT\d+:|$)"
        matches = re.findall(pattern, response, flags=re.DOTALL)

        if not matches:
            print(f"  ❌ No inputs generated for PID {pid}")
            continue

        for idx, input_content in enumerate(matches[:num_inputs], start=1):
            input_file = pid_dir / f"chatGenInput_{idx}.in"
            input_file.write_text(input_content.strip(), encoding="utf-8")
            print(f"  ✅ Saved {input_file.name}")

    print("\n🎯 모든 문제의 테스트 입력 생성 완료!")


def verify_variants(problems, output_base_path):
    """
    생성된 variant 코드 중 원본 테스트 케이스를 모두 통과하는 코드만 저장합니다.
    """
    genprogs_base = Path(output_base_path) / "GenProgs"
    verified_base = Path(output_base_path) / "GenProgsVerified"
    verified_base.mkdir(parents=True, exist_ok=True)

    for pid, problem in problems.items():
        lang = problem["language"]
        print(f"\n--- Verifying PID: {pid} ({lang}) ---")
        
        variant_dir = genprogs_base / pid
        if not variant_dir.exists():
            print(f"[Warning] No variants found for {pid}")
            continue

        test_dir = Path(problem["meta"]["pid_path"]) / "original_test_cases"
        if not test_dir.exists():
            print(f"[Warning] No original_test_cases for {pid}")
            continue

        variant_files = list(variant_dir.glob(f"*.{lang}"))
        for variant_file in variant_files:
            print(f"Checking {variant_file.name} ...")
            all_passed = True

            if lang == "cpp":
                exec_file = variant_file.parent / "tmp_exec"
                try:
                    compile_cmd = ["g++", "-std=c++17", str(variant_file), "-o", str(exec_file)]
                    subprocess.run(compile_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Compile failed: {e}")
                    all_passed = False
                    continue
            else:
                exec_file = variant_file

            input_files = sorted(test_dir.glob("*.in"))
            for in_file in input_files:
                out_file = test_dir / (in_file.stem + ".out")
                if not out_file.exists():
                    continue

                try:
                    if lang == "cpp":
                        result = subprocess.run([str(exec_file)], input=in_file.read_bytes(),
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                    else:
                        result = subprocess.run(["python3", str(exec_file)], input=in_file.read_bytes(),
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)

                    expected_output = out_file.read_bytes()
                    if result.stdout.strip() != expected_output.strip():
                        all_passed = False
                        print(f"  ❌ Test failed: {in_file.name}")
                        break
                except subprocess.TimeoutExpired:
                    all_passed = False
                    print(f"  ❌ Test timed out: {in_file.name}")
                    break

            if all_passed:
                pid_verified_dir = verified_base / pid
                pid_verified_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(variant_file, pid_verified_dir / variant_file.name)
                print(f"  ✅ Passed all tests: {variant_file.name}")

            if lang == "cpp" and exec_file.exists():
                exec_file.unlink()

    print("\n🎯 모든 문제의 Verified variants 처리 완료!")


def run_code(lang:str, code_path:Path, input_bytes:bytes, timeout:int=5) -> bytes:
    """C++ 또는 Python 코드를 실행하고 stdout 반환"""
    if lang == "cpp":
        exec_file = code_path.parent / "tmp_exec"
        try:
            subprocess.run(["g++", "-std=c++17", str(code_path), "-o", str(exec_file)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[Compile Error] {code_path}: {e}")
            return b""
        try:
            result = subprocess.run([str(exec_file)], input=input_bytes,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            return b"TIMEOUT"
        finally:
            if exec_file.exists():
                exec_file.unlink()
        return result.stdout
    elif lang == "python":
        try:
            result = subprocess.run(["python3", str(code_path)], input=input_bytes,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            return b"TIMEOUT"
        return result.stdout
    else:
        raise ValueError(f"Unsupported language: {lang}")


def task_oracle(problems:Dict, inputs_base_path:str, lang:str):
    """
    각 문제의 PUT과 reference 솔루션에 대해 LLM이 생성한 테스트 입력을 실행하고
    출력 비교 후 트리거 데이터프레임 생성
    
    Returns:
        pd.DataFrame(columns=['pid', 'input_name', 'sol_name', 'out', 'input_valid', 'out_correct'])
    """
    df_rows = []
    inputs_base = Path(inputs_base_path)
    
    for pid, problem in problems.items():
        pid_inputs_dir = inputs_base / pid
        if not pid_inputs_dir.exists():
            print(f"[Warning] No test inputs for {pid}")
            continue
        input_files = sorted(pid_inputs_dir.glob("*.in"))
        
        # 모든 솔루션 (PUT + other) 경로 수집
        sol_files = [(problem['put_code'], 'put')]
        # GenProgsVerified가 있다면 reference 솔루션 추가
        verified_dir = Path(inputs_base_path).parent / "GenProgsVerified" / pid
        if verified_dir.exists():
            for f in verified_dir.glob(f"*.{lang}"):
                sol_files.append((f, f.stem))
        
        for input_file in input_files:
            input_bytes = input_file.read_bytes()
            for code, sol_name in sol_files:
                if isinstance(code, str):
                    # PUT 코드는 임시 파일로 저장
                    tmp_file = Path(f"tmp_put_{pid}.{lang}")
                    tmp_file.write_text(code, encoding="utf-8")
                    code_path = tmp_file
                else:
                    code_path = code  # 이미 파일인 reference
                out = run_code(lang, code_path, input_bytes)
                if isinstance(code, str) and tmp_file.exists():
                    tmp_file.unlink()
                input_valid = out not in [b"", b"TIMEOUT"]
                df_rows.append({
                    "pid": pid,
                    "input_name": input_file.name,
                    "sol_name": sol_name,
                    "out": out,
                    "input_valid": input_valid,
                    "out_correct": input_valid  # PUT 기준으로 판단 가능
                })
    
    df = pd.DataFrame(df_rows)
    return df

combination_cache = {}
def Cnk(n, k):
    if (n, k) in combination_cache:
        return combination_cache[(n, k)]
    result = math.comb(n, k)
    combination_cache[(n, k)] = result
    return result

def get_trigger_df(df:pd.DataFrame,lang:str,method_type:str):
    '''
    method_type: 'dfp' or 'tc'
    lang: 'cpp' or 'python'
    '''
    assert lang in ['cpp','python']
    if method_type!='dfp' and method_type!='tc':
        raise RuntimeError(f"Wrong method_type: {method_type}")

    result_df = pd.DataFrame(columns=['pid', 'input_name','out','sol_names','input_valid','input_valid_byref','out_correct'])

    # A: the PUT
    # B: other programs
    if lang=='cpp':
        A = df[df['sol_name'].str.startswith('sol_')]
        B = df[~df['sol_name'].str.startswith('sol_')]
    elif lang=='python':
        A = df[~df['sol_name'].str.startswith('p0')]
        B = df[df['sol_name'].str.startswith('p0')]
    
    A_outputs = A[['pid', 'input_name', 'out']].drop_duplicates()

    # check B outputs with the same value
    B_grouped = (B.groupby(['pid', 'input_name']))
    totoal_len=len(B_grouped)
    count=0
    for (pid, input_name), group_df in B_grouped:
        count+=1
        if count%1000==0:
            print(f"get_triger: {count}/{totoal_len}")
        group_df=group_df.drop_duplicates(subset=['pid','input_name','sol_name','out'])
        # deduplicate is import when multiple ref_out occur
        # For Example:
        # A[(A['pid']=='p02550') & (A['input_name']=='1_1_1.in.json')]
        # pid	input_name	sol_name	out	is_out_hash	input_valid	number_of_sols	ref_out	is_refout_hash	input_valid_byref	out_correct
        # 121	p02550	1_1_1.in.json	sol_129.out	0	False	0	4	104	False	False	False
        # 122	p02550	1_1_1.in.json	sol_129.out	0	False	0	1	0	False	False	True
        out_values = group_df['out'].values
        unique_out_values = set(out_values)        

        for out_value in unique_out_values:
            
            a_out_df=A_outputs[(A_outputs['pid'] == pid) & (A_outputs['input_name'] == input_name)]
         
            if len(a_out_df)<1 :
                continue
            a_out = A[(A['pid'] == pid) & (A['input_name'] == input_name)]['out'].values[0]
            if out_value==a_out:
                continue
        # get sol_name with the same out
            matching_sol_names = group_df[group_df['out']==out_value]['sol_name'].to_list()
            try:
                input_valid=group_df[group_df['out']==out_value]['input_valid'].to_list()[0]
                input_valid_byref=group_df[group_df['out']==out_value]['input_valid_byref'].to_list()[0]
                out_correct=group_df[group_df['out']==out_value]['out_correct'].to_list()[0]
            except:
                continue
            if method_type=='dfp' and len(matching_sol_names) < 2 :
                continue

            matching_sol_names=tuple(matching_sol_names)
            result_df.loc[len(result_df)] = [pid,input_name,out_value,matching_sol_names,input_valid,input_valid_byref,out_correct]
            
    return result_df


def compute_res_df(df_triger:pd.DataFrame,num_of_ref_progs:int,method_type:str):
    assert(method_type in ['tc','dfp'])
    res_df=pd.DataFrame(columns=['pid','total','TP','FP'])
    ori_n=num_of_ref_progs
    for pid in df_triger['pid'].unique():
        # print(f"{pid} start")
        num_of_ref_progs=ori_n
        df_triger_pid=df_triger[df_triger['pid']==pid].copy()
        sols_set = set()
        df_triger_pid['sol_names'].apply(lambda x: sols_set.update( [sol_name.split('_')[1] for sol_name in x] ))
        sols_list=sorted(list(sols_set))
        total_sols_num=df_triger_pid['total_sols_num'].max()
        if total_sols_num<num_of_ref_progs:
            num_of_ref_progs=total_sols_num
        total=Cnk(total_sols_num,num_of_ref_progs)

        all_sols_name=['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9']
        out_sols_num=total_sols_num-len(sols_list)
        if out_sols_num!=0:
            out_sols_name=set(all_sols_name)-set(sols_list)
            out_sols_name=list(out_sols_name)
            out_sols_name.sort()
            out_sols_list=out_sols_name[:out_sols_num]
        else:
            out_sols_list=[]
        to_use_sols_name=sols_list+out_sols_list

        combos=list(combinations(to_use_sols_name,num_of_ref_progs))
        combos.sort()

        if method_type=='dfp':
            tp,fp=0,0
            # pick num_of_ref_progs sols and check whether they are in the same group
            # compute the average tp and fp among all inputs

            for combo in combos:
                combo=list(combo)
        
                df_triger_pid['combo_all_true']=df_triger_pid[combo].all(axis=1)
                df_tmp=df_triger_pid[df_triger_pid['combo_all_true']]
                if len(df_tmp)<1:
                    continue
                tp+= len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp+= 1-len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                if fp<0:
                    print(f"NOW TP:{tp},FP:{fp}")
                    print(len(df_tmp),len(df_tmp[df_tmp['final_valid']==True]),len(df_tmp[df_tmp['final_valid']==False]))
                    
            if tp==0 and fp==0:
                continue
            res_df.loc[len(res_df)]=[pid,total,tp,fp]
        
        elif method_type=='tc':
            tp,fp=0,0
            for combo in combos:
                
                combo=list(combo)    
                df_triger_pid['combo_any_true']=df_triger_pid[combo].any(axis=1)
                # first find all triger sol groups
                df_tmp=df_triger_pid[df_triger_pid['combo_any_true']].copy()
                if len(df_tmp)<1:
                    continue
                
                df_tmp['sols_in_combo_num'] = df_tmp.loc[:,combo].sum(axis=1)
                max_sols_in_combo = df_tmp.groupby('input_name')['sols_in_combo_num'].transform('max')
                df_tmp=df_tmp[df_tmp['sols_in_combo_num']==max_sols_in_combo]
                tp+=len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp+= 1 - len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
            if tp==0 and fp==0:
                continue
            res_df.loc[len(res_df)]=[pid,total,tp,fp]

    
    res_df['TP_rate']=res_df['TP']/res_df['total']
    res_df['FP_rate']=res_df['FP']/res_df['total']
    res_df['precision']=res_df['TP']/(res_df['TP']+res_df['FP'])
    return res_df


def ep_get_ref_df(json_path:str):
    df = pd.DataFrame(columns=['task_id', 'input_name', 'ref_out', 'input_valid'])
    with open(json_path,"r") as f:
        ref_json=json.load(f)
    for task_key in ref_json:
        this_task_json=ref_json[task_key]
        task_id=task_key.replace('/','_')
        inp_len=len(this_task_json)
        new_rows=[]
        for i in range(inp_len):
            #print(f"{task_id} input_{i}")
            valid=this_task_json[i][0]
            output=this_task_json[i][1]
            if valid==True:
                new_row=[task_id,f"input_{i}",output,True]
            else:
                new_row=[task_id,f"input_{i}",None,False]
            new_rows.append(new_row)
        df_new_rows=pd.DataFrame(new_rows, columns=['task_id', 'input_name', 'ref_out', 'input_valid'])
        df=pd.concat([df,df_new_rows],ignore_index=True)
    return df

def to_hashable(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return frozenset((key, to_hashable(value)) for key, value in obj.items())
    else:
        return obj
    
def ep_get_trigger_df(df:pd.DataFrame,method_type:str):
    assert method_type in ['dfp','tc']
    df['out']=df['out'].apply(to_hashable)
    result_df=pd.DataFrame(columns=['task_id', 'input_name','out','sol_names','input_valid','out_correct'])
    A=df[(df['sol_name']=='put')]
    B=df[~(df['sol_name']=='put')]
    A_outputs=A[['task_id', 'input_name', 'out']].drop_duplicates()
    B_grouped = (B.groupby(['task_id', 'input_name']))
    totoal_len=len(B_grouped)
    count=0
    for (task_id, input_name), group_df in B_grouped:
        count+=1
        if count%1000==0:
            print(f"get_triger: {count}/{totoal_len}")
        group_df=group_df.drop_duplicates(subset=['task_id','input_name','sol_name','out'])
        out_values = group_df['out'].values
        unique_out_values = set(out_values) 
        for out_value in unique_out_values:
            a_out_df=A_outputs[(A_outputs['task_id'] == task_id) & (A_outputs['input_name'] == input_name)]
            if len(a_out_df)<1 :
                continue
            a_out = A[(A['task_id'] == task_id) & (A['input_name'] == input_name)]['out'].values[0]
            if out_value==a_out:
                continue
            matching_sol_names = group_df[group_df['out']==out_value]['sol_name'].to_list()
            input_valid=group_df[group_df['out']==out_value]['input_valid'].to_list()[0]
            out_correct=group_df[group_df['out']==out_value]['out_correct'].to_list()[0]
            if method_type=='dfp' and len(matching_sol_names) < 2 :
                continue
            matching_sol_names=tuple(matching_sol_names)
            result_df.loc[len(result_df)] = [task_id,input_name,out_value,matching_sol_names,input_valid,out_correct]
    result_df['len_sol_names']=result_df['sol_names'].apply(lambda x:len(x))
    result_df['final_valid']= ( (result_df['input_valid'] == True) & (result_df['out_correct']) )
    result_df['total_sols_num']=result_df['task_id'].apply(lambda x:len(df[df['task_id']==x].drop_duplicates(subset=['sol_name']))-1)
    for i in range(10):
        column_name=f"num{i}"
        result_df[column_name] = result_df['sol_names'].apply(lambda x: 1 if f"sol{i}" in x else 0)
    return result_df


def ep_compute_res(df_triger:pd.DataFrame,num_of_ref_progs:int,method_type:str):
    assert method_type in ['dfp','tc']
    res_df=pd.DataFrame(columns=['task_id','total','TP','FP'])
    ori_n=num_of_ref_progs
    for task_id in df_triger['task_id'].unique():
        num_of_ref_progs=ori_n
        df_triger_pid=df_triger[df_triger['task_id']==task_id].copy()
        sols_set = set()
        df_triger_pid['sol_names'].apply(lambda x: sols_set.update( [f"num{sol_name[-1]}" for sol_name in x] ))
        sols_list=sorted(list(sols_set))
        total_sols_num=df_triger_pid['total_sols_num'].max()
        if total_sols_num<num_of_ref_progs:
            num_of_ref_progs=total_sols_num
        total=Cnk(total_sols_num,num_of_ref_progs)
        all_sols_name=['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9']
        out_sols_num=total_sols_num-len(sols_list)
        if out_sols_num!=0:
            out_sols_name=set(all_sols_name)-set(sols_list)
            out_sols_name=list(out_sols_name)
            out_sols_name.sort()
            out_sols_list=out_sols_name[:out_sols_num]
        else:
            out_sols_list=[]    
        to_use_sols_name=sols_list+out_sols_list

        combos=list(combinations(to_use_sols_name,num_of_ref_progs))
        combos.sort()
        if method_type=='dfp':
            tp,fp=0,0
            for combo in combos:
                combo=list(combo)
                df_triger_pid['combo_all_true']=df_triger_pid[combo].all(axis=1)
                df_tmp=df_triger_pid[df_triger_pid['combo_all_true']]
                if len(df_tmp)<1:
                    continue
                tp+= len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp+= 1-len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                if fp<0:
                    print(f"NOW TP:{tp},FP:{fp}")
                    print(len(df_tmp),len(df_tmp[df_tmp['final_valid']==True]),len(df_tmp[df_tmp['final_valid']==False]))
                    
            if tp==0 and fp==0:
                continue
            res_df.loc[len(res_df)]=[task_id,total,tp,fp]
        elif method_type=='tc':
            tp,fp=0,0
            for combo in combos:
                combo=list(combo)    
                df_triger_pid['combo_any_true']=df_triger_pid[combo].any(axis=1)
                df_tmp=df_triger_pid[df_triger_pid['combo_any_true']].copy()
                if len(df_tmp)<1:
                    continue
                
                df_tmp['sols_in_combo_num'] = df_tmp.loc[:,combo].sum(axis=1)
                max_sols_in_combo = df_tmp.groupby('input_name')['sols_in_combo_num'].transform('max')
                df_tmp=df_tmp[df_tmp['sols_in_combo_num']==max_sols_in_combo]
                tp+=len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp+= 1 - len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
            if tp==0 and fp==0:
                continue
            res_df.loc[len(res_df)]=[task_id,total,tp,fp]
        else:
            raise RuntimeError(f"Wrong method_type: {method_type}")
    res_df['TP_rate']=res_df['TP']/res_df['total']
    res_df['FP_rate']=res_df['FP']/res_df['total']
    res_df['precision']=res_df['TP']/(res_df['TP']+res_df['FP'])
    return res_df


def ep_can_compute_res(df_triger:pd.DataFrame,num_of_ref_progs:int,method_type:str):
    assert method_type in ['dfp','tc']
    res_df=pd.DataFrame(columns=['task_id','total','TP','FP','FP_bad_input'])
    ori_n=num_of_ref_progs
    for task_id in df_triger['task_id'].unique():
        num_of_ref_progs=ori_n
        df_triger_pid=df_triger[df_triger['task_id']==task_id].copy()
        sols_set = set()
        df_triger_pid['sol_names'].apply(lambda x: sols_set.update( [f"num{sol_name[-1]}" for sol_name in x] ))
        sols_list=sorted(list(sols_set))
        total_sols_num=df_triger_pid['total_sols_num'].max()
        if total_sols_num<num_of_ref_progs:
            num_of_ref_progs=total_sols_num
        total=Cnk(total_sols_num,num_of_ref_progs)
        all_sols_name=['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9']
        out_sols_num=total_sols_num-len(sols_list)
        if out_sols_num!=0:
            out_sols_name=set(all_sols_name)-set(sols_list)
            out_sols_name=list(out_sols_name)
            out_sols_name.sort()
            out_sols_list=out_sols_name[:out_sols_num]
        else:
            out_sols_list=[]    
        to_use_sols_name=sols_list+out_sols_list

        combos=list(combinations(to_use_sols_name,num_of_ref_progs))
        combos.sort()
        if method_type=='dfp':
            tp,fp=0,0
            fp_bad_input=0
            for combo in combos:
                combo=list(combo)
                df_triger_pid['combo_all_true']=df_triger_pid[combo].all(axis=1)
                df_tmp=df_triger_pid[df_triger_pid['combo_all_true']]
                if len(df_tmp)<1:
                    continue
                tp+= len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp+= 1-len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp_bad_input+=len(df_tmp[df_tmp['input_valid']!=True])/len(df_tmp)
                if fp<0:
                    print(f"NOW TP:{tp},FP:{fp}")
                    print(len(df_tmp),len(df_tmp[df_tmp['final_valid']==True]),len(df_tmp[df_tmp['final_valid']==False]))
                    
            if tp==0 and fp==0:
                continue
            res_df.loc[len(res_df)]=[task_id,total,tp,fp,fp_bad_input]
        elif method_type=='tc':
            tp,fp=0,0
            fp_bad_input=0
            for combo in combos:
                combo=list(combo)    
                df_triger_pid['combo_any_true']=df_triger_pid[combo].any(axis=1)
                df_tmp=df_triger_pid[df_triger_pid['combo_any_true']].copy()
                if len(df_tmp)<1:
                    continue
                df_tmp['sols_in_combo_num'] = df_tmp.loc[:,combo].sum(axis=1)
                max_sols_in_combo = df_tmp.groupby('input_name')['sols_in_combo_num'].transform('max')
                df_tmp=df_tmp[df_tmp['sols_in_combo_num']==max_sols_in_combo]
                tp+=len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp+= 1 - len(df_tmp[df_tmp['final_valid']==True])/len(df_tmp)
                fp_bad_input+=len(df_tmp[df_tmp['input_valid']!=True])/len(df_tmp)
            if tp==0 and fp==0:
                continue
            res_df.loc[len(res_df)]=[task_id,total,tp,fp,fp_bad_input]
        else:
            raise RuntimeError(f"Wrong method_type: {method_type}")
    res_df['TP_rate']=res_df['TP']/res_df['total']
    res_df['FP_rate']=res_df['FP']/res_df['total']
    res_df['precision']=res_df['TP']/(res_df['TP']+res_df['FP'])
    res_df['FP_bad_input_rate']=res_df['FP_bad_input']/res_df['total']
    return res_df

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # 1. TrickyBugs 데이터 로드
    # ----------------------------
    base_path = "/local_datasets/a2024105535/TrickCatcher/Datasets/TrickyBugs"
    problems = load_trickybugs_data_revised(base_path, lang="cpp")

    # ----------------------------
    # 2. 변형 코드 생성
    # ----------------------------
    output_base_path = "/local_datasets/a2024105535/TrickCatcher/Outputs"
    num_variants = 3

    create_variants(problems, output_base_path, k=num_variants)

    '''# ----------------------------
    # 3. 테스트 입력 생성
    # ----------------------------
    num_test_inputs = 10
    generate_buggy_test_inputs(problems, output_base_path, num_inputs=num_test_inputs)

    # ----------------------------
    # 4. 변형 코드 검증
    # ----------------------------
    verify_variants(problems, output_base_path)

    # ----------------------------
    # 5. 테스크 오라클 평가
    # ----------------------------
    ref_json_path = "/local_datasets/a2024105535/TrickCatcher/Datasets/TrickyBugsRef/ref_outputs.json"
    df_ref = ep_get_ref_df(ref_json_path)

    # GenProgsVerified에서 각 문제별 변형 코드 출력 수집
    variant_results = []
    inputs_base_path = Path(output_base_path) / "chat_generated_inputs"
    
    df_variants = task_oracle(problems, str(inputs_base_path), lang="cpp")

    # trigger df 생성
    df_trigger = ep_get_trigger_df(df_variants, method_type="dfp")

    # 최종 평가
    res_df = ep_compute_res(df_trigger, num_of_ref_progs=2, method_type="dfp")

    print("\n✅ TrickCatcher 평가 완료!")
    print(res_df.head())
    
    # 결과 저장
    result_save_path = Path(output_base_path) / "evaluation_results.csv"
    res_df.to_csv(result_save_path, index=False)
    print(f"📊 평가 결과 저장: {result_save_path}")'''
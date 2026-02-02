from tempfile import TemporaryDirectory
import os,time,glob,random
from typing import Dict, List
import subprocess
from pathlib import Path
from .macnet_srdd import Node

class Pool:
    def __init__(self, unit_num: int,env):
        self.unit_num = unit_num
        self.waiting_queue: Dict[str, str] = {}
        self.cycle = 0
        self.round = 0
        self.task_prompt = ""
        self.response_text = ""
        self.env=env

    def state_pool_add(self, phase_name: str, phase_prompt: str, 
                     task_prompt: str,  store_dir: str, temperature: float) -> Node:
        self.task_prompt = task_prompt
        self._collect_codes(phase_name)
        
        if len(self.waiting_queue) < 2:
            return None
        
        log_content, result = self._run_improvement_cycles(phase_prompt, phase_name, temperature)
        
        with open(store_dir, 'w') as f:
            f.write(log_content + '\n' + self.response_text)
            
        return result

    def _collect_codes(self, phase_name: str):
        for f in glob.glob(f"{phase_name}/**/solution_*", recursive=True):
            with open(f) as file:
                if (content := file.read().strip()) == 'pass': 
                    continue
                team_id = f"team_{time.time_ns()}"
                self.waiting_queue[team_id] = content

    def _run_improvement_cycles(self, phase_prompt: str, phase_name: str, temperature: float) -> tuple:
        log = []
        self.round = 0
        
        while len(self.waiting_queue) > self.unit_num:
            self.round += 1
            log.append(f"\n\n—— Round {self.round} Begin ——\n")
            
            groups = self._form_groups()
            improved = {}
            
            for group_id, codes in groups.items():
                content = self._competitive_cooperation(group_id, codes, phase_prompt, phase_name, temperature)
                log.append(content)
                improved[group_id] = self.response_text
                
            self.waiting_queue = improved
            self._prune_test()  
            
        return '\n'.join(log), self._final_process(phase_prompt, temperature)

    def _competitive_cooperation(self, team_key: str, code_list: list, phase_prompt: str, 
                                phase_name: str, temperature: float) -> str:
        self.cycle += 1
        content = [
            f"Teams {team_key} are collaborating",
            f"Phase: {phase_name} | Cycle: {self.cycle}",
            f"Prompt: {phase_prompt[:50]}...",
            "Calling LLM API..."
        ]
        
        prompt = self._build_prompt(code_list, phase_prompt)
        self.response_text=self.env.call_llm(prompt=prompt, temperature=temperature)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        content.extend([
            f"Time: {timestamp}",
            f"Response ({len(self.response_text)} chars):",
            "—— End of Cycle ——"
        ])
        
        return '\n'.join(content)

    def _build_prompt(self, code_list: List[str], phase_prompt: str) -> str:
        prompt = [phase_prompt, f"\nTask: {self.task_prompt}"]
        for idx, code in enumerate(code_list):
            prompt.append(f"\n—— Team {idx} Code ——\n{code}")
        return '\n'.join(prompt)


    def _prune_test(self):
        if len(self.waiting_queue) > 4:
            scored = {}
            for team_id, code in self.waiting_queue.items():
                with TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    (tmp_path / "main.py").write_text(code)
                    scored[team_id] = competition_filter(tmp_path)
            
            keep_num = max(1, len(scored) // 2)
            self.waiting_queue = dict(sorted(
                scored.items(), 
                key=lambda x: -x[1]
            )[:keep_num])

    def _form_groups(self) -> Dict[str, List[str]]:
        codes = list(self.waiting_queue.values())
        random.shuffle(codes)
        return {f"Group_{i}": codes[i:i+self.unit_num] 
               for i in range(0, len(codes), self.unit_num)}
               
    def _final_process(self) -> Node:
        final_code = '\n\n'.join(self.waiting_queue.values())
        node = Node()
        node.code_process(final_code)
        return node

def competition_filter(dir_program: Path) -> float:
    def _get_code_files(path: Path) -> list:
        return list(path.rglob("*.py"))

    def _read_all_code(path: Path) -> str:
        return "\n".join(f.read_text() for f in _get_code_files(path))

    def _get_completeness(code: str) -> float:
        forbidden = {"password", "passenger", "passed", "passes"}
        code_lower = code.lower()
        has_pass = any(w in code_lower for w in {"pass", "todo"})
        has_forbidden = any(w in code_lower for w in forbidden)
        return 0.0 if has_pass and not has_forbidden else 1.0

    def _get_executability(main_file: Path) -> float:
        if not main_file.exists():
            return 0.0
        try:
            result = subprocess.run(
                ["python3", str(main_file)],
                cwd=main_file.parent,
                timeout=10,
                capture_output=True,
                text=True
            )
            return 1.0 if result.returncode == 0 else 0.0
        except Exception:
            return 0.0

    main_file = dir_program / "main.py"
    code_content = _read_all_code(dir_program)
    
    completeness = _get_completeness(code_content)
    executability = _get_executability(main_file)
    
    return (executability * 10 + completeness) * 10000
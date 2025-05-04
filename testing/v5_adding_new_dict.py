"""
개선된 프롬프트로 LLM 응답 품질 향상
"""

import json
from functools import lru_cache
from transformers import pipeline
import re
from datetime import datetime
import os

class ImprovedConceptExpander:
    """LLM 프롬프트와 파싱 로직을 개선한 개념 확장기"""
    
    def __init__(self, 
                 use_llm=True, 
                 llm_model_name="google/flan-t5-base",
                 save_path="improved_concepts.json"):
        """
        초기화
        """
        # 기본 딕셔너리에서 시작
        self.related_concepts = {
            "강아지": ["개", "반려동물", "반려견", "멍멍이", "puppy", "dog", "토이푸들", "푸들"],
            "생일": ["생신", "축하", "축하하다", "birthday", "밤파티", "생신축하", "생일잔치"],
            "체중": ["무게", "몸무게", "키로", "파운드", "weight", "kg", "몸매", "비만", "다이어트"],
            "음식": ["먹다", "식사", "요리", "음식점", "맛집", "food", "meal", "restaurant"],
            "반려동물": ["강아지", "개", "고양이", "펫", "pet", "pet-care", "애완동물", "반려견"]
        }
        
        self.save_path = save_path
        
        # 이전에 학습한 개념이 있으면 로드
        self._load_learned_concepts()
        
        # 역방향 인덱스 구축
        self.reverse_index = {}
        self._build_reverse_index()
        
        # LLM 설정
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm = pipeline("text2text-generation", model=llm_model_name)
                print(f"LLM 로드 성공: {llm_model_name}")
            except Exception as e:
                print(f"LLM 로드 실패: {e}, 규칙 기반만 사용")
                self.use_llm = False
        
        # 학습 통계
        self.stats = {
            "total_concepts": len(self.related_concepts),
            "llm_calls": 0,
            "concepts_learned": 0,
            "successful_learnings": 0,
            "failed_learnings": 0,
            "last_learning_date": None
        }
    
    def _load_learned_concepts(self):
        """이전에 학습한 개념들을 파일에서 로드"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    learned_concepts = data.get('concepts', {})
                    # 기본 딕셔너리에 학습된 개념 추가
                    for key, values in learned_concepts.items():
                        if key not in self.related_concepts:
                            self.related_concepts[key] = list(values)
                        else:
                            # 기존 개념에 새로운 값 추가
                            existing_values = set(self.related_concepts[key])
                            existing_values.update(values)
                            self.related_concepts[key] = list(existing_values)
                    print(f"학습된 개념 로드 완료: {len(learned_concepts)}개 항목")
            except Exception as e:
                print(f"학습된 개념 로드 실패: {e}")
    
    def _save_concepts(self):
        """현재 딕셔너리를 파일에 저장"""
        try:
            data = {
                "concepts": self.related_concepts,
                "stats": self.stats,
                "last_saved": datetime.now().isoformat()
            }
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"개념 저장 완료: {self.save_path}")
        except Exception as e:
            print(f"개념 저장 실패: {e}")
    
    def _build_reverse_index(self):
        """역방향 인덱스 구성"""
        self.reverse_index = {}
        for main_concept, related_terms in self.related_concepts.items():
            # 메인 개념 인덱싱
            if main_concept not in self.reverse_index:
                self.reverse_index[main_concept] = set()
            self.reverse_index[main_concept].add(main_concept)
            self.reverse_index[main_concept].update(related_terms)
            
            # 관련 개념 인덱싱
            for term in related_terms:
                if term not in self.reverse_index:
                    self.reverse_index[term] = set()
                self.reverse_index[term].add(main_concept)
                self.reverse_index[term].update(related_terms)
    
    def _clean_llm_response(self, response):
        """LLM 응답 정제"""
        # 응답에서 따옴표나 특수 문자 제거
        cleaned = re.sub(r'[""''\(\)\[\]\{\}]', '', response)
        # 여러 개의 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # "is a", 숫자 등 무의미한 단어 제거
        cleaned = re.sub(r'\b(is\s+a|the|and|or|of|to|in|a|\d+\s*[-–]\s*\d+)\b', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()
    
    def _extract_concepts_from_llm(self, trigger_word, context=None):
        """LLM을 통한 개념 추출 - 개선된 프롬프트"""
        # 더 명확한 프롬프트
        prompt = f"한글 관련 단어: {trigger_word}\n관련된 비슷한 다른 단어들:\n"
        
        try:
            outputs = self.llm(prompt, max_length=50, num_beams=2)
            response = outputs[0]["generated_text"]
            
            # 응답 정제
            cleaned_response = self._clean_llm_response(response)
            
            # 단어 추출 (쉼표, 공백, 줄바꿈 등으로 구분)
            words = re.split(r'[,\s]+|[.]|[･]', cleaned_response)
            
            concepts = set()
            for word in words:
                word = word.strip()
                if (word and 
                    len(word) > 1 and 
                    word != trigger_word and
                    not word.isdigit() and
                    not word.isalpha() == False or word in ['개', '생일', '케이크']):  # 한글 단어 포함
                    concepts.add(word)
            
            self.stats["llm_calls"] += 1
            
            print(f"[LLM 원본] '{trigger_word}' → {repr(response)}")
            print(f"[LLM 정제] '{trigger_word}' → {concepts}")
            
            return concepts
        except Exception as e:
            print(f"LLM 처리 오류: {e}")
            return set()
    
    def learn_concepts(self, trigger_word, new_concepts):
        """새로운 개념들을 학습하여 딕셔너리에 추가"""
        if not new_concepts:
            self.stats["failed_learnings"] += 1
            return
            
        if trigger_word not in self.related_concepts:
            self.related_concepts[trigger_word] = []
        
        # 기존 개념 리스트에 새로운 개념 추가
        existing_concepts = set(self.related_concepts[trigger_word])
        new_unique_concepts = set(new_concepts) - existing_concepts - {trigger_word}
        
        if new_unique_concepts:
            existing_concepts.update(new_unique_concepts)
            self.related_concepts[trigger_word] = list(existing_concepts)
            
            # 역방향 인덱스 재구축
            self._build_reverse_index()
            
            # 학습 통계 업데이트
            self.stats["concepts_learned"] += len(new_unique_concepts)
            self.stats["successful_learnings"] += 1
            self.stats["last_learning_date"] = datetime.now().isoformat()
            
            # 파일에 저장
            self._save_concepts()
            
            print(f"[학습 성공] '{trigger_word}'에 {len(new_unique_concepts)}개 개념 추가: {new_unique_concepts}")
        else:
            print(f"[학습 실패] '{trigger_word}': 유효한 새 개념 없음")
            self.stats["failed_learnings"] += 1
    
    @lru_cache(maxsize=128)
    def expand_concept(self, trigger_word, context=None, max_concepts=10):
        """트리거 단어에 대한 개념 확장"""
        concepts = set()
        
        # 1. 규칙 기반 검색
        if trigger_word in self.reverse_index:
            concepts.update(self.reverse_index[trigger_word])
            print(f"[규칙] '{trigger_word}' → {len(concepts)}개 개념 발견")
        
        # 2. LLM을 통한 추가 개념 검색 (규칙에 없거나 개념이 부족한 경우)
        if self.use_llm and (len(concepts) == 0 or len(concepts) < 3):
            llm_concepts = self._extract_concepts_from_llm(trigger_word, context)
            
            if llm_concepts:
                # 학습하여 다음에 사용
                self.learn_concepts(trigger_word, llm_concepts)
                concepts.update(llm_concepts)
        
        # 3. 최소한 자기 자신은 포함
        if not concepts:
            concepts.add(trigger_word)
        
        # 4. 최대 개념 수 제한
        return set(list(concepts)[:max_concepts])
    
    def get_stats(self):
        """학습 통계 반환"""
        self.stats["total_concepts"] = len(self.related_concepts)
        return self.stats.copy()

# 시연
def demonstrate_improved_learning():
    """개선된 학습형 개념 확장기 데모"""
    
    print("=== 개선된 학습형 개념 확장기 데모 ===")
    expander = ImprovedConceptExpander(save_path="improved_concepts.json")
    
    # 초기 통계
    print(f"초기 상태: {expander.get_stats()}")
    
    print("\n=== 개념 확장 테스트 ===")
    test_words = ["케이크", "컴퓨터", "코딩", "강아지", "맛있는것"]
    
    for word in test_words:
        print(f"\n-----------------")
        print(f"'{word}' 개념 확장:")
        concepts = expander.expand_concept(word)
        print(f"최종 결과: {concepts}")
    
    print("\n=== 최종 통계 ===")
    print(f"통계: {expander.get_stats()}")

if __name__ == "__main__":
    demonstrate_improved_learning()
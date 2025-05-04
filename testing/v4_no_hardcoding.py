"""
하이브리드 개념 확장기 - LLM과 규칙 기반 통합
"""

import json
from functools import lru_cache
from transformers import pipeline
import re

class HybridConceptExpander:
    """규칙 기반과 LLM을 결합한 하이브리드 개념 확장기"""
    
    def __init__(self, use_llm=True, llm_model_name="google/flan-t5-base"):
        """
        초기화
        
        Args:
            use_llm: LLM 사용 여부
            llm_model_name: 사용할 LLM 모델명
        """
        # 기존 규칙 기반 딕셔너리
        self.related_concepts = {
            "강아지": ["개", "반려동물", "반려견", "멍멍이", "puppy", "dog", "토이푸들", "푸들"],
            "생일": ["생신", "축하", "축하하다", "birthday", "밤파티", "생신축하", "생일잔치"],
            "체중": ["무게", "몸무게", "키로", "파운드", "weight", "kg", "몸매", "비만", "다이어트"],
            "음식": ["먹다", "식사", "요리", "음식점", "맛집", "food", "meal", "restaurant"],
            "반려동물": ["강아지", "개", "고양이", "펫", "pet", "pet-care", "애완동물", "반려견", "반려묘"]
        }
        
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
    
    def _build_reverse_index(self):
        """역방향 인덱스 구성"""
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
    
    def _parse_llm_response_fallback(self, response):
        """LLM 응답을 파싱하는 대체 방법들"""
        
        # 방법 1: 괄호 안의 내용 추출
        match = re.search(r'\[(.*?)\]', response)
        if match:
            content = match.group(1)
            # 따옴표로 구분된 단어들 추출
            words = re.findall(r'"([^"]*)"', content)
            if words:
                return set(words)
        
        # 방법 2: 쉼표로 구분된 직접 파싱
        words = response.split(',')
        cleaned_words = []
        for word in words:
            cleaned = re.sub(r'[^\w가-힣 ]', '', word).strip()
            if cleaned:
                cleaned_words.append(cleaned)
        
        return set(cleaned_words) if cleaned_words else set()
    
    def _extract_concepts_from_llm(self, trigger_word, context=None):
        """LLM을 통한 개념 추출 (프롬프트 개선 버전)"""
        # 더 단순한 프롬프트로 변경
        prompt = f"관련 단어들을 한글로 대답해주세요.\n"
        prompt += f"주제: {trigger_word}\n"
        if context:
            prompt += f"상황: {context}\n"
        prompt += "답변 예시: 강아지, 개, 반려견"
        
        try:
            outputs = self.llm(prompt, max_length=100, num_beams=2)
            response = outputs[0]["generated_text"]
            
            # 응답에서 단어 추출
            words = response.split(',')
            return set([word.strip() for word in words if word.strip()])
        except Exception as e:
            print(f"LLM 처리 오류: {e}")
            return set()
    
    @lru_cache(maxsize=128)
    def expand_concept(self, trigger_word, context=None, max_concepts=10):
        """
        트리거 단어에 대한 개념 확장
        
        Args:
            trigger_word: 확장할 기본 트리거 단어
            context: 문맥 정보
            max_concepts: 반환할 최대 개념 수
            
        Returns:
            set: 관련 개념들의 집합
        """
        concepts = set()
        
        # 1. 규칙 기반 검색
        if trigger_word in self.reverse_index:
            concepts.update(self.reverse_index[trigger_word])
        
        # 2. LLM을 통한 추가 개념 검색 (활성화된 경우)
        if self.use_llm and len(concepts) == 0:
            llm_concepts = self._extract_concepts_from_llm(trigger_word, context)
            concepts.update(llm_concepts)
        
        # 3. 최소한 자기 자신은 포함
        if not concepts:
            concepts.add(trigger_word)
        
        # 4. 최대 개념 수 제한
        return set(list(concepts)[:max_concepts])

# 사용 예시
def test_hybrid_expander():
    """하이브리드 확장기 테스트"""
    
    # 1. LLM 없이 사용
    print("=== 규칙 기반만 사용 ===")
    expander_no_llm = HybridConceptExpander(use_llm=False)
    
    test_words = ["강아지", "개", "생일", "케이크", "컴퓨터"]
    for word in test_words:
        concepts = expander_no_llm.expand_concept(word)
        print(f"'{word}' 확장 결과: {concepts}")
    
    # 2. LLM 포함 사용
    print("\n=== LLM + 규칙 기반 ===")
    expander_with_llm = HybridConceptExpander(use_llm=True)
    
    for word in test_words:
        concepts = expander_with_llm.expand_concept(word)
        print(f"'{word}' 확장 결과: {concepts}")

if __name__ == "__main__":
    test_hybrid_expander()
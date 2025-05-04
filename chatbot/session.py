"""
대화 세션 관리 모듈
사용자별 대화 컨텍스트 관리
"""
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class ConversationSession:
    """
    대화 세션 클래스
    
    기능:
    - 대화 컨텍스트 유지
    - 활성 개념 추적
    - 세션 통계 관리
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        
        # 대화 기록 (제한된 크기)
        self.context_window = deque(maxlen=config.get('context_window_size', 10))
        
        # 활성 개념
        self.active_concepts = set()
        self.concept_activation_time = {}
        
        # 세션 정보
        self.session_start = datetime.now()
        self.last_interaction = datetime.now()
        self.interaction_count = 0
        
        # 사용자 선호도 및 패턴
        self.user_preferences = {}
        self.interaction_patterns = {
            'frequent_topics': {},
            'question_types': {},
            'response_feedback': []
        }
    
    def update(self, user_input: str, assistant_reply: str, concepts: List[str]) -> None:
        """세션 업데이트"""
        current_time = datetime.now()
        
        # 대화 컨텍스트 추가
        self.context_window.append({
            'user': user_input,
            'assistant': assistant_reply,
            'concepts': concepts,
            'timestamp': current_time
        })
        
        # 활성 개념 업데이트
        self._update_active_concepts(concepts, current_time)
        
        # 세션 통계 업데이트
        self.last_interaction = current_time
        self.interaction_count += 1
        
        # 상호작용 패턴 분석
        self._analyze_interaction_pattern(user_input)
    
    def _update_active_concepts(self, concepts: List[str], timestamp: datetime) -> None:
        """활성 개념 업데이트"""
        for concept in concepts:
            self.active_concepts.add(concept)
            self.concept_activation_time[concept] = timestamp
        
        # 오래된 개념 정리
        inactive_concepts = []
        for concept, activation_time in self.concept_activation_time.items():
            if timestamp - activation_time > timedelta(minutes=30):
                inactive_concepts.append(concept)
        
        for concept in inactive_concepts:
            self.active_concepts.discard(concept)
            del self.concept_activation_time[concept]
    
    def _analyze_interaction_pattern(self, user_input: str) -> None:
        """상호작용 패턴 분석"""
        # 주제 분석 (간단한 구현)
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:  # 단어 길이 필터
                self.interaction_patterns['frequent_topics'][word] = \
                    self.interaction_patterns['frequent_topics'].get(word, 0) + 1
        
        # 질문 유형 분석
        if '?' in user_input:
            if any(word in user_input for word in ['누구', '뭐', '언제', '어디', '왜', '어떻게']):
                self.interaction_patterns['question_types']['interrogative'] = \
                    self.interaction_patterns['question_types'].get('interrogative', 0) + 1
            else:
                self.interaction_patterns['question_types']['confirmation'] = \
                    self.interaction_patterns['question_types'].get('confirmation', 0) + 1
    
    def get_context(self) -> Dict[str, Any]:
        """현재 컨텍스트 반환"""
        return {
            'recent_utterances': list(self.context_window),
            'active_concepts': list(self.active_concepts),
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'interaction_count': self.interaction_count,
            'frequent_topics': dict(sorted(
                self.interaction_patterns['frequent_topics'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }
    
    def is_inactive(self, threshold: timedelta = timedelta(hours=1)) -> bool:
        """세션 비활성화 확인"""
        return datetime.now() - self.last_interaction > threshold
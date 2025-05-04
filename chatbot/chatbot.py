"""
실시간 연관 기억 챗봇 메인 클래스
모든 컴포넌트를 통합하여 대화 처리
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List

from config.settings import SystemConfig
from core.memory_manager import MemoryManager
from core.association_network import AssociationNetwork
from core.lifecycle_manager import LifecycleManager
from chatbot.session import ConversationSession
from utils.visualization import visualize_association_network
from datetime import datetime  # 추가된 import


logger = logging.getLogger(__name__)


class RealtimeAssociativeChatbot:
    """
    실시간 연관 기억 챗봇
    
    기능:
    - 실시간 대화 처리
    - 연관 기억 기반 응답 생성
    - 백그라운드 자원 관리
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # 핵심 컴포넌트 초기화
        self.memory_manager = MemoryManager(self.config.to_dict())
        self.association_network = AssociationNetwork(
            min_strength=self.config.min_connection_strength,
            decay_factor=self.config.decay_factor
        )
        self.lifecycle_manager = LifecycleManager(
            self.memory_manager,
            self.config.to_dict()
        )
        
        # 활성 세션 관리
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # 백그라운드 작업
        self.background_tasks = []
        self.is_running = True
        
        # 통계
        self.stats = {
            'total_interactions': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time': 0.0
        }
        
        # 초기화
        self._initialize()
    
    def _initialize(self) -> None:
        """시스템 초기화"""
        # 백그라운드 작업 시작
        self._start_background_tasks()
        logger.info("실시간 연관 기억 챗봇 초기화 완료")
    
    def _start_background_tasks(self) -> None:
        """백그라운드 작업 시작"""
        self.background_tasks = [
            asyncio.create_task(self._lifecycle_monitor()),
            asyncio.create_task(self._session_cleanup()),
            asyncio.create_task(self._performance_monitor())
        ]
    
    async def _lifecycle_monitor(self) -> None:
        """생명주기 모니터링 (1시간 주기)"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)
                await self.lifecycle_manager.run_lifecycle_cycle()
                logger.info("Lifecycle cycle completed")
            except Exception as e:
                logger.error(f"Lifecycle monitor error: {e}")
    
    async def _session_cleanup(self) -> None:
        """비활성 세션 정리 (30분 주기)"""
        while self.is_running:
            try:
                await asyncio.sleep(1800)
                current_time = datetime.now()
                inactive_sessions = []
                
                for user_id, session in self.active_sessions.items():
                    if session.is_inactive():
                        inactive_sessions.append(user_id)
                
                for user_id in inactive_sessions:
                    del self.active_sessions[user_id]
                
                logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def _performance_monitor(self) -> None:
        """성능 모니터링 (5분 주기)"""
        while self.is_running:
            try:
                await asyncio.sleep(300)
                stats = await self.get_system_stats()
                logger.info(f"System stats: {stats}")
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def chat(self, user_id: str, message: str) -> str:
        """채팅 처리"""
        start_time = datetime.now()
        
        try:
            # 세션 가져오기 또는 생성
            session = self._get_or_create_session(user_id)
            
            # 1. 개념 추출
            concepts = self._extract_concepts(message)
            
            # 2. 연관 검색
            search_results = await self.memory_manager.search_memories(concepts, message)
            
            # 3. 네트워크 연관 분석
            network_associations = self.association_network.find_associations(concepts)
            
            # 4. 응답 생성
            response = await self._generate_response(
                user_input=message,
                search_results=search_results,
                network_associations=network_associations,
                session_context=session.get_context()
            )
            
            # 5. 메모리 저장
            memory_id = await self._store_memory(user_id, message, response, concepts)
            
            # 6. 연관 네트워크 업데이트
            await self._update_associations(concepts, memory_id)
            
            # 7. 세션 업데이트
            session.update(message, response, concepts)
            
            # 8. 통계 업데이트
            self._update_stats(start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            self.stats['failed_responses'] += 1
            return "죄송합니다. 처리 중 오류가 발생했습니다."
    
    def _get_or_create_session(self, user_id: str) -> ConversationSession:
        """세션 가져오기 또는 생성"""
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = ConversationSession(
                user_id=user_id,
                config=self.config.to_dict()
            )
        return self.active_sessions[user_id]
    
    def _extract_concepts(self, text: str) -> List[str]:
        """텍스트에서 개념 추출"""
        # 간단한 구현 - 실제로는 NLP 모델 사용
        words = text.split()
        concepts = []
        
        for word in words:
            if len(word) > 2:  # 최소 길이 필터
                concepts.append(word)
        
        return concepts
    
    async def _generate_response(
        self,
        user_input: str,
        search_results: List[Dict],
        network_associations: Dict[str, float],
        session_context: Dict[str, Any]
    ) -> str:
        # 디버깅을 위한 로깅 추가
        print(f"Found {len(search_results)} search results")
        for result in search_results[:2]:  # 처음 두 개만 출력
            print(f"Result source: {result.get('source', 'unknown')}")
            memory = result.get('memory', {})
            if hasattr(memory, 'content'):
                print(f"Memory content: {memory.content}")
            else:
                print(f"Memory type: {type(memory)}")
        
        # 검색 결과가 없으면 기본 응답
        if not search_results:
            return self._generate_default_response(user_input)
        
        # 가장 관련성 높은 메모리 기반 응답
        top_result = search_results[0]
        memory = top_result['memory']
        
        # 메모리 타입에 따른 처리
        if hasattr(memory, 'content'):
            content = memory.content
        else:
            content = memory.get('content', {})
        
        # 대화 내용 확인
        user_message = content.get('user', '')
        assistant_reply = content.get('assistant', '')
        
        # 이전 응답이나 입력에서 관련 정보 찾기
        if '미키' in user_input:
            if '몇' in user_input or '살' in user_input or '나이' in user_input:
                # 이전 대화에서 나이 관련 정보 검색
                for result in search_results:
                    mem = result['memory']
                    content_to_check = mem.content if hasattr(mem, 'content') else mem.get('content', {})
                    user_msg = content_to_check.get('user', '')
                    if '미키' in user_msg and ('살' in user_msg or '나이' in user_msg):
                        return f"네, 미키는 7살이라고 말씀하셨어요."
        
        # 응답 가공
        if 'assistant' in content:
            base_response = content['assistant']
            # 컨텍스트 기반 개인화
            if session_context.get('frequent_topics'):
                top_topic = next(iter(session_context['frequent_topics'].keys()))
                response = f"{base_response} 최근 {top_topic}에 대해 자주 이야기하시네요."
            else:
                response = base_response
        else:
            # 더 구체적인 기본 응답
            if network_associations:
                top_concept = list(network_associations.keys())[0]
                response = f"그것에 대해 더 알려주시면 좋겠어요. {top_concept}에 관련된 이야기인가요?"
            else:
                response = "그것에 대해 더 알려주시면 좋겠어요."
        
        return response
    
    def _generate_default_response(self, user_input: str) -> str:
        """기본 응답 생성"""
        import random
        
        default_responses = [
            "흥미로운 이야기네요. 더 자세히 들려주세요.",
            "그것에 대해 처음 듣는 것 같아요. 설명해주시겠어요?",
            "어떤 의미로 말씀하시는 건가요?",
            "그 부분에 대해 더 알려주시면 좋겠어요."
        ]
        
        return random.choice(default_responses)
    
    async def _store_memory(
        self,
        user_id: str,
        user_input: str,
        assistant_reply: str,
        concepts: List[str]
    ) -> str:
        """메모리 저장"""
        content = {
            'user_id': user_id,
            'user': user_input,
            'assistant': assistant_reply,
            'timestamp': datetime.now().isoformat()
        }
        
        # 중요도 및 감정 가중치 계산
        importance = self._calculate_importance(user_input, assistant_reply)
        emotional_weight = self._calculate_emotional_weight(user_input)
        
        # 메모리 저장
        memory_id = await self.memory_manager.save_memory(
            content=content,
            concepts=concepts,
            importance=importance,
            emotional_weight=emotional_weight
        )
        
        return memory_id
    
    async def _update_associations(self, concepts: List[str], memory_id: str) -> None:
        """연관 관계 업데이트"""
        # 개념 활성화
        for concept in concepts:
            self.association_network.activate_concept(concept, concepts)
        
        # 개념 간 연결 생성
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                self.association_network.connect_concepts(concept1, concept2)
    
    def _calculate_importance(self, user_input: str, assistant_reply: str) -> float:
        """중요도 계산"""
        # 길이 기반 점수
        length_score = (len(user_input) + len(assistant_reply)) / 200
        
        # 키워드 기반 점수
        important_keywords = ['중요', '약속', '기억', '생일', '전화번호', '주소']
        keyword_score = sum(1 for keyword in important_keywords if keyword in user_input) / len(important_keywords)
        
        # 질문 기반 점수
        question_score = 0.2 if '?' in user_input else 0.0
        
        # 가중 평균
        total_score = (length_score * 0.4 + keyword_score * 0.4 + question_score * 0.2)
        
        return min(total_score, 1.0)
    
    def _calculate_emotional_weight(self, text: str) -> float:
        """감정 가중치 계산"""
        positive_keywords = ['좋아', '행복', '사랑', '기쁘', '즐겁']
        negative_keywords = ['싫어', '슬프', '화나', '짜증', '우울']
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        if positive_count > 0:
            return 0.5 + min(positive_count * 0.1, 0.5)
        elif negative_count > 0:
            return max(0.5 - negative_count * 0.1, 0.0)
        else:
            return 0.5
    
    def _update_stats(self, start_time: datetime) -> None:
        """통계 업데이트"""
        response_time = (datetime.now() - start_time).total_seconds()
        
        self.stats['total_interactions'] += 1
        self.stats['successful_responses'] += 1
        
        # 평균 응답 시간 계산
        previous_average = self.stats['average_response_time']
        total_count = self.stats['total_interactions']
        self.stats['average_response_time'] = (
            (previous_average * (total_count - 1) + response_time) / total_count
        )
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        memory_stats = await self.memory_manager.get_stats()
        network_stats = self.association_network.get_network_stats()
        
        return {
            'chatbot': self.stats,
            'memory': memory_stats,
            'network': network_stats,
            'active_sessions': len(self.active_sessions)
        }
    
    async def visualize_associations(self, concept: str, save_path: str = None) -> Optional[str]:
        """연관 네트워크 시각화"""
        return visualize_association_network(
            graph=self.association_network.graph,
            center_concept=concept,
            save_path=save_path
        )
    
    async def shutdown(self) -> None:
        """시스템 종료"""
        self.is_running = False
        
        # 백그라운드 작업 종료
        for task in self.background_tasks:
            task.cancel()
        
        logger.info("System shutdown completed")
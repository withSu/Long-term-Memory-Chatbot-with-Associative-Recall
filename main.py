"""
실시간 연관 기억 챗봇 실행 예제
전체 시스템을 통합하여 실행
"""
import asyncio
import logging
import json
from datetime import datetime

from config.settings import SystemConfig
from chatbot.chatbot import RealtimeAssociativeChatbot


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def run_demo():
    """데모 실행"""
    # 설정 로드
    config = SystemConfig()
    
    # 챗봇 초기화
    chatbot = RealtimeAssociativeChatbot(config)
    
    print("=== 실시간 연관 기억 챗봇 데모 ===")
    print("마지막 문장에 '종료'를 입력하시면 시스템이 종료됩니다.")
    print("='시각화 <개념>'을 입력하시면 해당 개념의 연관 네트워크를 시각화합니다.")
    print("-" * 50)
    
    # 테스트 사용자 ID
    user_id = "demo_user"
    
    # 대화 루프
    try:
        while True:
            # 사용자 입력 받기
            user_input = input("\n사용자: ")
            
            # 종료 명령 처리
            if user_input.lower() == '종료':
                print("\n채팅을 종료합니다.")
                break
            
            # 시각화 명령 처리
            if user_input.startswith('시각화 '):
                concept = user_input.replace('시각화 ', '').strip()
                timestamp = int(datetime.now().timestamp())
                save_path = f"visualization_{concept}_{timestamp}.png"
                
                result = await chatbot.visualize_associations(concept, save_path)
                if result:
                    print(f"챗봇: 연관 네트워크를 {save_path}에 저장했습니다.")
                else:
                    print(f"챗봇: '{concept}'에 대한 연관 정보를 찾을 수 없습니다.")
                continue
            
            # 응답 생성
            response = await chatbot.chat(user_id, user_input)
            print(f"챗봇: {response}")
            
            # 주기적 통계 출력 (10번째 상호작용마다)
            stats = await chatbot.get_system_stats()
            if stats['chatbot']['total_interactions'] % 10 == 0:
                print("\n[시스템 통계]")
                print(f"- 총 상호작용: {stats['chatbot']['total_interactions']}회")
                print(f"- 평균 응답 시간: {stats['chatbot']['average_response_time']:.3f}초")
                print(f"- 활성 세션: {stats['active_sessions']}개")
                print("-" * 30)
    
    except KeyboardInterrupt:
        print("\n\n시스템을 종료합니다...")
    
    finally:
        # 시스템 종료
        await chatbot.shutdown()
        
        # 최종 통계 출력
        final_stats = await chatbot.get_system_stats()
        print("\n=== 최종 시스템 통계 ===")
        print(json.dumps(final_stats, indent=2, ensure_ascii=False))


async def run_integration_test():
    """통합 테스트 실행"""
    config = SystemConfig()
    chatbot = RealtimeAssociativeChatbot(config)
    
    # 테스트 시나리오
    test_scenarios = [
        # 기본 소개 대화
        ("안녕하세요!", "안녕하세요! 만나서 반가워요."),
        ("제 이름은 김철수입니다.", "김철수님, 반가워요!"),
        
        # 반려동물 정보 입력
        ("저는 강아지 한 마리를 키우고 있어요.", "강아지를 키우고 계시는군요!"),
        ("강아지 이름은 바둑이고 3살짜리 토이푸들이에요.", "바둑이라는 이름의 토이푸들이군요."),
        ("바둑이는 생선을 좋아해요.", "바둑이가 생선을 좋아하는군요."),
        ("다음 주 화요일이 바둑이 생일이에요.", "바둑이의 생일을 기억하고 있어야겠네요."),
        
        # 가족 정보 입력
        ("제 여동생도 강아지를 좋아해요.", "여동생분도 반려동물을 좋아하시는군요."),
        ("여동생 이름은 영희예요.", "영희님도 강아지를 좋아하시는군요."),
        
        # 연관 회상 테스트
        ("바둥이에 대해 뭘 기억하고 있어요?", "바둑이는 3살짜리 토이푸들이고 생선을 좋아한다고 하셨어요."),
        ("강아지 생일이 언제였죠?", "바둑이의 생일은 다음 주 화요일이라고 하셨어요."),
        ("제 여동생에 대해 뭘 알아요?", "영희님은 강아지를 좋아하시는 것 같아요."),
        
        # 복합 연관 질문
        ("생선을 좋아하는 동물이 있나요?", "바둑이가 생선을 좋아한다고 하셨어요."),
        ("강아지나 여동생 이야기 해주세요.", "바둑이에 대해 이야기하셨고, 영희님도 강아지를 좋아하신다고 하셨어요."),
    ]
    
    print("=== 통합 테스트 실행 ===")
    user_id = "test_user"
    
    for i, (user_input, expected_theme) in enumerate(test_scenarios, 1):
        print(f"\n[테스트 {i}]")
        print(f"사용자: {user_input}")
        
        response = await chatbot.chat(user_id, user_input)
        print(f"챗봇: {response}")
        
        # 약간의 지연 추가 (자연스러운 대화 흐름)
        await asyncio.sleep(0.5)
    
    # 시각화 테스트
    print("\n=== 연관 네트워크 시각화 테스트 ===")
    test_concepts = ["강아지", "바둑이", "생선", "생일"]
    
    for concept in test_concepts:
        viz_path = f"test_visualization_{concept}.png"
        result = await chatbot.visualize_associations(concept, viz_path)
        if result:
            print(f"'{concept}' 연관 네트워크 저장: {viz_path}")
    
    # 최종 시스템 상태 확인
    stats = await chatbot.get_system_stats()
    print("\n=== 테스트 완료 후 시스템 상태 ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 시스템 종료
    await chatbot.shutdown()


if __name__ == "__main__":
    # 실행 모드 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 통합 테스트 실행
        asyncio.run(run_integration_test())
    else:
        # 인터랙티브 데모 실행
        asyncio.run(run_demo())

